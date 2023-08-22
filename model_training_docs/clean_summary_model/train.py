from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk, load_metric
import logging
import sys
import argparse
import os
import torch
import tarfile
import boto3
import gc
from transformers import T5TokenizerFast


# these dependecies were missing from the training docker image so I am installing them here
# they are needed to use the T5-small tokenizer which is used in the compute_loss function.
import subprocess
subprocess.run(["pip", "install", "nltk", "rouge_score"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
 #############################################################################################################################

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    
#############################################################################################################################
    
    # rogue is a metric for summarization that measures the 
    # overlapping n-gram delta between target and infered summary 
    
    rouge_metric = load_metric("rouge") 

    # Instantiate the tokenizer used to encode our datasets
    tokenizer = T5TokenizerFast.from_pretrained('t5-small') 

    def compute_metrics(pred):
        '''
        This function computes the rouge training metric for 
        summarization models. It measures n-gram overlap between 
        target and pred. Rouge requires targets and pred to be 
        converted to strings, this is done in the list comprehensions.
        
        function and 
        added to the rouge metric batch cache. Then the garbage is 
        collected to gpu  resources.
        '''
        
        print("\n\nStarting compute_metrics()\n\n")
        
        # Decode preds w/ tokenizer
        decoded_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in pred.predictions]
        
        print("Decoded predictions with tokenizer\n\n")
        
        #decode labels w/ tokenizer
        decoded_labels = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in pred.label_ids]
        
        print("Decoded labels with tokenizer\n\n")
            
        # Compute rouge with batches. Preds are cached in batches so no args are required
        rouge_output = rouge_metric.compute(predictions= decoded_preds, references= decoded_labels)  
        
        print("Begining rouge computation\n\n")
            
        # garbage collection to free up memory --- these stored batch_decoced_preds contributed to out of memory error
        del decoded_preds, decoded_labels
        gc.collect()
        
        print("Cleaning decoded pred and labels garbage\n\n")

       
        return {
            "rouge1": rouge_output["rouge1"].mid.fmeasure,
            "rouge2": rouge_output["rouge2"].mid.fmeasure,
            "rougeL": rouge_output["rougeL"].mid.fmeasure
        }


#############################################################################################################################
    #retrieve model from s3 bucket
    def get_model(s3_model_uri):
        # Split the provided S3 URI to get the bucket and prefix
        s3_bucket, s3_prefix = s3_model_uri.replace("s3://", "").split("/", 1)

        # Ensure s3_prefix ends with a '/'
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'

        # Define local path to download model to
        local_path = "/tmp/t5-small-summarization-model"    
        
        # Ensure directory exists
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        # Create a session and download the model files
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(s3_bucket)

        # The two model filesare config.json and pytorch_model.bin.
        # We'll download them into the local directory.
        files_to_download = ["config.json", "pytorch_model.bin"]

        for file_name in files_to_download:
            s3_file_path = os.path.join(s3_prefix, file_name)
            local_file_path = os.path.join(local_path, file_name)
            bucket.download_file(s3_file_path, local_file_path)

        # Now that the files are downloaded, you can use from_pretrained to load the model.
        # We are specifying that it is for seq2seq here instead of seq classification like the RAC
        
        return AutoModelForSeq2SeqLM.from_pretrained(local_path)
    
    
    #get model 
    model_uri = "s3://hambart-training/t5-small-summarization-model/"  
    model = get_model(model_uri)
    
    gc.collect() #collect garbage on any unnecessary files from the model download
    
 #############################################################################################################################

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="no",   #I am also turning off eval strat for memory issues and will eval locally
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        save_strategy="no"  # This turns off model checkpointing. This is important!! Was running into major memory issues 
                              # By not turning this off. Depending on hardware it might not be important.
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    

    # train model
    trainer.train()
    
    print("\n\n\n\n\n\n\nThe Trainer has finished training at this point\n\n\n\n\n\n\n\n")
    
    
    ## I've removed final model eval within the training job due to memory contstrains

    # Saves the model to s3; default is /opt/ml/model which SageMaker sends to S3
    trainer.save_model(args.model_dir)
