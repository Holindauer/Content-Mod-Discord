from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, BertTokenizer, DataCollatorForTokenClassification
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import datasets
import numpy as np
import boto3
import subprocess

'''
This training script contains an adjustment from the original NER
training script in the load_tokenizer() function. It now loads in
a tokenizer that was trained on both the conll2003 and the wikipedia
set.
'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    
    #Model and Tokenizer s3 URIs
    #parser.add_argument("--tokenizer_s3_uri", type=str, default=None, help="Path to the tokenizer")
    #parser.add_argument("--model_s3_uri", type=str, default=None, help="Path to the model")


    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
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

   #####################################################################################################################################################

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")


   #####################################################################################################################################################

    
    subprocess.run(["pip", "install", "seqeval"])
    metric = datasets.load_metric("seqeval") #load in seqeval metric after install


    def compute_metrics(p): 
        '''
            this function unpacks the predictions and labels from p. Then it applies argmax to the prediction logics which converts
            them to indices within the labels_list. Then assigned to true_predictions is a list comprehension of those indices converted 
            to their label names. The true_labels list has this analogous operation performed on the label indices of the targets for 
            that example. Then the true_predictiosn and true_labels are evaluated for precision, recall, and f1 using the seqeval package.
        '''
        #NER labels specific to the conll2003 task
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'] 

        #unpack predictions
        predictions, labels = p 

        #get prediction indices for use in labels_list by argmaxing logits
        predictions = np.argmax(predictions, axis=2) 

        #prediction indicies ---> labels
        true_predictions = [ 
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100] for prediction, label in zip(predictions, labels) 
        ] 

        #Ground truth indicies ---> labels
        true_labels = [ 
            [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100] for prediction, label in zip(predictions, labels) 
        ] 

        #get score
        results = metric.compute(predictions=true_predictions, references=true_labels) 

        return { 
            "precision": results["overall_precision"], 
            "recall": results["overall_recall"], 
            "f1": results["overall_f1"], 
            "accuracy": results["overall_accuracy"], 
        } 

   #####################################################################################################################################################
    
    def load_model(bucket_name, model_path):
        
        # Define local paths to save model and tokenizer
        local_model_path = '/tmp/model/'
        
        # Create local directories
        os.makedirs(local_model_path, exist_ok=True)
        
        # Download model files from S3 to local
        s3.download_file(bucket_name, f"{model_path}pytorch_model.bin", f"{local_model_path}pytorch_model.bin")
        s3.download_file(bucket_name, f"{model_path}config.json", f"{local_model_path}config.json")
        
        # Load model and tokenizer from local paths
        return AutoModelForTokenClassification.from_pretrained(local_model_path)

    ## Here is the new load_tokenizer function ##
    def load_tokenizer(bucket_name, tokenizer_path):

        # Define local paths to save model and tokenizer
        local_tokenizer_path = '/tmp/tokenizer/'

        # Create local directories
        os.makedirs(local_tokenizer_path, exist_ok=True)

        # Download tokenizer files from S3 to local
        # Assuming s3 is an initialized boto3 client
        s3.download_file(bucket_name, 
                         f"{tokenizer_path}conll_wiki_tokenizer-vocab.txt", 
                         f"{local_tokenizer_path}conll_wiki_tokenizer-vocab.txt")

        # Load BertWordPieceTokenizer using vocab.txt
        return BertTokenizer.from_pretrained(f"{local_tokenizer_path}conll_wiki_tokenizer-vocab.txt", do_lower_case=True)

    
    # Define S3 bucket and paths
    bucket_name = 'conll2003-task'
    model_path = 'distilbert-base-uncased/' #full uri is concatenated in the 
    tokenizer_path = 'tokenizer-with-wiki/'
    
    
    #establish s3 client
    try:
        # Initialize boto3 client
        s3 = boto3.client('s3')
    except Exception as e:
        print(f"An error occurred establishing the s3 client: {e}")
    
    #load model
    try:
        model = load_model(bucket_name, model_path)
    except Exception as e:
        print(f"An error occurred when loading the model: {e}")
        
    #load model
    try:
        tokenizer = load_tokenizer(bucket_name, tokenizer_path)
    except Exception as e:
        print(f"An error occurred when loading the tokenizer: {e}")
        
    #make data collator
    try:
        data_collator = DataCollatorForTokenClassification(tokenizer)
    except Exception as e:
        print(f'an error occurred when making the data collator: {e}')
        
   #####################################################################################################################################################

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        weight_decay = 0.01, #added hardcoded regularization
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)