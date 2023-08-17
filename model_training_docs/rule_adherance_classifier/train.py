from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import tarfile
import boto3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
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

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    #retrieve model from s3 bucket
    def get_model(s3_model_uri):
        # Split the provided S3 URI to get the bucket and prefix
        s3_bucket, s3_prefix = s3_model_uri.replace("s3://", "").split("/", 1)

        # Ensure s3_prefix ends with a '/'
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'

        # Define local path to download model to
        local_path = "/tmp/distilbert-base-uncased-model"

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
        model = AutoModelForSequenceClassification.from_pretrained(local_path)
        return model

    #get model 
    model_uri = "s3://hambart-training/distilbert-base-uncased-model/"
    model = get_model(model_uri)


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

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 output
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\\n")

    # Saves the model to s3; default is /opt/ml/model which SageMaker sends to S3
    trainer.save_model(args.model_dir)
