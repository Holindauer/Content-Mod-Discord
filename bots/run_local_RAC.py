import pandas as pd
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertConfig, DistilBertForSequenceClassification
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

"""
See RAC.run() at the bottom for explantation of how this class is used.
"""


class RAC():
    def __init__(self):
        # Get the absolute path of the current script
        script_path = os.path.abspath(__file__)

        # Get the directory containing the script
        script_dir = os.path.dirname(script_path)
        print(f'\nFinding Rule Adherance Classifier...\nScript Directory {script_dir}')

        parent_dir = "\\".join(script_dir.split(os.sep)[:-1])
        print(f'Parent Directory {parent_dir}')

        #Append repo structure to local machine path
        model_path = parent_dir + '\\local_models\\Models\\Rule Adherence Classifier'
        print(f'Model Path {model_path}')
        print(f'Rule Adherance Classifier Model Files: {os.listdir(model_path)}\n\n')


        #load fine tuned classifier for inference
        config = DistilBertConfig.from_pretrained(model_path, num_labels=2)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    #Preration of Text Data#############################################################################################################
    def encode(self, comment, label):
        #tokenize
        encoded = self.tokenizer(comment, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        
        # Convert 0d tensors to python numbers using .item() for each element in enocded vectors
        attention_mask = [i.item() for i in encoded['attention_mask'][0]]
        input_ids = [i.item() for i in encoded['input_ids'][0]]
        label = label.item() if isinstance(label, torch.Tensor) else label
        
        # Return data in a dictionary format
        return {
            'attention_mask': attention_mask,
            'input_ids': input_ids,
            'label': label,
            'text': comment
        }
    
    def apply_encoding(self, df):
        '''
        This function uses the encode() function define above
        to tokenize the discord_message and convert to a pandas df
        '''
        # Apply the encode function to each row of the dataframe
        encoded_data = df.apply(lambda row: self.encode(row['comment_text'], row['toxic']), axis=1)
        
        # Convert encoded data to a list of dictionaries
        list_of_dicts = [item for item in encoded_data]
        
        # Convert list of dictionaries to dataframe
        return pd.DataFrame(list_of_dicts)
    #####################################################################################################################################


    def inference(self, model, encoded_message_df):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")

        # Convert 'input_ids' and 'attention_mask' columns to tensors for model input
        input_ids = torch.tensor(encoded_message_df['input_ids'].tolist())
        attention_mask = torch.tensor(encoded_message_df['attention_mask'].tolist())

        # Move and model input data to the device
        model.to(device)
        model.eval()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Perform inference 
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]

        # Use the argmax function to get the predicted binary labels
        return torch.argmax(logits, dim=1).cpu()

    #####################################################################################################################################
    
    def run(self, discord_message):
        '''
        This function takes a single discord message string as input, converts it to a dataframe,
        then passes it to apply_encoding() which tokenizes the input data. apply_encoding()
        also outputs a df. Then inference() is passed that df whichs passed the embedded
        tokens and attention mask into the model as pt tensors. inference() returns a tensor
        of the argmax of the logits.

        Currently this function will process each message at a time on the local machine. This is 
        not ideal for performance nor deployment but will do for now as other aspects of this project 
        are expanded upon.

        '''
        # Prepare the data
        message_df = pd.DataFrame({'comment_text': [discord_message], 'toxic': None})
        encoded_message_df = self.apply_encoding(message_df)

        # Perform inference
        prediction = self.inference(self.model, encoded_message_df)

        # Return the prediction
        return prediction.item()
    
