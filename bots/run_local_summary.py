import pandas as pd
import torch
from transformers import T5TokenizerFast
from transformers import T5Config, T5ForConditionalGeneration
import transformers
import os

"""
This script is used to run the clean summary model locally within the discord bot.
When run.py is run, the model files will be downloaded (if needed) to the correct
direcory for running this script. The Sumamry class is imported into run.py where it
is used to make predictions on discord messages.
"""


class Summary():
    def __init__(self):
        dash_line = "-"*50 + "\n"
        print(f"{dash_line*3}\nInsantiating Local Clean Summary Model...\n{dash_line*3}")
        # Get the absolute path of the current script
        script_path = os.path.abspath(__file__)

        # Get the directory containing the script
        script_dir = os.path.dirname(script_path)
        print(f'\nFinding Clean Summary Model...\nScript Directory {script_dir}')

        parent_dir = "\\".join(script_dir.split(os.sep)[:-1])  
        print(f'Parent Directory {parent_dir}')

        #Append repo structure to local machine path
        model_path = parent_dir + '\\local_models\\Models\\clean_summary_model\\'
        print(f'Model Path {model_path}')
        print(f'Rule Adherance Classifier Model Files: {os.listdir(model_path)}\n\n')

        # currently the model is not fine tuned enough to be deployed. I am working on constructing a 
        # larger and better dataset to train on. Currently the model outputs no tokens and I am not sure why.
        # in the meantime, in order to develop the bot further I am just loading in the vase t5-model for summary
        # the base model does summarize but it does not give a clean summary like I intend to implement.
        # This init function can still access the fine tuneed model files, just replace 't5-small' .from_pretrained() path
        # with the model_path variable in config and self.model but not in the tokenizer. The tokenizer must be the base model

        #load fine tuned classifier for inference
        config = T5Config.from_pretrained('t5-small')  
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small', config=config)
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-small')


    #Preration of Text Data#############################################################################################################

    def encode(self, comment):
        # Encode the text with the prefix "summarize: " for the T5 model
        encoded = self.tokenizer(f"summarize: {comment}", truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        
        # Convert 0d tensors to python numbers using .tolist() for the entire tensor
        attention_mask = encoded['attention_mask'][0].tolist()
        input_ids = encoded['input_ids'][0].tolist()
        
        # Return data in a dictionary format with appropriate keys for the trainer
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def apply_encoding(self, df):
        # Apply the encode() to the message within the df
        # df.apply() returns a dict of pandas series 
        encoded_data = df.apply(lambda row: self.encode(row['message']), axis=1)
        
        list_of_dicts = [item for item in encoded_data] # move the pandas series into a list

        return pd.DataFrame(list_of_dicts) #return as df 
    
    #####################################################################################################################################

    def inference(self, model, encoded_message_df):
        '''
        This function runs inference on the encoded data. The model and message df are passed in as an argument
        '''
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert 'input_ids' and 'attention_mask' columns to tensors for model input
        input_ids = torch.tensor(encoded_message_df['input_ids'].tolist())
        attention_mask = torch.tensor(encoded_message_df['attention_mask'].tolist())

        # Move and model input data to the device
        model.to(device)
        model.eval()
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device) # move tensors to device

        # Perform inference 
        with torch.no_grad():
            #.generate() returns tokenized ids of the generated summary
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10)

        # Decode the output tokens to a string
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #####################################################################################################################################
    
    def run(self, discord_message):
        '''
        This function takes a single discord message string as input, converts it to a dataframe,
        then passes it to apply_encoding() which tokenizes the input data. apply_encoding()
        also outputs a df. Then inference() is passed that df whichs passed the embedded
        tokens and attention mask into the model as pt tensors. inference() returns a string.

        Currently this function will process each message at a time on the local machine. This is 
        not ideal for performance nor deployment but will do for now as other aspects of this project 
        are expanded upon.

        '''
        # Prepare the data
        message_df = pd.DataFrame({'message': [discord_message]})
        encoded_message_df = self.apply_encoding(message_df)

        # Return the prediction
        return self.inference(self.model, encoded_message_df)

    
    

