from bot import Bot 
from run_local_RAC import RAC
from run_local_summary import Summary
from download_model_files import Downloader

'''
run.py is the main coordinator between the scripts within the bots directory.
It is responsible for starting the download of model files, instantiating the 
models locally for inference, and instanting and starting the bot.
'''

def get_dependencies():
    downloader = Downloader() #instantiate the model files downloader
    downloader.get_models() #download the model files if they do not exist locally

    local_RAC = RAC() #instantiate the local Rule Adherence Classifier
    local_summary = Summary() #instantiate the local Clean Summary Model

    bot = Bot(local_RAC, local_summary, token)  #instantiate the bot with the local Rule Adherence Classifier

    return bot


def main():
    bot = get_dependencies()  #instantiate bot with necessary dependencies
    bot.start() #start the bot


token =  #token goes here

if __name__ == "__main__":
    main()