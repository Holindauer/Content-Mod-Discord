from bot import Bot 
from rule_adherance_classifier import RAC
from download_model_files import Downloader

'''
run.py is the main coordinator between the scripts within the bots directory.
It is responsible for starting the download of model files, instantiating the 
models locally for inference, and instanting and starting the bot.
'''


def main():

    #instantiate the local Rule Adherence Classifier
    rac = RAC() 

    # instantiate the bot with the read in token
    token = open("token.txt", "r").read().strip()
    bot = Bot(rac, token)  

    # start up the bot on the server
    bot.start() 


if __name__ == "__main__":
    main()