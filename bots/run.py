from bot import Bot 
from run_local_RAC import RAC
from download_model_files import Downloader


def get_dependencies():
    downloader = Downloader() #instantiate the model files downloader
    downloader.get_models() #download the model files if they do not exist locally

    local_RAC = RAC() #instantiate the local Rule Adherence Classifier

    bot = Bot(local_RAC, token)  #instantiate the bot with the local Rule Adherence Classifier

    return downloader, bot


def main():

    downloader, bot = get_dependencies()  #instantiate the dependencies

    bot.start() #starts the bot



token =  #token goes here

if __name__ == "__main__":
    main()