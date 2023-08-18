import gdown
import os

'''
This class definitions uses the gdown library to download the model files from google drive.
It is called by the run.py script to download the model files if they do not exist locally.
'''


class Downloader():
    def __init__(self):
        dash_line = "-"*50 + "\n"
        print(f"{dash_line*3}\nInsantiating Model Files Downloadeer...")

    def obtain_download_path(self):
        '''
        This method finds the path to this script on whatever machine it is run on.
        It then backtracks from that path into the directory where the model files 
        are to be saved to run the bot locally.
        '''
        # Get the absolute path of the current script
        script_path = os.path.abspath(__file__)
        print(f'\nFound download_model_files.py Script Path {script_path}')

        # Get the directory containing the script
        script_dir = os.path.dirname(script_path)

        parent_dir = "\\".join(script_dir.split(os.sep)[:-1])
        print(f'Parent Directory {parent_dir}')

        #Append repo structure to local machine path
        download_path = parent_dir + '\\local_models\\Models\\'
        print(f'Creating Download Path {download_path}\n')


        return download_path


    def download_model_files(self):
        '''
        This functions calls on the obtain_download_path() method to find the path to the
        directory where the model files are to be saved. It then checks if the model files
        already exist in that directory. If they do not, it downloads them from google drive.
        If this fails for any reason, it prints an error message and instructs the user to
        download the model files manually.
        '''
        try:
            model_files_dir = self.obtain_download_path()
            print(f": {model_files_dir}")
            print(f"Checking if Model Files Exist Locally At {model_files_dir}\n")\
                
            dir_contents = os.listdir(model_files_dir) #get model files if they exist locally
            if dir_contents:
                print(f"Model Files Found Locally: {dir_contents}\n")
            elif not dir_contents:
                print("Model Files Not Found\n Starting Model Files Download...\n")

                print("Starting Rule Adherence Classifier Model Download from Google Drive ...\n")
                #RAC Model Links
                url_list = {
                    "config.json" : "https://drive.google.com/uc?id=1QC6AGEdKRyMQIrgoGnwl_S7tmOipB0VS", 
                    "pytorch_model.bin" : "https://drive.google.com/uc?id=1OGyQygvKiphhLw5jnVY3QJoAK0EDOIPT"
                }  #these are the direct download links from google drive. Not view links

                output_directory = self.obtain_download_path() 

                [gdown.download(url, output_directory + file_name, quiet=False) for file_name, url in url_list.items()] #downloads the files from the links

        except:
            print("\n\n\nError Downloading Model Files. \nTo Download Manually, Follow the Instructions at Content-Mod-Bot\\local_models\\Models\\Instructions.txt\n\n\n ")

        