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

        #Rule Adherance Classifier Model Links
        self.RAC_links = {
            "config.json" : "https://drive.google.com/uc?id=1QC6AGEdKRyMQIrgoGnwl_S7tmOipB0VS", 
            "pytorch_model.bin" : "https://drive.google.com/uc?id=1OGyQygvKiphhLw5jnVY3QJoAK0EDOIPT"
            }  #these are the direct download links from google drive. Not view links
                    
        # Clean Summary Model Links
        self.summary_links = {
            "config.json": "https://drive.google.com/uc?id=1Gy4FDA98Bfgq32a0xwW5sAPc6YHnzERc",
            "generation_config.json": "https://drive.google.com/uc?id=18x8E0DS23kV3wWNf1STBxoTW4QOD8o_o",
            "pytorch_model.bin": "https://drive.google.com/uc?id=1_ZYMu4kk5FatpNn7foVyI6jGH5Qb8jG_",
            "training_args.bin": "https://drive.google.com/uc?id=1wd7ZML1_SC1uO9B0UnX2fv1IoQi2v6_e"
        }

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

        # Append repo structure to local machine path
        # When The reason the repository contains the empty Models directory is so that the
        # download_model_files() method can check if there are model files inside of it. The
        # local_model dir contains other files so this mechanism would not work otherwise.
        download_path = parent_dir + '\\local_models\\Models\\'
        print(f'Creating Download Path {download_path}\n')


        return download_path
    
    def download(self, subdir, url_dict):
        '''
        This function is called within the download_model_files() method. It downloads the
        model files for all models from google drive. The model files
        are saved to the directory returned by the obtain_download_path() with the addition
        of the specified subdir.
        '''

        print("Starting Rule Adherence Classifier Model Download from Google Drive ...\n")

        output_directory = os.path.join(self.obtain_download_path(), subdir) #specify where to save the model files
        
        # Create directory if not exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
   
        for file_name, url in url_dict.items():
            gdown.download(url, os.path.join(output_directory, file_name), quiet=False) #downloads the files from the links


    def get_models(self):
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
            print(f"Checking if Model Files Exist Locally At {model_files_dir}\n")
                    
            dir_contents = os.listdir(model_files_dir) #get model files if they exist locally
            if dir_contents:
                print(f"Model Files Found Locally: {dir_contents}\n")

            elif not dir_contents:
                print("Model Files Not Found\n Starting Model Files Download...\n")
                    
                download_specs = {"rule_adherance_classifier" : self.RAC_links, "clean_summary_model" : self.summary_links}  #specify what and where to download
                [self.download(subdir=subdir, url_dict=links) for subdir, links in download_specs.items()]  #download all files
            
        except:
            print("\n\n\nError Downloading Model Files. \nTo Download Manually, Follow the Instructions at Content-Mod-Bot\\local_models\\Models\\Instructions.txt\n\n\n ")

