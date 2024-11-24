"""
This script downloads the dataset given as parameter 'dataset_name' and unzips it to the subfolder 'data' on the top level.
The script leverages the Kaggle SDK for authenticating with the Kaggle API and downloading the dataset.
Make sure that the 'kaggle.json' file with the Kaggle API credentials is properly set up in the system.
You can follow the instructions at https://www.kaggle.com/docs/api.

The main function `load_via_kaggle` performs the authentication and download steps.

This script can be run directly, and upon execution, it will download and unzip the specified dataset to the
destination path.

Dependencies:
    - Kaggle SDK
    - os module

Usage:
    python load_data.py

"""
import argparse
import os
from kaggle import KaggleApi

script_folder = os.path.dirname(os.path.abspath(__file__))
destination_path = os.path.join(script_folder, "data")

def create_argument_parser():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-p', '--dataset_path', type=str, default='Unknown')
    return parser

def load_via_kaggle(dataset_name):
    # Authenticating with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Downloading the dataset
    api.dataset_download_files(dataset_name, path=destination_path, unzip=True)


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.dataset_path == 'Unknown':
        print("Please, use the ['--datatset_path'] to give the path to your kaggle data!")
    else:
        load_via_kaggle(args.dataset_path)
