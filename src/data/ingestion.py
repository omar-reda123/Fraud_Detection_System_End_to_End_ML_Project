import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
class DataIngestor:
    def __init__(self,data_path:str="data/processed"):
        self.path=data_path

    def load_data(self,file_name:str)->pd.DataFrame: 
        full_path = os.path.join(self.path, file_name)

        try:
            logging.info(f"Attempting to read file from {full_path}")
            df=pd.read_csv(full_path)
            logging.info(f"Successfully loaded data from {full_path}, with shape of {df.shape}")
            return df
        
        except FileNotFoundError:
            logging.error(f"File not found at: {full_path}. Please check the directory and file name.")
            raise

        except pd.errors.EmptyDataError:
            logging.error(f"File found at: {full_path} is empty.")
            raise

        except Exception as E:
            logging.error(f"An unexpected error occurred while loading {file_name}: {E}")
            raise
    





        