import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
class FeatureEngineer:
    def __init__(self, time_column='Time', new_column='Hour'):
        self.time_col = time_column
        self.new_col = new_column
    def add_features(self,df:pd.DataFrame):
        data=df.copy()
        data[self.new_col]=((data[self.time_col]//3600)%24).astype(int)
        logging.info(f"Created new feature {self.new_col} from old feature {self.time_col}")
        data=data.drop(columns=[self.time_col])
        logging.info(f"Dropped original feature '{self.time_col}'.")
        return data


        
    