import pandas as pd
from data.ingestion import DataIngestor


class FeatureEngineer:
    def __init__(self, time_column='Time', new_column='Hour'):
        self.time_col = time_column
        self.new_col = new_column
    def add_features(self,df:pd.DataFrame):
        data=df.copy()
        data[self.new_col]=((data[self.time_col]//3600)%24).astype(int)
        return data


        
    