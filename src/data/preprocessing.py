import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
class FraudPreprocessor:
    def __init__(self):
        self.scaler=RobustScaler()
        self.imputer=SimpleImputer(strategy='median')
        logging.info("FraudPreprocessor initialized successfully with RobustScaler and median SimpleImputer.")
    def clean_data(self,df:pd.DataFrame,is_train=False):
        data=df.copy()
        if is_train:
            before = len(data)
            data = data.drop_duplicates(keep='first')
            after = len(data)
            logging.info(f"Training Mode: Dropped {before - after} duplicate rows.")
        else:
            logging.info("Evaluation Mode: No duplicates dropped.")
        
        return data
    def fit(self,df:pd.DataFrame):
        self.scaler.fit(df[['Amount']])
        self.imputer.fit(df)
        logging.info("Fitting completed successfully.")
        return self
    
    def transform(self,df:pd.DataFrame):
        data=df.copy()
        imputed_data=self.imputer.transform(data) #returns array not dataframe
        data=pd.DataFrame(imputed_data,columns=data.columns,index=data.index)
        data[['Amount']]=self.scaler.transform(data[['Amount']])
        logging.info("Transformation completed successfully.")
        return data
    def fit_transform(self, df: pd.DataFrame):
        logging.info("Starting fit_transform process...")
        return self.fit(df).transform(df)













