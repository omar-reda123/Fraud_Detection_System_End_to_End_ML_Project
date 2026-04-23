import pandas as pd
import logging
import os
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ModelTrainer:
    def __init__(self):
        self.model=LogisticRegression(class_weight='balanced',max_iter=10000,random_state=42)
    def train_data(self,X_train:pd.DataFrame,y_train:pd.Series):
         logging.info("Training in progress!")
         self.model.fit(X_train,y_train)
         logging.info("Training finished!")
         return self.model
    def save_model(self,path:str="models/fraud_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model,path)
        logging.info(f"model saved at: {path}!")









