import pandas as pd
import joblib
import logging
from src.data.ingestion import DataIngestor
from src.features.feature_engineering import FeatureEngineer
from src.data.preprocessing import FraudPreprocessor
from src.models.evaluate import ModelEvaluator


#testing on test.csv
class Test:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.engineer = FeatureEngineer()
        self.preprocessor = joblib.load("models/preprocessor.pkl") 
        self.model=joblib.load('models/fraud_model.pkl')       
        self.evaluator = ModelEvaluator()
    def run_test(self,filename='test.csv'):
        logging.info("Started testing pipeline")

        # 1. Ingestion
        test_df=self.ingestor.load_data(file_name=filename,is_train=False)
        # 2. Feature Engineering
        test_df=self.engineer.add_features(test_df)
        # 3. Preprocessing (Scaling/Imputing)
        test_df=self.preprocessor.transform(test_df)
        # 4. Split X and y
        y_test = test_df['Class']
        X_test = test_df.drop('Class', axis=1)

        logging.info("--- Evaluating Model on Testing Data ---")
        self.evaluator.evaluate_model(self.model, X_test, y_test)


