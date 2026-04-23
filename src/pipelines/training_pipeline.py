import logging
from src.data.ingestion import DataIngestor
from src.features.feature_engineering import FeatureEngineer
from src.data.preprocessing import FraudPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TrainingPipeline:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.engineer = FeatureEngineer()
        self.preprocessor = FraudPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator() 

    def run_pipeline(self, train_path="cleaned_train.csv", val_path="val.csv"):
        logging.info("--- Starting ML Pipeline ---")
        
        # 1. Ingestion
        train_df = self.ingestor.load_data(train_path)
        val_df = self.ingestor.load_data(val_path)
        
        # 2. Feature Engineering
        train_df = self.engineer.add_features(train_df)
        val_df = self.engineer.add_features(val_df)
        
        # 3. Preprocessing (Cleaning & Scaling/Imputing)
        train_df = self.preprocessor.clean_data(train_df, is_train=True)
        val_df = self.preprocessor.clean_data(val_df, is_train=False)
        
        train_df = self.preprocessor.fit_transform(train_df)
        val_df = self.preprocessor.transform(val_df)
        
        # 4. Split X and y
        y_train = train_df['Class']
        X_train = train_df.drop('Class', axis=1)
        
        y_val = val_df['Class']
        X_val = val_df.drop('Class', axis=1)
        
        
        # 5. Train Model
        self.trainer.train_data(X_train, y_train)
        self.trainer.save_model()
        
        # 6. Evaluate Model
        logging.info("--- Evaluating Model on Validation Data ---")
        self.evaluator.evaluate_model(self.trainer.model, X_val, y_val)
        
        logging.info("--- Pipeline Completed Successfully! ---")
        
        return self.trainer.model