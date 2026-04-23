import pandas as pd
import logging
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ModelEvaluator:
    def __init__(self):
        pass
    def evaluate_model(self,model,X_val:pd.DataFrame,y_val:pd.Series):
        predictions=model.predict(X_val)
        probabilities=model.predict_proba(X_val)[ : , 1]
        f1=f1_score(y_val,predictions)
        roc_auc=roc_auc_score(y_val,probabilities)
        report=classification_report(y_val,predictions)
        logging.info(f"f1 score: {f1} \n  roc_auc: {roc_auc} \n report: {report}")
        return f1 , roc_auc
                


