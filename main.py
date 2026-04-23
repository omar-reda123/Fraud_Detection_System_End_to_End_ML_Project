from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.logger import setup_logger

if __name__ == "__main__":
    setup_logger()
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()