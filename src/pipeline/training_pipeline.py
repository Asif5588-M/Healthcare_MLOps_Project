import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            print("\n=================== Stage 1: Data Ingestion Started ===================")
            DATASET_SLUG = "uciml/pima-indians-diabetes-database"
            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion(DATASET_SLUG)
            print("=================== Stage 1 Completed Successfully ===================\n")

            print("=================== Stage 2: Data Transformation Started ===================")
            transformation = DataTransformation()
            train_path, test_path = transformation.initiate_data_transformation()
            print("=================== Stage 2 Completed Successfully ===================\n")

            print("=================== Stage 3: Model Training Started ===================")
            trainer = ModelTrainer()
            trainer.initiate_model_trainer()
            print("=================== Stage 3 Completed Successfully ===================\n")

            print("End-to-End Training Pipeline executed successfully without errors!")

        except Exception as e:
            print(f"Pipeline execution failed at runtime: {e}")
            sys.exit(1)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
