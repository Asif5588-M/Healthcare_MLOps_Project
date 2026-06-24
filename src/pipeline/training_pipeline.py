from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    print("Running Training Pipeline...")
    DATASET_SLUG = "uciml/pima-indians-diabetes-database"
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion(DATASET_SLUG)
