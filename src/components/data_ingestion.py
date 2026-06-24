import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# .env folder ke conflict se bachne ke liye custom file load kar rahe hain
load_dotenv(dotenv_path="config.env")

class DataIngestion:
    def __init__(self, download_path="data/"):
        self.download_path = download_path

    def initiate_data_ingestion(self, dataset_slug):
        try:
            # Check karain ke credentials sahi load hui hain
            if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
                raise ValueError("Kaggle credentials config.env file me missing hain!")

            print("Kaggle API initialize ho rahi hy...")
            api = KaggleApi()
            api.authenticate()
            
            os.makedirs(self.download_path, exist_ok=True)
            print(f"Dataset '{dataset_slug}' download ho raha hy...")
            
            api.dataset_download_files(dataset_slug, path=self.download_path, unzip=True)
            print(f"Data successfully download aur extract ho gaya hy: '{self.download_path}'")
            
        except Exception as e:
            print(f"Error while downloading data: {e}")

if __name__ == "__main__":
    DATASET_SLUG = "uciml/pima-indians-diabetes-database"
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion(DATASET_SLUG)
