import os
from kaggle.api.kaggle_api_extended import KaggleApi

class DataIngestion:
    def __init__(self, download_path="data/"):
        self.download_path = download_path

    def initiate_data_ingestion(self, dataset_slug):
        try:
            print("Kaggle API initialize ho rahi hy...")
            api = KaggleApi()
            api.authenticate()
            
            os.makedirs(self.download_path, exist_ok=True)
            print(f"Dataset {dataset_slug} download ho raha hy...")
            
            api.dataset_download_files(dataset_slug, path=self.download_path, unzip=True)
            print(f"Data successfully extracted to '{self.download_path}'")
            
        except Exception as e:
            print(f"Error while downloading data: {e}")
