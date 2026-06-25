import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, data_path="data/diabetes.csv", artifacts_dir="artifacts/"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir

    def initiate_data_transformation(self):
        try:
            print("Data Transformation stage shuru ho raha hy...")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file '{self.data_path}' nahi mili!")
                
            df = pd.read_csv(self.data_path)
            print(f"Dataset shape: {df.shape} successfully load ho gaya.")

            zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_columns:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())
            
            print("Invalid zero values ko median se successfully impute kar diya hy.")

            X = df.drop(columns=['Outcome'], axis=1)
            y = df['Outcome']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print("Train-Test split mukammal (80% Train, 20% Test).")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Saving the scaler object for the prediction pipeline
            os.makedirs(self.artifacts_dir, exist_ok=True)
            scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")
            with open(scaler_path, "wb") as scaler_file:
                pickle.dump(scaler, scaler_file)
            print("Scaler object successfully saved as pickle file.")

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            np.save(os.path.join(self.artifacts_dir, "train.npy"), train_arr)
            np.save(os.path.join(self.artifacts_dir, "test.npy"), test_arr)
            print(f"Transformed matrices 'artifacts/' folder me save ho gayi hain.")

            return (
                os.path.join(self.artifacts_dir, "train.npy"),
                os.path.join(self.artifacts_dir, "test.npy")
            )

        except Exception as e:
            print(f"Error during Data Transformation: {e}")
            raise e

if __name__ == "__main__":
    transformation = DataTransformation()
    transformation.initiate_data_transformation()
