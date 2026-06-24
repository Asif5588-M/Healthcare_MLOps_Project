import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, data_path="data/diabetes.csv", artifacts_dir="artifacts/"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir

    def initiate_data_transformation(self):
        try:
            print("Data Transformation stage shuru ho raha hy...")
            
            # 1. Data read karna
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file '{self.data_path}' nahi mili!")
                
            df = pd.read_csv(self.data_path)
            print(f"Dataset shape: {df.shape} successfully load ho gaya.")

            # 2. Invalid Zeros ko NaN se replace karna (Healthcare specific cleaning)
            zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_columns:
                df[col] = df[col].replace(0, np.nan)
                # Median se missing values fill karna
                df[col] = df[col].fillna(df[col].median())
            
            print("Invalid zero values ko median se successfully impute kar diya hy.")

            # 3. Features aur Target ko alag karna
            X = df.drop(columns=['Outcome'], axis=1)
            y = df['Outcome']

            # 4. Train-Test Split (80% Train, 20% Test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print("Train-Test split mukammal (80% Train, 20% Test).")

            # 5. Feature Scaling (Production alignment)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 6. Preprocessed Data ko saving ke liye prepare karna
            os.makedirs(self.artifacts_dir, exist_ok=True)
            
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Save as numpy arrays for efficient loading in trainer stage
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
