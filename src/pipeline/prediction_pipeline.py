import os
import pickle
import pandas as pd
import numpy as np

class PredictionPipeline:
    def __init__(self, artifacts_dir="artifacts/"):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(self.artifacts_dir, "model.pkl")
        self.scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")

    def predict(self, features: list) -> int:
        """
        Takes a raw list of features representing patient health data metrics,
        scales them, and runs the loaded model for classification prediction.
        Expects 8 feature values: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        """
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                raise FileNotFoundError("Trained model or scaler artifacts missing!")

            # Loading serialized scaler and model objects
            with open(self.scaler_path, "rb") as scaler_file:
                scaler = pickle.load(scaler_file)

            with open(self.model_path, "rb") as model_file:
                model = pickle.load(model_file)

            # Transforming input list to proper dataframe/array format
            features_array = np.array(features).reshape(1, -1)
            
            # Applying standard scaling transformations
            scaled_features = scaler.transform(features_array)

            # Executing inference classification
            prediction = model.predict(scaled_features)
            
            return int(prediction[0])

        except Exception as e:
            print(f"Error encountered during prediction inference workflow: {e}")
            raise e

if __name__ == "__main__":
    # Dummy patient feature testing criteria
    # Schema: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age
    test_patient_metrics = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    
    pipeline = PredictionPipeline()
    result = pipeline.predict(test_patient_metrics)
    print(f"\nTest Execution Result -> Diagnostic Prediction outcome class: {result}")
