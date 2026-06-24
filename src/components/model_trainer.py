import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

class ModelTrainer:
    def __init__(self, artifacts_dir="artifacts/"):
        self.artifacts_dir = artifacts_dir
        self.train_array_path = os.path.join(self.artifacts_dir, "train.npy")
        self.test_array_path = os.path.join(self.artifacts_dir, "test.npy")

    def initiate_model_trainer(self):
        try:
            print("Loading transformed datasets...")
            if not os.path.exists(self.train_array_path) or not os.path.exists(self.test_array_path):
                raise FileNotFoundError("Transformed data arrays not found in artifacts!")

            # Loading numpy arrays
            train_data = np.load(self.train_array_path)
            test_data = np.load(self.test_array_path)

            # Splitting Features and Target
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            # Starting MLflow Experiment Tracking
            mlflow.set_experiment("Healthcare_Diabetes_Prediction")

            with mlflow.start_run():
                print("Training Random Forest Classifier...")
                # Defining Hyperparameters
                n_estimators = 100
                max_depth = 5
                
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Evaluation Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                print(f"Model Metrics -> Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

                # Logging parameters and metrics to MLflow
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)

                # Logging the model into MLflow artifact store
                mlflow.sklearn.log_model(model, "model")
                print("Model and metrics successfully logged into MLflow tracker.")

                # Save model locally for backup deployment
                model_dir = os.path.join(self.artifacts_dir, "model")
                os.makedirs(model_dir, exist_ok=True)
                # We can also rely completely on MLflow registry later

        except Exception as e:
            print(f"Error occurred during model training: {e}")
            raise e

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer()
