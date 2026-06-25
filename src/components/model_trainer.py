import os
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

            train_data = np.load(self.train_array_path)
            test_data = np.load(self.test_array_path)

            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            mlflow.set_experiment("Healthcare_Diabetes_Prediction")

            with mlflow.start_run():
                print("Training Random Forest Classifier...")
                n_estimators = 100
                max_depth = 5
                
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                print(f"Model Metrics -> Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)

                mlflow.sklearn.log_model(model, "model")
                print("Model and metrics successfully logged into MLflow tracker.")

                # Exporting the model object for local API utilization
                model_path = os.path.join(self.artifacts_dir, "model.pkl")
                with open(model_path, "wb") as model_file:
                    pickle.dump(model, model_file)
                print("Model object successfully saved locally as pickle file.")

        except Exception as e:
            print(f"Error occurred during model training: {e}")
            raise e

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer()
