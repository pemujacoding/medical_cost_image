import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import os
import shutil
import dagshub

def modelling():

    # Load preprocessed data
    df = pd.read_csv("MLProject/medical_cost_preprocessed.csv")
    X = df.drop(columns=["annual_medical_cost"])
    y = df["annual_medical_cost"]

    model = SGDRegressor(
        max_iter=1, 
        learning_rate="invscaling",
        eta0=0.01,
        random_state=42
    )

    batch_size = 256

    with mlflow.start_run():
        for i in range(0, len(X), batch_size):
            X_batch = X.iloc[i:i + batch_size]
            y_batch = y.iloc[i:i + batch_size]
            model.partial_fit(X_batch, y_batch)

        # Evaluation
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # 1. Save as a single file (for your artifact upload step)
        dump(model, "MLProject/online_sgd_model.joblib")
        mlflow.log_artifact("MLProject/online_sgd_model.joblib")

        # 2. Log to DagsHub (Remote)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="online_model",
            input_example=X.iloc[:5]
        )

        # 3. SAVE LOCALLY (For Docker build step)
        # This creates the MLProject/online_model folder with MLmodel metadata
        local_model_path = "MLProject/MLModel"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
            
        mlflow.sklearn.save_model(
            sk_model=model,
            path=local_model_path,
            input_example=X.iloc[:5]
        )

print("Online training completed.")

if __name__ == "__main__":
    modelling()