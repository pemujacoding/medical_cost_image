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
    # DagsHub/MLflow setup
    dagshub.init(
        repo_owner='pemujacoding',
        repo_name='medical_cost_model',
        mlflow=True,
    )
    
    mlflow.set_tracking_uri("https://dagshub.com/pemujacoding/medical_cost_model.mlflow/")
    experiment_name = "Online Training Medical Cost"

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.autolog(log_models=True)

    # Load data
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

        # ===== ONLINE TRAINING =====
        for i in range(0, len(X), batch_size):
            X_batch = X.iloc[i:i + batch_size]
            y_batch = y.iloc[i:i + batch_size]
            model.partial_fit(X_batch, y_batch)

        # ===== TRIGGER AUTOLOG MODEL =====
        # Fit sekali dengan batch kecil
        model.fit(X.iloc[:batch_size], y.iloc[:batch_size])

        # Evaluation
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, preds)

    print("Training with MLflow autolog completed.")


if __name__ == "__main__":
    modelling()