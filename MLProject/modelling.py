import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import os
import dagshub
dagshub.init(
    repo_owner='pemujacoding',
    repo_name='medical_cost_model',
    mlflow=True)

def modelling() :

    # MLflow dagshub
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(f"https://{dagshub_token}@dagshub.com/pemujacoding/medical_cost_model.mlflow")

    # MLflow setup
    experiment_name = "Online Training Medical Cost"

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Load preprocessed data
    df = pd.read_csv("medical_cost_preprocessed.csv")

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

            # Incremental training
            model.partial_fit(X_batch, y_batch)

        # Evaluation (after training)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        dump(model, "MLProject/online_sgd_model.joblib")
        mlflow.log_artifact("MLProject/online_sgd_model.joblib")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="online_model",
            input_example=X.iloc[:5]
        )

print("Online training completed.")

if __name__ == "__main__":
    modelling()