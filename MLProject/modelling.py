import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import dagshub

# =====================
# DAGSHUB + MLFLOW
# =====================
dagshub.init(
    repo_owner="pemujacoding",
    repo_name="medical_cost_model",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/pemujacoding/medical_cost_model.mlflow"
)

# =====================
# EXPERIMENT
# =====================
experiment_name = "Online Training Medical Cost"
mlflow.set_experiment(experiment_name)

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("MLProject/medical_cost_preprocessed.csv")
X = df.drop(columns=["annual_medical_cost"])
y = df["annual_medical_cost"]

# =====================
# PARAMS (WAJIB DICT)
# =====================
params = {
    "max_iter": 1,
    "learning_rate": "invscaling",
    "eta0": 0.01,
    "random_state": 42
}

model = SGDRegressor(**params)
batch_size = 256

# =====================
# TRAINING
# =====================
with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.log_params(params)

    for i in range(0, len(X), batch_size):
        model.partial_fit(
            X.iloc[i:i + batch_size],
            y.iloc[i:i + batch_size]
        )

    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    dump(model, "online_sgd_model.joblib")
    mlflow.log_artifact("online_sgd_model.joblib")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X.iloc[:5]
    )


with open("run_id.txt", "w") as f:
    f.write(run_id)
print("✅ Online training completed & logged to Dagshub")
