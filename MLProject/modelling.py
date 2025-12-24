import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================
# AUTOLOG (WAJIB)
# =====================
mlflow.autolog(log_models=True)

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("medical_cost_preprocessed.csv")
X = df.drop(columns=["annual_medical_cost"])
y = df["annual_medical_cost"]

# =====================
# MODEL
# =====================
model = SGDRegressor(
    max_iter=1,
    learning_rate="invscaling",
    eta0=0.01,
    random_state=42
)

batch_size = 256

# =====================
# ONLINE TRAINING
# =====================
for i in range(0, len(X), batch_size):
    model.partial_fit(
        X.iloc[i:i + batch_size],
        y.iloc[i:i + batch_size]
    )

# =====================
# TRIGGER AUTOLOG MODEL
# =====================
model.fit(X.iloc[:batch_size], y.iloc[:batch_size])

# =====================
# EVALUATION
# =====================
preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))
r2 = r2_score(y, preds)

print("✅ Training with MLflow autolog completed")
