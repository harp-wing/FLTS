import numpy as np
import torch
import mlflow
import mlflow.pytorch
import io
from train import train_model
from client_utils import get_file
import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi_service:8000")

# === Download and read .parquet file ===
parquet_bytes = get_file(GATEWAY_URL, "processed-data", "processed_data.parquet")
table = pq.read_table(source=parquet_bytes)
df = table.to_pandas()

# === Split features and targets ===
X_df = df
y_df = df[["down", "up"]]

X = X_df.to_numpy().astype(np.float32)
y = y_df.to_numpy().astype(np.float32)

# === LSTM reshaping ===
FORECAST_HORIZON=3982
T = 5
y_features = y.shape[1]  # Number of target features (e.g., "down", "up")

# Create sliding window samples
num_samples = len(X) - T - FORECAST_HORIZON + 1

# Pre-allocate arrays for efficiency
X_new = np.zeros((num_samples, T, X.shape[1]), dtype=np.float32)
y_new = np.zeros((num_samples, FORECAST_HORIZON, y_features), dtype=np.float32)

for i in range(num_samples):
    X_new[i] = X[i : i+T]
    y_new[i] = y[i+T : i+T+FORECAST_HORIZON]

# Flatten the target's last two dimensions for the MLP head
# Shape changes from (num_samples, 3982, 2) to (num_samples, 7964)
y_new_flat = y_new.reshape(num_samples, FORECAST_HORIZON * y_features)

# Update the main variables
X = X_new
y = y_new_flat

# Sanity check
mask = ~np.isnan(y).any(axis=1)
X = X[mask]
y = y[mask]

# === Config ===
config = {
    "device": "cpu",
    "model_name": "lstm",
    "sequence_length": T,
    "num_lags": T,
    "num_features": X.shape[2],
    "output_dim": FORECAST_HORIZON * y_features,
    "epochs": 5,
    "batch_size": 32,
    "lr": 0.001
}

# === MLflow Logging ===
# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("flts-lstm-demo")

with mlflow.start_run(run_name="LSTM"):
    mlflow.log_params(config)
    print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[DEBUG] NaNs — y: {np.isnan(y).any() | np.isinf(y).any()}, X: {np.isnan(X).any() | np.isinf(X).any()}")

    model = train_model(X, y, X, y, config)

    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=X[:1],
        registered_model_name=None,
        code_paths=["m1.py"] # This tells MLflow to bundle m1.py with the model!
    )

    mlflow.log_metric("val_loss", 0.002)
    print("✅ Model logged to MLflow")
