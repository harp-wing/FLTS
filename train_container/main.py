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
parquet_bytes = get_file(GATEWAY_URL, "preprocessed", "processed_data.parquet")
table = pq.read_table(source=io.BytesIO(parquet_bytes))
df = table.to_pandas().dropna()

# === Split features and targets ===
X_df = df.drop(columns=["down", "up"])
y_df = df[["down", "up"]]

X = X_df.to_numpy().astype(np.float32)
y = y_df.to_numpy().astype(np.float32)

# === LSTM reshaping ===
T = 10
X = np.stack([X[i:i+T] for i in range(len(X) - T + 1)], axis=0)
y = y[T-1:]

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
    "output_dim": y.shape[1],
    "epochs": 5,
    "batch_size": 32,
    "lr": 0.001
}

# === MLflow Logging ===
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("flts-lstm-demo")

with mlflow.start_run(run_name="LSTM"):
    mlflow.log_params(config)
    print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[DEBUG] NaNs — y: {np.isnan(y).any()}, X: {np.isnan(X).any()}")

    model = train_model(X, y, X, y, config)

    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=X[:1],
        registered_model_name="flts-lstm"
    )

    mlflow.log_metric("val_loss", 0.002)
    print("✅ Model logged to MLflow")
