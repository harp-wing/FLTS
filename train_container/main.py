import numpy as np
import mlflow # type: ignore
import mlflow.pytorch # type: ignore
from train import train_model
from data_utils import window_data
from client_utils import get_file
import os
import pyarrow.parquet as pq

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi_service:8000")

# === Download and read .parquet file ===
parquet_bytes = get_file(GATEWAY_URL, "processed-data", "processed_data.parquet")
table = pq.read_table(source=parquet_bytes)
df = table.to_pandas()
parquet_bytes = get_file(GATEWAY_URL, "processed-data", "test_processed_data.parquet")
table = pq.read_table(source=parquet_bytes)
test_df = table.to_pandas()


NUM_FEATURES = df.shape[1]
TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

INPUT_SEQUENCE_LENGTH = 10
OUTPUT_SEQUENCE_LENGTH = 1
TRAIN_TEST_SPLIT = 0.8

print(df.columns.tolist())

X, y = window_data(df, TIME_FEATURES, input_len=INPUT_SEQUENCE_LENGTH, output_len=OUTPUT_SEQUENCE_LENGTH)

train_size = int(len(X) * TRAIN_TEST_SPLIT)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

config = {
    "device": "cpu",
    "model_name": "lstm",
    "num_features": NUM_FEATURES,
    "num_exogenous_features": len(TIME_FEATURES),
    "output_dim": OUTPUT_SEQUENCE_LENGTH,
    "hidden_size": 64,
    "num_layers": 4,
    "batch_size": 32,
    "epochs": 40,
    "lr": 0.001
}

# === MLflow Logging ===
# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Default")
mlflow.autolog()


with mlflow.start_run(run_name="LSTM", log_system_metrics=True):
    mlflow.log_params(config)
    print(f"[DEBUG] X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"[DEBUG] NaNs — y: {np.isnan(y_train).any() | np.isinf(y_train).any()}, X: {np.isnan(X_train).any() | np.isinf(X_train).any()}")

    model = train_model(X_train, y_train, X_test, y_test, config)

    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=X_train[:1],
        registered_model_name=None,
        code_paths=["lstm.py"] # This tells MLflow to bundle m1.py with the model!
    )

    print("✅ Model logged to MLflow")
