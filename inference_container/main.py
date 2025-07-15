# # predict_container/main.py

# import pandas as pd
# import numpy as np
# import os

# from predict import predict, load_model
# from shared.data_utils import to_Xy, generate_time_lags, remove_identifiers, to_timeseries_rep

# def run_inference():
#     df = pd.read_csv("../data/ElBorn.csv", parse_dates=["time"])

#     num_lags = 10
#     targets = ["down", "up"]

#     X_df, y_df = to_Xy(df, targets=targets)
#     X_df = generate_time_lags(X_df, num_lags)
#     y_df = generate_time_lags(y_df, num_lags, is_y=True)
#     X_df, y_df = remove_identifiers(X_df, y_df)

#     num_features = X_df.shape[1] // num_lags
#     X_np = to_timeseries_rep(X_df.to_numpy(), num_lags=num_lags, num_features=num_features)
#     y_np = y_df.to_numpy()

#     model = load_model("../outputs/models/lstm.pt", input_dim=num_features, output_dim=y_np.shape[1], num_lags=num_lags)

#     y_pred = predict(model, X_np)

#     os.makedirs("../outputs/predictions", exist_ok=True)
#     np.save("../outputs/predictions/elborn_preds.npy", y_pred)
#     print("✅ Predictions saved to outputs/predictions/elborn_preds.npy")

# if __name__ == "__main__":
#     run_inference()


# inference_container/main.py

import numpy as np
import mlflow
import io
from client_utils import get_file, post_file
import os
import pandas as pd
import pyarrow.parquet as pq

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi_service:8000")
INFERENCE_LENGTH=3982

# === Pull data for model prediction from MinIO ===
input_data_bytes = get_file(GATEWAY_URL, "processed-data", "processed_data.parquet")
table = pq.read_table(source=input_data_bytes)
input_data = table.to_pandas()
x = input_data.to_numpy().astype(np.float32)

# The model expects an input of shape (batch_size, sequence_length, num_features).
# From the error, we know the sequence_length (T) is 5.
# We will use the last 5 data points to make our prediction.
T = 5
inference_input = x[-T:]

# Reshape the input from (5, 72) to (1, 5, 72) to represent a single sample.
inference_input_reshaped = inference_input.reshape(1, T, x.shape[1])

# === Load model from MLflow ===
experiment_name = "flts-lstm-demo"
runs_df = mlflow.search_runs(
    experiment_names=[experiment_name],
    order_by=["start_time desc"],
    max_results=1
)
# Check if any runs were found
if runs_df.empty:
    raise Exception(f"No runs found in experiment '{experiment_name}'.")

# Get the 'run_id' from the first row of the DataFrame.
run_id = runs_df.loc[0, 'run_id']
print(f"Found run with ID: {run_id}")

# The artifact path is the folder name used when logging the model.
artifact_path = "model"
model_uri = f"runs:/{run_id}/{artifact_path}"

# Load the model
print(f"Loading model from: {model_uri}")
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Listing artifacts to help debug.
    print(f"\nListing artifacts for run ID {run_id} to help find the correct path:")
    try:
        artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
        for artifact in artifacts:
            print(f"- {artifact.path}")
    except Exception as list_e:
        print(f"Could not list artifacts: {list_e}")
    exit()

y_pred = model.predict(inference_input_reshaped)

# === Push predictions to MinIO ===
preds_bytes = io.BytesIO()
np.save(preds_bytes, y_pred)
preds_bytes.seek(0)
post_file(GATEWAY_URL, "predictions", "poblesec_pred.npy", preds_bytes.read())
print("Predictions pushed to MinIO")
