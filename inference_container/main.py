# predict_container/main.py
import numpy as np
import mlflow # type: ignore
import io
from client_utils import get_file, post_file
from data_utils import window_data, check_uniform, time_to_feature
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # type: ignore

FASTAPI_URL = os.environ.get("GATEWAY_URL", "http://fastapi_service:8000")
parquet_bytes = get_file(FASTAPI_URL, "processed-data", "test_processed_data.parquet")
table = pq.read_table(source=parquet_bytes)
df_eval = table.to_pandas()

TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
FEATURES = df_eval.columns.difference(TIME_FEATURES, sort=False).tolist()

N_ENDO_FEATURES = len(FEATURES)

# To be imported from other containers
MODEL_NAME = "LSTM"
OUTPUT_SEQ_LEN = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_IDX = 0  # Index to start prediction from
INFERENCE_LENGTH = 720  # Number of steps to predict


X_eval, y_eval = window_data(df_eval, TIME_FEATURES)

def create_subset_scaler(original_scaler, original_columns, subset_columns):
    """
    Creates a new scaler for a subset of features from an existing fitted scaler.
    This version works for both StandardScaler and MinMaxScaler.

    Args:
        original_scaler: The scaler object (StandardScaler or MinMaxScaler).
        original_columns (pd.Index): The .columns attribute from the original DataFrame.
        subset_columns (list): A list of column names for the new scaler.

    Returns:
        A new, configured scaler for the subset of data.
    """
    if original_columns == subset_columns:
        return original_scaler

    # Find the integer indices of the subset columns
    subset_indices = [original_columns.get_loc(col) for col in subset_columns]

    # Check scaler type and assign the correct attributes
    if isinstance(original_scaler, StandardScaler):
        subset_scaler = StandardScaler()
        # StandardScaler uses 'mean_' and 'scale_' (std dev)
        if original_scaler.mean_ is None or original_scaler.scale_ is None:
            raise ValueError("The original StandardScaler is not fitted or missing required attributes.")
        subset_scaler.mean_ = original_scaler.mean_[subset_indices]
        subset_scaler.scale_ = original_scaler.scale_[subset_indices]
    elif isinstance(original_scaler, MinMaxScaler):
        subset_scaler = MinMaxScaler()
        # MinMaxScaler uses 'min_' and 'scale_' (feature range)
        if original_scaler.min_ is None or original_scaler.scale_ is None:
            raise ValueError("The original MinMaxScaler is not fitted or missing required attributes.")
        subset_scaler.min_ = original_scaler.min_[subset_indices]
        subset_scaler.scale_ = original_scaler.scale_[subset_indices]
    else:
        raise TypeError("Unsupported scaler type. This function only supports StandardScaler and MinMaxScaler.")

    # Set feature info for scikit-learn validation
    subset_scaler.n_features_in_ = len(subset_columns)
    subset_scaler.feature_names_in_ = np.array(subset_columns, dtype=object)

    return subset_scaler


if parquet_bytes:
    # 2. Read the schema to access the file's metadata
    # This is more efficient than reading the entire table if you only need metadata.
    schema = pq.read_schema(parquet_bytes)
    custom_metadata = schema.metadata

    # 3. Retrieve and deserialize the scaler object
    serialized_scaler = custom_metadata.get(b'scaler_object')
    
    if serialized_scaler:
        scaler = pickle.loads(serialized_scaler)
        
        # You can also retrieve the scaler type you saved
        scaler_type = custom_metadata.get(b'scaler_type', b'Unknown').decode('utf-8')

        print("✅ Scaler object retrieved successfully!")
        print(f"Scaler Type: {scaler_type}")
        print("Scaler Object:", scaler)
    else:
        print("❌ 'scaler_object' not found in the file's metadata.")


# === Load model from MLflow ===
experiment_name = "Default"
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

X_eval_tensor = torch.from_numpy(X_eval).float()
y_eval_tensor = torch.from_numpy(y_eval).float()

eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)


# Determine the sampling frequency from the data
timedelta = check_uniform(df_eval)

# How much real data is available for the forecast window
remaining_real_data = X_eval.shape[0] - SAMPLE_IDX
available_future_steps = min(remaining_real_data, INFERENCE_LENGTH)
num_extension_steps = INFERENCE_LENGTH - available_future_steps

df_predictions = pd.DataFrame(
    index=pd.date_range(
        start=df_eval.index[SAMPLE_IDX],
        periods=INFERENCE_LENGTH,
        freq=timedelta
    ),
    columns=df_eval.columns
)

df_predictions = time_to_feature(df_predictions)

current_sequence = X_eval_tensor[SAMPLE_IDX].unsqueeze(0).to(device)

with torch.no_grad():
    step = 0
    while step < INFERENCE_LENGTH:
        multi_step_pred = model.predict(current_sequence.cpu().numpy())   # Shape: (1, OUTPUT_SEQ_LEN, num_features)
        remaining_steps = INFERENCE_LENGTH - step   # Decide how many of the predicted steps to use in this loop
        steps_to_use = min(OUTPUT_SEQ_LEN, remaining_steps)

        for i in range(steps_to_use):
            absolute_step = step + i
            next_step = absolute_step + 1
            if absolute_step >= INFERENCE_LENGTH:
                break

            # Get prediction and store it
            current_pred = multi_step_pred[:, i, :].flatten()
            df_predictions.loc[df_predictions.index[absolute_step], FEATURES] = current_pred

            # Update sequence
            if next_step <= available_future_steps:
                current_sequence = X_eval_tensor[SAMPLE_IDX + next_step].unsqueeze(0).to(device)
            else:
                extension_idx = next_step - available_future_steps
                print("CALLED EXTENSION MODE")

                if extension_idx < df_predictions.shape[0]:
                    # Extract future exogenous data from df_extension
                    extension_row = df_predictions.iloc[[extension_idx]][TIME_FEATURES]
                    numpy_extension, _ = window_data(
                        extension_row,
                        TIME_FEATURES,
                        input_len=1, output_len=1
                    )
                    exog_tensor = torch.from_numpy(numpy_extension).float()
                    pred_tensor = torch.tensor(current_pred).view(1, 1, -1)
                    print(f"Current Sequence shape: {current_sequence.shape}")
                    print(f"Prediction Tensor shape: {pred_tensor.shape}")
                    print(f"Exogenous Tensor shape: {exog_tensor.shape}")

                    current_pred = torch.cat((pred_tensor, exog_tensor), dim=-1)
                    current_sequence = torch.cat((current_sequence[:, 1:, :], current_pred), dim=1)
                else:
                    print(f"[Warning] df_extension exhausted at index {extension_idx}")
                    break

        step += steps_to_use


df_predictions = df_predictions.drop(columns=TIME_FEATURES)
df = pd.DataFrame(
    scaler.inverse_transform(df_predictions),
    index=df_predictions.index,
    columns=df_predictions.columns
)

print(f"Inference completed:")
print(f"- Used actual future values for first {min(available_future_steps, INFERENCE_LENGTH)} steps")
if INFERENCE_LENGTH > available_future_steps:
    print(f"- Switched to recursive mode after step {available_future_steps}")
print(f"- Model predicts {OUTPUT_SEQ_LEN} step(s) at a time")
print(f"- Total predictions generated: {df.shape[0]}")

output_table = pa.Table.from_pandas(df)
parquet_buffer = io.BytesIO()
pq.write_table(output_table, parquet_buffer)
content_bytes = parquet_buffer.getvalue()

post_file(FASTAPI_URL, "predictions", f"{MODEL_NAME}.parquet", content_bytes)

print("Predictions pushed to MinIO")
