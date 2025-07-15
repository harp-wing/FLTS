# predict_container/main.py
import numpy as np
import mlflow
import io
from client_utils import get_file, post_file
import os
import pickle
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # type: ignore

FASTAPI_URL = os.environ.get("GATEWAY_URL", "http://fastapi_service:8000")
INFERENCE_LENGTH=3982


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


# === Pull data for model prediction from MinIO ===
input_data_bytes = get_file(FASTAPI_URL, "processed-data", "processed_data.parquet")

if input_data_bytes:
    # 2. Read the schema to access the file's metadata
    # This is more efficient than reading the entire table if you only need metadata.
    schema = pq.read_schema(input_data_bytes)
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

# We add a check to ensure the reshape is possible.
if y_pred.shape[1] % 2 != 0:
    raise ValueError("The number of columns must be an even number to reshape into (2, n).")
print(y_pred.shape)
reshaped_array = y_pred.reshape((2, -1)).T
print(reshaped_array.shape)
subset_scaler = create_subset_scaler(scaler, input_data.columns, ["down", "up"])

df = pd.DataFrame(reshaped_array, columns=["down", "up"])
df = pd.DataFrame(subset_scaler.inverse_transform(df), columns=df.columns)
print(df)

# === Push predictions to MinIO ===

output_table = pa.Table.from_pandas(df)
parquet_buffer = io.BytesIO()
pq.write_table(output_table, parquet_buffer)
content_bytes = parquet_buffer.getvalue()

post_file(FASTAPI_URL, "predictions", "LSTM.parquet", content_bytes)

print("Predictions pushed to MinIO")
