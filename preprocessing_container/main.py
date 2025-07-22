# preprocessing_container/main.py
from data_utils import *
import io
import pickle
from client_utils import get_file, post_file
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

FASTAPI_URL = "http://fastapi-app:8000"
INPUT_BUCKET = "dataset"
OBJECT_NAME = "full_dataset.csv"
TEST_OBJECT_NAME = "PobleSec_test.csv"

# Bucket names must be between 3 and 63 characters long, contain only lowercase letters, numbers, and hyphens (-), and must start and end with a letter or number.
OUTPUT_BUCKET = "processed-data"

file_content = get_file(FASTAPI_URL, INPUT_BUCKET, OBJECT_NAME)
test_file_content = get_file(FASTAPI_URL, INPUT_BUCKET, TEST_OBJECT_NAME)


df = read_data(file_content, "PobleSec")
df = handle_nans(df)
df, scaler = scale_data(df)
# bin_outliers(df)
df = time_to_feature(df)

test_df = read_data(test_file_content)
test_df = handle_nans(test_df)
test_df, _ = scale_data(test_df, scale=scaler)
test_df = time_to_feature(test_df)

print(f"df: {df}, test_df: {test_df}")

#----------------------------------
# ---- Export data to Parquet ----
#----------------------------------

serialized_scaler = pickle.dumps(scaler)
scaler_type = scaler.__class__.__name__
custom_metadata = {
    b'scaler_object': serialized_scaler,
    b'scaler_type': scaler_type.encode('utf-8')
}

print(f"Scaler: {scaler_type}")

schema = pa.Schema.from_pandas(df)
table = pa.Table.from_pandas(df, schema=schema.with_metadata(custom_metadata), preserve_index=True)
parquet_buffer = io.BytesIO()
pq.write_table(table, parquet_buffer)
content_bytes = parquet_buffer.getvalue()

post_file(FASTAPI_URL, OUTPUT_BUCKET, "processed_data.parquet", content_bytes)


serialized_scaler = pickle.dumps(scaler)
scaler_type = scaler.__class__.__name__
custom_metadata = {
    b'scaler_object': serialized_scaler,
    b'scaler_type': scaler_type.encode('utf-8')
}
schema = pa.Schema.from_pandas(test_df)
table = pa.Table.from_pandas(test_df, schema=schema.with_metadata(custom_metadata), preserve_index=True)

parquet_buffer = io.BytesIO()
pq.write_table(table, parquet_buffer)
content_bytes = parquet_buffer.getvalue()

post_file(FASTAPI_URL, OUTPUT_BUCKET, "test_processed_data.parquet", content_bytes)
