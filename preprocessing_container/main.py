# # preprocessing_container/main.py
# from data_utils import *
# import io
# import requests # type: ignore
# from client_utils import get_file, post_file
# import pyarrow as pa # type: ignore
# import pyarrow.parquet as pq # type: ignore

# FASTAPI_URL = "http://fastapi-app:8000"
# BUCKET_NAME = "dataset"
# OBJECT_NAME = "full_dataset.csv"

# file_content = get_file(FASTAPI_URL, BUCKET_NAME, OBJECT_NAME)

# df = read_data(file_content)
# df = handle_nans(df)
# df = scale_data(df)
# # bin_outliers(df)
# df = generate_lags(df, n_lags=5)
# df = time_to_feature(df)

# # Convert the DataFrame to a bytes-like object
# csv_data = df.to_csv(index=True).encode('utf-8')

# post_file(FASTAPI_URL, BUCKET_NAME, "processed.csv", csv_data)

# # out_table = pa.Table.from_pandas(df, preserve_index=True)
# # pq.write_table(out_table, PATH + "processed_data.parquet")


from data_utils import *
import io
<<<<<<< HEAD
import pickle
=======
import requests  # type: ignore
>>>>>>> 108a1859dfa80a64cca629008c7536e9ea034a5f
from client_utils import get_file, post_file
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

FASTAPI_URL = "http://fastapi-app:8000"
INPUT_BUCKET = "dataset"
OUTPUT_BUCKET = "preprocessed"
OBJECT_NAME = "full_dataset.csv"
# Bucket names must be between 3 and 63 characters long, contain only lowercase letters, numbers, and hyphens (-), and must start and end with a letter or number.
OUTPUT_BUCKET = "processed-data"

file_content = get_file(FASTAPI_URL, INPUT_BUCKET, OBJECT_NAME)

df = read_data(file_content)
df = handle_nans(df)
<<<<<<< HEAD
df, scaler = scale_data(df)
# bin_outliers(df)
df = generate_lags(df, n_lags=5)
df = time_to_feature(df)

# ----- Upload the Dataframe to MinIO as a .parquet -----
# Serialize the fitted scaler object into bytes.
serialized_scaler = pickle.dumps(scaler)
scaler_type = scaler.__class__.__name__
# Create the custom metadata dictionary. Keys and values must be bytes.
custom_metadata = {
    b'scaler_object': serialized_scaler,
    b'scaler_type': scaler_type.encode('utf-8')
}
# Infer the schema from the pandas DataFrame
schema = pa.Schema.from_pandas(df)
# Convert the pandas DataFrame to a PyArrow Table.
table = pa.Table.from_pandas(df, schema=schema.with_metadata(custom_metadata))
# Write the PyArrow Table to an in-memory Parquet file (buffer).
parquet_buffer = io.BytesIO()
pq.write_table(table, parquet_buffer)
content_bytes = parquet_buffer.getvalue()

post_file(FASTAPI_URL, OUTPUT_BUCKET, "processed_data.parquet", content_bytes)
=======
df = scale_data(df)
df = generate_lags(df, n_lags=5)
df = time_to_feature(df)

# Serialize to .parquet and upload to MinIO
table = pa.Table.from_pandas(df, preserve_index=True)
buf = io.BytesIO()
pq.write_table(table, buf)
buf.seek(0)

post_file(FASTAPI_URL, OUTPUT_BUCKET, "processed_data.parquet", buf.read())
print("✅ Uploaded processed_data.parquet to MinIO")
>>>>>>> 108a1859dfa80a64cca629008c7536e9ea034a5f
