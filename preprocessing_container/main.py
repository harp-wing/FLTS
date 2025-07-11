# preprocessing_container/main.py
from data_utils import *
import io
import requests # type: ignore
from client_utils import get_file, post_file
import pyarrow as pa # type: ignore
import pyarrow.parquet as pq # type: ignore

FASTAPI_URL = "http://fastapi-app:8000"
BUCKET_NAME = "dataset"
OBJECT_NAME = "full_dataset.csv"

file_content = get_file(FASTAPI_URL, BUCKET_NAME, OBJECT_NAME)

df = read_data(file_content)
df = handle_nans(df)
df = scale_data(df)
# bin_outliers(df)
df = generate_lags(df, n_lags=5)
df = time_to_feature(df)

# Convert the DataFrame to a bytes-like object
csv_data = df.to_csv(index=True).encode('utf-8')

post_file(FASTAPI_URL, BUCKET_NAME, "processed.csv", csv_data)

# out_table = pa.Table.from_pandas(df, preserve_index=True)
# pq.write_table(out_table, PATH + "processed_data.parquet")