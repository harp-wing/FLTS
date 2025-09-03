# preprocessing_container/main.py
from data_utils import *
import io
import pickle
import traceback
from druid_utils import DruidIngester
from client_utils import get_file, post_file
from kafka_utils import create_producer, produce_message, publish_error
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

FASTAPI_URL = "http://fastapi-app:8000"
IDENTIFIER = "PobleSec"
INPUT_BUCKET = "dataset"
OBJECT_NAME = "full_dataset.csv"
TEST_OBJECT_NAME = f"{IDENTIFIER}_test.csv"

# Bucket names must be between 3 and 63 characters long, contain only lowercase letters, numbers, and hyphens (-), and must start and end with a letter or number.
OUTPUT_BUCKET = "processed-data"
OUTPUT_FILENAME = "processed_data"
OUTPUT_TEST_FILENAME = "test_processed_data"

file_content = get_file(FASTAPI_URL, INPUT_BUCKET, OBJECT_NAME)
test_file_content = get_file(FASTAPI_URL, INPUT_BUCKET, TEST_OBJECT_NAME)


#----------------------------------
# ------- Preprocess Data --------
#----------------------------------

SCALER = os.environ.get("SCALER", None)
ADD_VAL = os.environ.get("ADD_VAL", None)

druid = DruidIngester()

df = read_data(file_content, IDENTIFIER)
df = handle_nans(df)
# df = clip_outliers(df, method="percentile", factor=0.25) # or bin_outliers(df) <-- potential method to train different models for spikes vs baseline

if SCALER is None:
    scaler = None
else:
    df, scaler = scale_data(df, SCALER)
if ADD_VAL is not None:
    df.add(float(ADD_VAL))
df = time_to_feature(df)

test_df = read_data(test_file_content)
test_df = handle_nans(test_df)

druid_df = test_df.reset_index(names="time")
task_id = druid.ingest_dataframe(druid_df, f"{IDENTIFIER}_test", "time")
if task_id:
    print(f"Data ingested successfully. Task ID: {task_id}")
else:
    print("Failed to ingest data")

# test_df = clip_outliers(test_df, method="percentile", factor=0.1)
test_df, _ = scale_data(test_df, scale=scaler)
if ADD_VAL is not None:
    test_df.add(float(ADD_VAL))
test_df = time_to_feature(test_df)

print(f"df: {df.head(3)}, test_df: {test_df.head(3)}")


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
test_content_bytes = parquet_buffer.getvalue()



#----------------------------------
# ---- Publish to Kafka Topic ----
#----------------------------------

producer = create_producer()
TRAIN_TOPIC = os.environ.get("PRODUCER_TOPIC_0")
TEST_TOPIC = os.environ.get("PRODUCER_TOPIC_1")

try:
    post_file(FASTAPI_URL, OUTPUT_BUCKET, f"{OUTPUT_FILENAME}.parquet", content_bytes)
    message = {
        "operation": "post: train data",
        "status": "SUCCESS",
        "bucket": OUTPUT_BUCKET,
        "object_key": f"{OUTPUT_FILENAME}.parquet",
        "identifier": IDENTIFIER
    }
    produce_message(producer, TRAIN_TOPIC, message)
except Exception as e:
    error_details = f"Failed to post train data: {str(e)}\n{traceback.format_exc()}"
    payload = {
        "endpoint": FASTAPI_URL,
        "bucket": OUTPUT_BUCKET,
        "object": f"{OUTPUT_FILENAME}.parquet",
        "content": content_bytes
    }
    publish_error(
        producer, 
        dlq_topic=f"DLQ-{TRAIN_TOPIC}",
        operation="Post File to MinIO",
        status="Failure",
        error_details=error_details,
        payload=payload
    )

try:
    post_file(FASTAPI_URL, OUTPUT_BUCKET, f"{OUTPUT_TEST_FILENAME}.parquet", test_content_bytes)
    message = {
        "operation": "post: test data",
        "status": "SUCCESS",
        "bucket": OUTPUT_BUCKET,
        "object_key": f"{OUTPUT_TEST_FILENAME}.parquet",
        "identifier": IDENTIFIER,
        "source_bucket": INPUT_BUCKET,
        "source_object": TEST_OBJECT_NAME
    }
    produce_message(producer, TEST_TOPIC, message)
except Exception as e:
    error_details = f"Failed to post test data: {str(e)}\n{traceback.format_exc()}"
    payload = {
        "endpoint": FASTAPI_URL,
        "bucket": OUTPUT_BUCKET,
        "object": f"{OUTPUT_TEST_FILENAME}.parquet",
        "content": content_bytes
    }
    publish_error(
        producer, 
        dlq_topic=f"DLQ-{TEST_TOPIC}",
        operation="Post File to MinIO",
        status="Failure",
        error_details=error_details,
        payload=payload
    )
