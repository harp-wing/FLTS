import numpy as np
import mlflow # type: ignore
import mlflow.pytorch # type: ignore
import os
import threading
import queue
import pyarrow.parquet as pq
from train import train_model
from data_utils import window_data
from client_utils import get_file
from kafka_utils import create_consumer, consume_messages, create_producer, produce_message, publish_error

GATEWAY_URL = os.environ.get("GATEWAY_URL")
if not GATEWAY_URL:
    raise TypeError("Environment variable, GATEWAY_URL, not defined")
message_queue = queue.Queue()

def callback(message):
    print(f"\nConsumer received message with key: {message.key} and added to queue.")
    message_queue.put(message)

def main(df, experiment_name: str="Default", run_name: str="LSTM", train_test_split: float=0.8, input_seq_len: int=10, output_seq_len: int=1):
    NUM_FEATURES = df.shape[1]
    TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
    TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

    print(df.columns.tolist())

    X, y = window_data(df, TIME_FEATURES, input_len=input_seq_len, output_len=output_seq_len)

    train_size = int(len(X) * train_test_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    config = {
        "device": "cpu",
        "model_name": "lstm",
        "num_features": NUM_FEATURES,
        "num_exogenous_features": len(TIME_FEATURES),
        "input_dim": input_seq_len,
        "output_dim": output_seq_len,
        "hidden_size": 64,
        "num_layers": 4,
        "batch_size": 32,
        "epochs": 40,
        "lr": 0.001
    }

    # === MLflow Logging ===
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()


    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_params(config)
        print(f"[DEBUG] X shape: {X_train.shape}, y shape: {y_train.shape}")
        print(f"[DEBUG] NaNs — y: {np.isnan(y_train).any() | np.isinf(y_train).any()}, X: {np.isnan(X_train).any() | np.isinf(X_train).any()}")

        model = train_model(X_train, y_train, X_test, y_test, config)

        producer = create_producer()
        topic = os.environ.get("PRODUCER_TOPIC")
        if not topic:
            raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

        try:
            mlflow.pytorch.log_model(
                model,
                name="model",
                input_example=X_train[:1],
                registered_model_name=None,
                code_paths=["lstm.py"] # This tells MLflow to bundle lstm.py with the model!
            )
            message = {
                "operation": "Trained: lstm",
                "status": "SUCCESS",
                "experiment": experiment_name,
                "run_name": run_name
            }
            
            produce_message(producer, topic, message)
        except Exception as e:
            publish_error(
                producer,
                f"DLQ-{topic}",
                "MLflow model log",
                "Failure",
                e,
                model
            )

        print("✅ Model logged to MLflow")

def message_handler(gateway_url: str=GATEWAY_URL):
    """
    Worker thread function that processes messages from the queue
    and triggers model training.
    """
    print("Train worker thread started. Waiting for messages in queue...")
    while True:
        try:
            # Block until a message is available in the queue
            claim_check_message = message_queue.get()
            print(f"Train worker received message from queue: {claim_check_message.key}")

            claim_check = claim_check_message.value
            operation = claim_check.get("operation")
            bucket = claim_check.get("bucket")
            object_key = claim_check.get("object_key")
        except Exception as e:
            print(f"Train worker failed to receive message error: {e}")

        if operation == "post: train data" and bucket and object_key:
            print(f"Train worker fetching data from object store: s3://{bucket}/{object_key}")
            try:
                parquet_bytes = get_file(gateway_url, bucket, object_key)
                table = pq.read_table(source=parquet_bytes)
                df = table.to_pandas()
            except Exception as e:
                print(f"Train worker error fetching or parsing data for {object_key}: {e}")
            

            print(f"Train worker starting model training for data from {object_key}...")
            try:

                main(df) # TRAINING LOGIC CALL

                print(f"Train worker finished model training for data from {object_key}.")

            except Exception as e:
                print(f"Train worker error during model training for {object_key}: {e}")
        else:
            print(f"Train worker WARN: Message received without complete claim check details or unknown operation: {claim_check}")

        # Mark the task as done after processing
        message_queue.task_done()

worker_thread = threading.Thread(target=message_handler, daemon=True)
worker_thread.start()

consumer = create_consumer(os.environ.get("CONSUMER_TOPIC"), os.environ.get("CONSUMER_GROUP_ID"))
consume_messages(consumer, callback)
