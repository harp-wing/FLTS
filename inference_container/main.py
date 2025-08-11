# predict_container/main.py
from client_utils import get_file, post_file
from kafka_utils import create_producer, create_consumer, produce_message, consume_messages, publish_error
from inferencer import Inferencer
import os
import pickle
import torch
import queue
import threading
import time
import pyarrow.parquet as pq

# --- Environment Variables ---
GATEWAY_URL = os.environ.get("GATEWAY_URL")
if not GATEWAY_URL:
    raise TypeError("Environment variable, GATEWAY_URL, not defined")
PREPROCESSING_TOPIC = os.environ.get("CONSUMER_TOPIC_0") # Topic for preprocessed data claim checks
if not PREPROCESSING_TOPIC:
    raise TypeError("Environment variable, PREPROCESSING_TOPIC, not defined")
TRAINING_TOPIC = os.environ.get("CONSUMER_TOPIC_1") # Topic for trained model claim checks
if not TRAINING_TOPIC:
    raise TypeError("Environment variable, TRAINING_TOPIC, not defined")
CONSUMER_GROUP_ID = os.environ.get("CONSUMER_GROUP_ID", "inference_group") # Consumer Group ID
if not CONSUMER_GROUP_ID:
    raise TypeError("Environment variable, CONSUMER_GROUP_ID, not defined")
PRODUCER_TOPIC = os.environ.get("PRODUCER_TOPIC") # Topic for inference results
if not PRODUCER_TOPIC:
    raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Kafka Message Queue ---
message_queue = queue.Queue() # A single queue to hold messages from both consumers

# --- Kafka Producer for Inference Output and DLQ ---
producer = create_producer()
dlq_topic = f"DLQ-{PRODUCER_TOPIC}"

# --- Kafka Callback Functions Factory ---
def _kafka_callback_factory(service_instance: Inferencer, source_name: str, message_queue_ref: queue.Queue):
    """Creates a callback function for Kafka consumers to put messages into the shared queue."""
    def callback(message):
        print(f"\nConsumer received {source_name} message with key: {message.key} and added to queue.")
        message_queue_ref.put({"source": source_name, "message": message})
    return callback

# --- Worker Thread Function ---
def message_handler(service: Inferencer, message_queue: queue.Queue):
    """
    Worker thread function that processes messages from the queue.
    It dispatches tasks based on the message source (training or preprocessing).
    """
    print("Inference worker thread started. Waiting for messages in queue...")
    while True:
        try:
            queue_item = message_queue.get()
            source = queue_item.get("source")
            message = queue_item.get("message")

            print(f"Inference worker received message from {source} queue with key: {message.key}")

            if source == "training":
                claim_check = message.value
                operation = claim_check.get("operation")
                status = claim_check.get("status")
                experiment = claim_check.get("experiment")
                run_name = claim_check.get("run_name")

                if operation and (status == "SUCCESS") and experiment and run_name:
                    print(f"Inference worker attempting to load new model for experiment '{experiment}', run '{run_name}'.")
                    service.load_model(experiment, run_name)

                    if service.df is not None:
                        service.perform_inference(service.df)
                else:
                    print(f"Inference worker WARN: Training message received without complete details or success status: {claim_check}")
                    publish_error(
                        service.producer,
                        service.dlq_topic,
                        "Training Message Parse",
                        "Failure",
                        "Incomplete training claim check",
                        claim_check
                    )
            elif source == "preprocessing":
                claim_check = message.value
                operation = claim_check.get("operation")
                bucket = claim_check.get("bucket")
                object_key = claim_check.get("object_key")

                if operation == "post: test data" and bucket and object_key:
                    print(f"Inference worker fetching data from object store: s3://{bucket}/{object_key}")
                    try:
                        parquet_bytes = get_file(service.gateway_url, bucket, object_key)
                        table = pq.read_table(source=parquet_bytes)
                        service.df = table.to_pandas()
                        if parquet_bytes:
                            # Read the schema to access the file's metadata
                            schema = pq.read_schema(parquet_bytes)
                            custom_metadata = schema.metadata

                            # 3. Retrieve and deserialize the scaler object
                            serialized_scaler = custom_metadata.get(b'scaler_object')
                            
                            if serialized_scaler:
                                inferencer.current_scaler = pickle.loads(serialized_scaler)
                                scaler_type = custom_metadata.get(b'scaler_type', b'Unknown').decode('utf-8')

                                print("Scaler object retrieved successfully.")
                                print(f"Scaler Type: {scaler_type}")
                                print("Scaler Object:", inferencer.current_scaler)
                            else:
                                print("'scaler_object' not found in the file's metadata.")
                            
                            if service.current_model is not None:
                                service.perform_inference(service.df)

                    except Exception as e:
                        print(f"Inference worker error fetching, parsing, or during inference for {object_key}: {e}")
                        publish_error(
                            service.producer,
                            service.dlq_topic,
                            "Data Fetch/Inference",
                            "Failure",
                            str(e),
                            {"bucket": bucket, "object_key": object_key}
                        )
                else:
                    print(f"Inference worker WARN: Preprocessing message received without complete claim check details or unknown operation: {claim_check}")
                    publish_error(
                        service.producer,
                        service.dlq_topic,
                        "Preprocessing Message Parse",
                        "Failure",
                        "Incomplete preprocessing claim check",
                        claim_check
                    )
            else:
                print(f"Inference worker WARN: Unknown message source: {source}. Message: {message.value}")
                publish_error(
                    service.producer,
                    service.dlq_topic,
                    "Unknown Message Source",
                    "Failure",
                    f"Message from unknown source '{source}'",
                    message.value
                )

        except Exception as e:
            print(f"Inference worker failed to process message from queue: {e}")
            publish_error(
                service.producer,
                service.dlq_topic,
                "Queue Processing",
                "Failure",
                str(e),
                "No specific payload (queue error)"
            )
        finally:
            message_queue.task_done()


inferencer = Inferencer(GATEWAY_URL, producer, dlq_topic, PRODUCER_TOPIC)

# Start the worker thread, passing the service instance and queue
worker_thread = threading.Thread(
    target=message_handler,
    args=(inferencer, message_queue),
    daemon=True
)
worker_thread.start()

#debug
from random import randint
CONSUMER_GROUP_ID = f"CONSUMER_GROUP_ID{randint(0, 999)}"

# Create and start consumers in their own threads
training_consumer = create_consumer(TRAINING_TOPIC, CONSUMER_GROUP_ID)
training_callback_func = _kafka_callback_factory(inferencer, "training", message_queue)
training_consumer_thread = threading.Thread(
    target=consume_messages,
    args=(training_consumer, training_callback_func),
    daemon=True
)
training_consumer_thread.start()
print(f"Started Kafka consumer for training topic: {TRAINING_TOPIC}")

preprocessing_consumer = create_consumer(PREPROCESSING_TOPIC, CONSUMER_GROUP_ID)
preprocessing_callback_func = _kafka_callback_factory(inferencer, "preprocessing", message_queue)
preprocessing_consumer_thread = threading.Thread(
    target=consume_messages,
    args=(preprocessing_consumer, preprocessing_callback_func),
    daemon=True
)
preprocessing_consumer_thread.start()
print(f"Started Kafka consumer for preprocessing topic: {PREPROCESSING_TOPIC}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Inference container stopped by user.")
finally:
    if training_consumer:
        training_consumer.close()
    if preprocessing_consumer:
        preprocessing_consumer.close()
    if producer:
        producer.close()
    print("Kafka consumers and producer closed.")

