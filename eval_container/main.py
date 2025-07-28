# eval_container/main.py
import io
import asyncio
import base64
import queue
import threading
import time
from functools import wraps
from client_utils import get_file
from data_utils import read_data
from kafka_utils import create_consumer, create_producer, consume_messages, produce_message, publish_error
from storage_classes import PredictionStore, TruthStore
from fastapi import FastAPI, Request                                            # type: ignore
from fastapi.responses import HTMLResponse                                      # type: ignore
from fastapi.templating import Jinja2Templates                                  # type: ignore
from starlette.concurrency import run_in_threadpool                             # type: ignore
import pandas as pd                                                             # type: ignore
import pyarrow.parquet as pq                                                    # type: ignore
import matplotlib.pyplot as plt                                                 # type: ignore
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas     # type: ignore

from metrics import *

GATEWAY_URL = os.environ.get("GATEWAY_URL")
if not GATEWAY_URL:
    raise TypeError("Environment variable, GATEWAY_URL, not defined")
TRUTH_TOPIC = os.environ.get("CONSUMER_TOPIC_0")
if not TRUTH_TOPIC:
    raise TypeError("Environment variable, TRUTH_DATA_TOPIC, not defined")
INFERENCE_TOPIC = os.environ.get("CONSUMER_TOPIC_1")  # Topic for inference results
if not INFERENCE_TOPIC:
    raise TypeError("Environment variable, INFERENCE_TOPIC, not defined")
CONSUMER_GROUP_ID = os.environ.get("CONSUMER_GROUP_ID", "eval_group")
if not CONSUMER_GROUP_ID:
    raise TypeError("Environment variable, CONSUMER_GROUP_ID, not defined")
PRODUCER_TOPIC = os.environ.get("PRODUCER_TOPIC")  # Topic for evaluation results
if not PRODUCER_TOPIC:
    raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")


# --- Kafka Callback Functions ---
def kafka_callback_factory(source_name: str, message_queue_ref: queue.Queue):
    """Creates a callback function for Kafka consumers to put messages into the shared queue."""
    def callback(message):
        print(f"\nEval consumer received {source_name} message with key: {message.key} and added to queue.")
        message_queue_ref.put({"source": source_name, "message": message})
    return callback

# --- Worker Thread Function ---
def message_handler(message_queue: queue.Queue):
    """
    Worker thread function that processes messages from the queue.
    It handles both inference result claim checks and ground truth data claim checks.
    """
    print("Eval worker thread started. Waiting for messages in queue...")
    while True:
        try:
            queue_item = message_queue.get()
            source = queue_item.get("source")
            message = queue_item.get("message")

            print(f"Eval worker received message from {source} queue with key: {message.key}")

            if source == "inference":
                claim_check = message.value
                operation = claim_check.get("operation")
                status = claim_check.get("status")
                bucket = claim_check.get("bucket")
                object_key = claim_check.get("object_key")

                if (operation == "Inference") and (status == "SUCCESS") and bucket and object_key:
                    print(f"Eval worker fetching prediction data from object store: s3://{bucket}/{object_key}")
                    try:
                        # Fetch prediction data from MinIO
                        parquet_bytes = get_file(GATEWAY_URL, bucket, object_key) # type: ignore
                        table = pq.read_table(source=parquet_bytes)
                        df_pred = table.to_pandas()
                        
                        # Extract model information from object key or metadata
                        model_name = object_key.split('_')[0] if '_' in object_key else "Unknown"
                        
                        # Store prediction with metadata
                        metadata = {
                            "timestamp": time.time(),
                            "object_key": object_key,
                            "bucket": bucket,
                            "shape": df_pred.shape
                        }
                        
                        prediction_store.add_prediction(model_name, df_pred, metadata)
                        
                        print(f"Eval worker processed prediction from {object_key}")
                        print(f"Prediction shape: {df_pred.shape}")

                        # Publish success message
                        produce_message(producer, PRODUCER_TOPIC, {
                            "operation": "Evaluation Data Received",
                            "status": "SUCCESS",
                            "model_name": model_name,
                            "bucket": bucket,
                            "object_key": object_key,
                            "prediction_shape": df_pred.shape
                        })

                    except Exception as e:
                        print(f"Eval worker error fetching or processing prediction data from {object_key}: {e}")
                        publish_error(
                            producer,
                            dlq_topic,
                            "Prediction Data Fetch/Process",
                            "Failure",
                            str(e),
                            {"bucket": bucket, "object_key": object_key}
                        )
                else:
                    print(f"Eval worker WARN: Inference message received without complete claim check details or non-success status: {claim_check}")
                    publish_error(
                        producer,
                        dlq_topic,
                        "Inference Message Parse",
                        "Failure",
                        "Incomplete inference claim check or non-success status",
                        claim_check
                    )
            
            elif source == "ground_truth":
                claim_check = message.value
                operation = claim_check.get("operation")
                status = claim_check.get("status")
                source_bucket = claim_check.get("source_bucket")
                source_object = claim_check.get("source_object")

                if status == "SUCCESS" and source_bucket and source_object:
                    print(f"Eval worker fetching ground truth data from object store: s3://{source_bucket}/{source_object}")
                    try:
                        file_bytes = get_file(GATEWAY_URL, source_bucket, source_object) # type: ignore
                        df_true = read_data(file_bytes)


                        metadata = {
                            "timestamp": time.time(),
                            "bucket": source_bucket,
                            "object_key": source_object,
                            "shape": df_true.shape
                        }
        
                        truth_store.set_ground_truth(df_true, metadata)
                        print(f"Ground truth data loaded successfully from {source_bucket}/{source_object}")
                        
                        if df_true is not None:
                            # Publish success message
                            produce_message(producer, PRODUCER_TOPIC, {
                                "operation": "Ground Truth Data Received",
                                "status": "SUCCESS",
                                "bucket": source_bucket,
                                "object_key": source_object
                            })
                        else:
                            raise Exception("Failed to load ground truth data")

                    except Exception as e:
                        print(f"Eval worker error loading ground truth data from {source_object}: {e}")
                        publish_error(
                            producer,
                            dlq_topic,
                            "Ground Truth Data Load",
                            "Failure",
                            str(e),
                            {"bucket": source_bucket, "object_key": source_object}
                        )
                else:
                    print(f"Eval worker WARN: Ground truth message received without complete claim check details or non-success status: {claim_check}")
                    publish_error(
                        producer,
                        dlq_topic,
                        "Ground Truth Message Parse",
                        "Failure",
                        "Incomplete ground truth claim check or non-success status",
                        claim_check
                    )
            
            else:
                print(f"Eval worker WARN: Unknown message source: {source}. Message: {message.value}")
                publish_error(
                    producer,
                    dlq_topic,
                    "Unknown Message Source",
                    "Failure",
                    f"Message from unknown source '{source}'",
                    message.value
                )

        except Exception as e:
            print(f"Eval worker failed to process message from queue: {e}")
            publish_error(
                producer,
                dlq_topic,
                "Queue Processing",
                "Failure",
                str(e),
                "No specific payload (queue error)"
            )
        finally:
            message_queue.task_done()

message_queue = queue.Queue()
producer = create_producer()
dlq_topic = "DLQ-Evaluation"

prediction_store = PredictionStore()
truth_store = TruthStore()

worker_thread = threading.Thread(
    target=message_handler,
    args=(message_queue,),
    daemon=True
)
worker_thread.start()

# Debug - randomize consumer group to avoid conflicts
from random import randint
CONSUMER_GROUP_ID = f"{CONSUMER_GROUP_ID}_{randint(0, 999)}"

inference_consumer = create_consumer(INFERENCE_TOPIC, CONSUMER_GROUP_ID)
inference_callback_func = kafka_callback_factory("inference", message_queue)
inference_consumer_thread = threading.Thread(
    target=consume_messages,
    args=(inference_consumer, inference_callback_func),
    daemon=True
)
inference_consumer_thread.start()

ground_truth_consumer = create_consumer(TRUTH_TOPIC, CONSUMER_GROUP_ID)
ground_truth_callback_func = kafka_callback_factory("ground_truth", message_queue)
ground_truth_consumer_thread = threading.Thread(
    target=consume_messages,
    args=(ground_truth_consumer, ground_truth_callback_func),
    daemon=True
)
ground_truth_consumer_thread.start()



# Create an instance of FastAPI
app = FastAPI()

# This will look for a directory named "templates" in the same directory as main.py
templates = Jinja2Templates(directory="templates")

def plot_to_base64(func):
    """
    A decorator that takes a plot-generating function,
    and handles the boilerplate of saving it to a BytesIO buffer,
    encoding it to base64, and closing the figure.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the original plot function to get the figure and title
        fig, title = func(*args, **kwargs)
        
        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        
        # Encode the buffer's content to a base64 string
        data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # IMPORTANT: Close the figure to free up memory
        plt.close(fig)
        
        # Return the formatted dictionary
        return {"title": title, "data": data}
    return wrapper

# Decorate the metrics plotting functions (Doing it here so the metrics function maintain figure return)
_plot_predictions = plot_to_base64(plot_predictions)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    This is the main endpoint for the application
    """
    # Get the latest prediction data
    predictions = prediction_store.get_predictions()
    df_pred = predictions["LSTM"]
    df_true = truth_store.get_ground_truth()
    
    if df_pred is None:
        # If no predictions available, return a message
        return templates.TemplateResponse(
            "index.html",
            {"request": request, 
             "title": "Evaluation Dashboard", 
             "plots": [],
             "message": "No prediction data available yet. Waiting for inference results..."},
        )
    
    if df_true is None:
        # If no ground truth available, return a message
        return templates.TemplateResponse(
            "index.html",
            {"request": request, 
             "title": "Evaluation Dashboard", 
             "plots": [],
             "message": "No ground truth data available yet. Waiting for ground truth data..."},
        )
    
    # Ensure ground truth data is aligned with prediction data
    df_true_aligned = df_true.iloc[:df_pred.shape[0], :]

    # Adjustable graph generator
    PLOTS = {
        "down" : {
            "title": "Comparison of Actual vs Predicted Down",
            "generator": lambda: _plot_predictions(df_true_aligned.index, df_true_aligned["down"], df_pred["down"], "Comparison of Actual vs Predicted Down")
        },
        "up" : {
            "title": "Comparison of Actual vs Predicted Up",
            "generator": lambda: _plot_predictions(df_true_aligned.index, df_true_aligned["up"], df_pred["up"], "Comparison of Actual vs Predicted Up")
        },
        "rnti_count" : {
            "title": "Comparison of Actual vs Predicted RNTI Count",
            "generator": lambda: _plot_predictions(df_true_aligned.index, df_true_aligned["rnti_count"], df_pred["rnti_count"], "Comparison of Actual vs Predicted RNTI Count")
        },
    }

    plot_functions = [plot_info["generator"] for plot_info in PLOTS.values()]
    
    # create plots in parallel
    tasks = [run_in_threadpool(func) for func in plot_functions] # type: ignore
    plot_results = await asyncio.gather(*tasks)
    
    # Get metadata for display
    prediction_metadata = prediction_store.get_metadata()
    ground_truth_metadata = truth_store.get_metadata()
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, 
         "title": "Evaluation Dashboard", 
         "plots": plot_results,
         "prediction_metadata": prediction_metadata,
         "ground_truth_metadata": ground_truth_metadata},
    )
