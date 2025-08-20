# nonML container
import os
import queue
import threading
import mlflow # type: ignore
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pyarrow.parquet as pq
from typing import Any, List, Tuple, Optional
from client_utils import get_file
from data_utils import check_uniform, subset_scaler
from kafka_utils import create_producer, create_consumer, produce_message, consume_messages, publish_error
from models import ProphetMultiFeatureModel, StatsForecastMultiFeatureModel

def env_var(var: str, default: Any=None) -> str:
    temp = os.environ.get(var, default)
    if temp is None:
        raise TypeError(f"Environment variable, {var}, not defined")
    else:
        return temp
    
def estimate_season_length(td: pd.Timedelta) -> int:
    """
    Infer smallest likely season length (in steps) greater than the data periodicity.
    
    Examples:
        1 min  -> 1440 (daily seasonality)
        5 min  -> 288  (daily seasonality)
        1 hour -> 24   (daily seasonality)
        1 day  -> 7    (weekly seasonality)
        1 week -> 52   (yearly seasonality)
    """
    seconds = td.total_seconds()

    # Define "likely" real-world seasonal cycles in seconds
    cycles = {
        "daily":   24 * 3600,
        "weekly":  7  * 24 * 3600,
        "yearly":  365 * 24 * 3600
    }

    # Find the smallest cycle > periodicity
    for name, cycle_seconds in cycles.items():
        if cycle_seconds > seconds:
            return int(round(cycle_seconds / seconds))

    # If the periodicity is larger than yearly, default to 1
    return 1

def callback(message):
    print(f"\nConsumer received message with key: {message.key} and added to queue.")
    message_queue.put(message)

def message_handler():
    """
    Worker thread function that processes messages from the queue.
    """
    print("NonML worker thread started. Waiting for messages in queue...")
    GATEWAY_URL = env_var("GATEWAY_URL")
    while True:
        try:
            # Block until a message is available in the queue
            claim_check_message = message_queue.get()
            print(f"NonML worker received message from queue: {claim_check_message.key}")

            claim_check = claim_check_message.value
            operation = claim_check.get("operation")
            bucket = claim_check.get("bucket")
            object_key = claim_check.get("object_key")
        except Exception as e:
            print(f"NonML worker failed to receive message error: {e}")

        if operation == "post: train data" and bucket and object_key:
            print(f"NonML worker fetching data from object store: s3://{bucket}/{object_key}")
            try:
                parquet_bytes = get_file(GATEWAY_URL, bucket, object_key)
                table = pq.read_table(source=parquet_bytes)
                df = table.to_pandas()
            except Exception as e:
                print(f"NonML worker error fetching or parsing data for {object_key}: {e}")
            

            print(f"NonML worker starting model training for data from {object_key}...")
            try:

                main(df)

                print(f"NonML worker finished model training for data from {object_key}.")

            except Exception as e:
                print(f"NonML worker error during model training for {object_key}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"NonML worker WARN: Message received without complete claim check details or unknown operation: {claim_check}")

        # Mark the task as done after processing
        message_queue.task_done()

def main(df: pd.DataFrame, experiment_name: str = "NonML"):
    OUTPUT_SEQ_LEN: int = int(os.environ.get("OUTPUT_SEQ_LEN", "1"))
    MODEL_TYPE: str = env_var("MODEL_TYPE")

    TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
    TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
    
    # === MLflow Logging ===
    mlflow.set_experiment(experiment_name)

    run_name = MODEL_TYPE

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        # mlflow.autolog()
        # scikit-learn==1.7.0
        # statsmodels==0.14.4

        # Prepare data
        timedelta = check_uniform(df)
        offset = to_offset(timedelta).freqstr # type: ignore

        df["unique_id"] = "1"
        df.index.rename("ds", inplace=True)
        df = df.reset_index()

        # REDUCE FEATURES HERE
        # ex:
        # scaler = subset_scaler(scaler, df.columns.to_list(), trims)

        # Get feature columns (excluding ds and unique_id)
        feature_columns = [col for col in df.columns if col not in (["ds", "unique_id"]+TIME_FEATURES)]
        
        print(f"Feature columns: {feature_columns}")
        print(f"Features: {df.columns.difference(feature_columns, sort=False).to_list()}")

        if MODEL_TYPE == "PROPHET":
            # Collect Prophet parameters
            prophet_params = {
                "growth": os.environ.get("GROWTH", "linear"),
                "n_changepoints": int(os.environ.get("N_CHANGEPOINTS", "25")),
                "changepoint_range": float(os.environ.get("CHANGEPOINT_RANGE", "0.8")),
                "yearly_seasonality": os.environ.get("YEARLY_SEASONALITY", "auto"),
                "weekly_seasonality": os.environ.get("WEEKLY_SEASONALITY", "auto"),
                "daily_seasonality": os.environ.get("DAILY_SEASONALITY", "auto"),
                "seasonality_mode": os.environ.get("SEASONALITY_MODE", "additive"),
                "seasonality_prior_scale": float(os.environ.get("SEASONALITY_PRIOR_SCALE", "10")),
                "holidays_prior_scale": float(os.environ.get("HOLIDAYS_PRIOR_SCALE", "10")),
                "changepoint_prior_scale": float(os.environ.get("CHANGEPOINT_PRIOR_SCALE", "0.05")),
                "country": os.environ.get("COUNTRY", "US")
            }
            
            # Log all Prophet parameters
            mlflow.log_params({f"{k}": v for k, v in prophet_params.items()})
            
            # Create and fit multi-feature Prophet model
            multi_prophet = ProphetMultiFeatureModel()
            multi_prophet.fit(df, feature_columns, prophet_params)
            
            # Log the bundled model
            mlflow.pyfunc.log_model(
                name=MODEL_TYPE,
                python_model=multi_prophet,
                input_example=df[feature_columns].head(1), # check
                code_paths=["models.py"]
            )
            
            # Perform cross-validation and log metrics
            cv_metrics = multi_prophet.get_cross_validation_metrics(df, offset)
            for metric_name, metric_value in cv_metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)
                
        elif MODEL_TYPE in ["AUTOARIMA", "AUTOETS", "AUTOTHETA", "AUTOMFLES", "AUTOTBATS"]:
            DOWNSAMPLING = os.environ.get("DOWNSAMPLING", "0")
            sl_env = os.environ.get("SEASON_LENGTH", "0")
            sl: List[int] = [int(x) for x in sl_env.strip("[]").split(",")]

            if sl[0] == 0:
                SEASON_LENGTH: List[int] = [estimate_season_length(timedelta)]
            else:
                SEASON_LENGTH = sl

            # Collect StatsForecast parameters
            statsforecast_params = {
                "model_type": MODEL_TYPE,
                "output_sequence_length": OUTPUT_SEQ_LEN,
                "season_length": SEASON_LENGTH,
                "downsampling": DOWNSAMPLING,
                "output_sequence_length": OUTPUT_SEQ_LEN,
                "frequency": offset,
            }
            
            # Log all StatsForecast parameters
            mlflow.log_params({f"{k}": v for k, v in statsforecast_params.items()})
            
            # Create and fit multi-feature StatsForecast model
            multi_statsforecast = StatsForecastMultiFeatureModel()
            multi_statsforecast.fit(df, feature_columns, statsforecast_params, TIME_FEATURES)
            
            # Log the bundled model
            mlflow.pyfunc.log_model(
                name=MODEL_TYPE,
                python_model=multi_statsforecast,
                # signature=multi_statsforecast.get_signature(),
                code_paths=["models.py"]
            )
            
        else:
            raise ValueError(f"{MODEL_TYPE} not supported")

        producer = create_producer()
        topic = os.environ.get("PRODUCER_TOPIC")
        if not topic:
            raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

        try:
            message = {
                "operation": f"Trained: {MODEL_TYPE}",
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
                MODEL_TYPE
            )

        print("✅ Model logged to MLflow")

message_queue = queue.Queue()
worker_thread = threading.Thread(target=message_handler, daemon=True)
worker_thread.start()

#debug
from random import randint
CONSUMER_GROUP_ID = f"CONSUMER_GROUP_ID{randint(0, 999)}"
consumer = create_consumer(env_var("CONSUMER_TOPIC"), CONSUMER_GROUP_ID)

# consumer = create_consumer(env_var("CONSUMER_TOPIC"), env_var("CONSUMER_GROUP_ID"))
consume_messages(consumer, callback)

