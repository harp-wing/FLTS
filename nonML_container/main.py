# nonML container
import os
import queue
import threading
import datetime
import mflow # type: ignore
import mlflow.statsforecast # type: ignore
import mlflow.prophet # type: ignore
import pandas as pd
import pyarrow.parquet as pq
from typing import List
from client_utils import get_file
from kafka_utils import create_producer, create_consumer, produce_message, consume_messages, publish_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsforecast import StatsForecast
from statsforecast.models import (                                                                                              # type: ignore
    AutoARIMA, # AutoRegressive Integrated Moving Average
    AutoETS, # Exponential Smoothing
    AutoTBATS # Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend components, Seasonal components
    # Consider also AutoMFLES
)

def env_var(var: str) -> str:
    temp = os.environ.get(var)
    if temp is None:
        raise TypeError(f"Environment variable, {var}, not defined")
    else:
        return temp

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
        else:
            print(f"NonML worker WARN: Message received without complete claim check details or unknown operation: {claim_check}")

        # Mark the task as done after processing
        message_queue.task_done()

def main(df: pd.DataFrame, experiment_name: str="NonML"):
    OUTPUT_SEQ_LEN: int = int(env_var("OUTPUT_SEQ_LEN"))
    MODEL_TYPE: str = env_var("MODEL_TYPE")
    
    config = {
        "horizon": OUTPUT_SEQ_LEN,
        "model_type": MODEL_TYPE,
    }
    
    # === MLflow Logging ===
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"{MODEL_TYPE}_{timestamp}"

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_params(config)

        # Prepare data for statsforecast and prophet
        # Statsforecast expects columns 'ds' (datestamp), 'y' (value), and 'unique_id'
        # Prophet expects columns 'ds' and 'y'
        
        df.rename(columns={'y_col_name': 'y'}, inplace=True)
        df['unique_id'] = "1"  # Statsforecast requires a unique_id for each time series
        df.index.rename('ds', inplace=True)
        df = df.reset_index()

        if MODEL_TYPE == "PROPHET":
            # Prophet requires specific column names 'ds' and 'y'
            # It also handles seasonality, holidays, and trend automatically
            prophet_df = df.rename(columns={"ds": "ds", "y": "y"})
            
            # Additional Prophet parameters from environment variables
            SEASONALITY_MODE = os.environ.get("SEASONALITY_MODE", "additive")
            config.update({"seasonality_mode": SEASONALITY_MODE})
            
            model = Prophet(seasonality_mode=SEASONALITY_MODE)
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=OUTPUT_SEQ_LEN)
            forecast = model.predict(future)
            
            # Log model and results to MLflow
            mlflow.prophet.log_model(model, "prophet_model")
            
            # Cross-validation and performance metrics can be logged as well
            cv_results = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
            df_p = performance_metrics(cv_results)
            mlflow.log_metric("prophet_mae", df_p['mae'].mean())
            
            mlflow.log_param("model_name", "Prophet")
            
        elif MODEL_TYPE in ["AUTOARIMA", "AUTOETS", "AUTOTBATS"]:
            models_dict = {
                "AUTOARIMA": [AutoARIMA(season_length=720)], # season_length is an example, should be dynamic
                "AUTOETS": [AutoETS(season_length=720)],
                "AUTOTBATS": [AutoTBATS(season_length=[720, 5040, 262980])],
            }
            
            sf = StatsForecast(
                models=models_dict[MODEL_TYPE],
                freq='D' # Frequency is an example, should be dynamic
            )
            
            # Fit the model
            sf.fit(df)
            
            # Make the forecast
            forecast_df = sf.predict(h=OUTPUT_SEQ_LEN)

            # Log model and results to MLflow
            mlflow.statsforecast.log_model(sf, "statsforecast_model")
            mlflow.log_param("model_name", MODEL_TYPE)
            
        else:
            raise ValueError(f"{MODEL_TYPE} not supported")

        producer = create_producer()
        topic = os.environ.get("PRODUCER_TOPIC")
        if not topic:
            raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

        try:
            # The MLflow logging for StatsForecast is done in the `if` block,
            # this section handles the message to Kafka.
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

consumer = create_consumer(env_var("CONSUMER_TOPIC"), env_var("CONSUMER_GROUP_ID"))
consume_messages(consumer, callback)

# -------------------- Statsforecast Models -------------------

