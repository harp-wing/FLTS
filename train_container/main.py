import numpy as np
import mlflow # type: ignore
import mlflow.pytorch # type: ignore
import datetime
import os
import threading
import queue
import pyarrow.parquet as pq
import torch
from typing import List
from models import LSTM, GRU, TETS, TCN
from train import prepare_data_loaders, train
from data_utils import window_data
from client_utils import get_file
from kafka_utils import create_consumer, consume_messages, create_producer, produce_message, publish_error

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
    Worker thread function that processes messages from the queue
    and triggers model training.
    """
    print("Train worker thread started. Waiting for messages in queue...")
    GATEWAY_URL = env_var("GATEWAY_URL")
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
                parquet_bytes = get_file(GATEWAY_URL, bucket, object_key)
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

def main(df, experiment_name: str="Default"):
    OUTPUT_SEQ_LEN: int = int(env_var("OUTPUT_SEQ_LEN"))
    INPUT_SEQ_LEN: int = int(env_var("INPUT_SEQ_LEN"))
    TRAIN_TEST_SPLIT: float = float(env_var("TRAIN_TEST_SPLIT"))
    BATCH_SIZE: int = int(env_var("BATCH_SIZE"))
    
    NUM_FEATURES = df.shape[1]
    TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
    TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
    N_EXO_FEATURES = len(TIME_FEATURES)

    print(df.columns.tolist())

    X, y = window_data(df, TIME_FEATURES, input_len=INPUT_SEQ_LEN, output_len=OUTPUT_SEQ_LEN)

    train_size = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE)

    MODEL_TYPE = env_var("MODEL_TYPE")
    EPOCHS: int = int(env_var("EPOCHS"))
    LEARNING_RATE: float = float(env_var("LEARNING_RATE"))
    EARLY_STOPPING: bool = bool(os.environ.get("EARLY_STOPPING", False))

    config = {
            "device": device,
            "input_size": NUM_FEATURES,
            "num_exgenous_features": N_EXO_FEATURES,
            "input_sequence_length": INPUT_SEQ_LEN,
            "output_sequence_length": OUTPUT_SEQ_LEN,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stopping": EARLY_STOPPING,
        }
    
    if EARLY_STOPPING:
        PATIENCE: int = int(env_var("PATIENCE"))
        config.update({"patience": PATIENCE})

    config.update({"model_type": MODEL_TYPE})
    
    if MODEL_TYPE == "LSTM":
        HIDDEN_SIZE: int = int(env_var("HIDDEN_SIZE"))
        NUM_LAYERS: int = int(env_var("NUM_LAYERS"))

        config.update({
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS
        })

        model = LSTM(
            input_size=NUM_FEATURES, 
            n_exo_features=N_EXO_FEATURES, 
            hidden_size=HIDDEN_SIZE, 
            output_size=OUTPUT_SEQ_LEN, 
            num_layers=NUM_LAYERS
        ).to(device)
    elif MODEL_TYPE == "GRU":
        HIDDEN_SIZE: int = int(env_var("HIDDEN_SIZE"))
        NUM_LAYERS: int = int(env_var("NUM_LAYERS"))

        config.update({
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS
        })

        model = GRU(
            input_size=NUM_FEATURES, 
            n_exo_features=N_EXO_FEATURES, 
            hidden_size=HIDDEN_SIZE, 
            output_size=OUTPUT_SEQ_LEN, 
            num_layers=NUM_LAYERS
        ).to(device)
    elif MODEL_TYPE == "TETS":
        MODEL_DIM: int = int(env_var("MODEL_DIM"))
        NUM_HEADS: int = int(env_var("NUM_HEADS"))
        NUM_LAYERS: int = int(env_var("NUM_LAYERS"))
        FEEDFORWARD_DIM: int = int(env_var("FEEDFORWARD_DIM"))
        DROPOUT: float = float(env_var("DROPOUT"))

        config.update({
            "model_dim": MODEL_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "feedforward_dim": FEEDFORWARD_DIM,
            "dropout": DROPOUT
        })

        model = TETS(
            input_size=NUM_FEATURES,
            n_exo_features=N_EXO_FEATURES,
            output_size=OUTPUT_SEQ_LEN,
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FEEDFORWARD_DIM,
            dropout=DROPOUT
        ).to(device)
    elif MODEL_TYPE == "TCN":
        # Parse LAYER_ARCHITECTURE from environment variable (e.g., "[32,64,128]")
        LAYER_ARCHITECTURE: List[int] = [int(x) for x in env_var("LAYER_ARCHITECTURE").strip("[]").split(",")]
        KERNEL_SIZE: int = int(env_var("KERNEL_SIZE"))
        DROPOUT: float = float(env_var("DROPOUT"))

        config.update({
            "layer_architecture": LAYER_ARCHITECTURE,
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT
        })

        model = TCN(
            input_size=NUM_FEATURES,
            output_size=OUTPUT_SEQ_LEN,
            n_exo_features=N_EXO_FEATURES,
            layer_architecture=LAYER_ARCHITECTURE,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT
        )
    else:
        raise ValueError(f"{MODEL_TYPE} not supported")

    # === MLflow Logging ===
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    with mlflow.start_run(run_name=MODEL_TYPE, log_system_metrics=True):
        mlflow.log_params(config)
        print(f"[DEBUG] X shape: {X_train.shape}, y shape: {y_train.shape}")
        print(f"[DEBUG] NaNs — y: {np.isnan(y_train).any() | np.isinf(y_train).any()}, X: {np.isnan(X_train).any() | np.isinf(X_train).any()}")

        model = train(model, train_loader, test_loader,
                          epochs=EPOCHS,
                          optimizer_type="adam", # "adam" | "sgd"
                          lr=LEARNING_RATE,
                          criterion="mse", # "mse" | "l1"
                          device=device,
                          early_stopping=True, patience=PATIENCE)

        producer = create_producer()
        topic = os.environ.get("PRODUCER_TOPIC")
        if not topic:
            raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

        try:
            mlflow.pytorch.log_model(
                model,
                name=MODEL_TYPE,
                input_example=X_train[:1],
                registered_model_name=None,
                code_paths=["models.py"]
            )
            print("✅ Model logged to MLflow")

            message = {
                "operation": f"Trained: {MODEL_TYPE}",
                "status": "SUCCESS",
                "experiment": experiment_name,
                "run_name": MODEL_TYPE
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


message_queue = queue.Queue()

worker_thread = threading.Thread(target=message_handler, daemon=True)
worker_thread.start()

consumer = create_consumer(os.environ.get("CONSUMER_TOPIC"), os.environ.get("CONSUMER_GROUP_ID"))
consume_messages(consumer, callback)
