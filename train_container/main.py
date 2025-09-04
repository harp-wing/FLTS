import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import pickle
import tempfile
import os
import threading
import queue
import pyarrow.parquet as pq
import torch
from typing import List
from ml_models import LSTM, GRU, TETS, TCN, EncoderLSTM
from train import prepare_data_loaders, train
from data_utils import window_data, subset_scaler
from client_utils import get_file
from kafka_utils import create_consumer, consume_messages, create_producer, produce_message, publish_error

def env_var(var: str) -> str:
    temp = os.environ.get(var)
    if not temp:
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
                if parquet_bytes:
                    # Read the schema to access the file's metadata
                    schema = pq.read_schema(parquet_bytes)
                    custom_metadata = schema.metadata

                    # Retrieve and deserialize the scaler object
                    serialized_scaler = custom_metadata.get(b'scaler_object')
                    
                    if serialized_scaler:
                        scaler = pickle.loads(serialized_scaler)
                        print(f"Scaler object, {scaler}, retrieved successfully.")
                    else:
                        print("'scaler_object' not found in the file's metadata.")
                if TRIMS:
                    scaler = subset_scaler(scaler, df.columns.to_list(), TRIMS)
                    df.drop(columns=df.columns.difference(TRIMS + TIME_FEATURES), inplace=True)

                # Save scaler to a temporary file because MLflow can only save artifacts from files
                scaler_path = os.path.join(tempfile.gettempdir(), "scaler.pkl")
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
            except Exception as e:
                print(f"Train worker error fetching or parsing data for {object_key}: {e}")
            

            print(f"Train worker starting model training for data from {object_key}...")
            try:

                main(df, scaler_path) # TRAINING LOGIC CALL

                print(f"Train worker finished model training for data from {object_key}.")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Train worker error during model training for {object_key}: {e}")
        else:
            print(f"Train worker WARN: Message received without complete claim check details or unknown operation: {claim_check}")

        # Mark the task as done after processing
        message_queue.task_done()

def main(df: pd.DataFrame, scaler_path, experiment_name: str="Default"):
    OUTPUT_SEQ_LEN: int = int(env_var("OUTPUT_SEQ_LEN"))
    INPUT_SEQ_LEN: int = int(env_var("INPUT_SEQ_LEN"))
    TRAIN_TEST_SPLIT: float = float(env_var("TRAIN_TEST_SPLIT"))
    BATCH_SIZE: int = int(env_var("BATCH_SIZE"))
    
    NUM_FEATURES = df.shape[1]
    TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
    TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
    N_EXO_FEATURES = len(TIME_FEATURES)

    MODEL_TYPE = env_var("MODEL_TYPE")
    EPOCHS: int = int(env_var("EPOCHS"))
    LEARNING_RATE: float = float(env_var("LEARNING_RATE"))
    EARLY_STOPPING: bool = bool(os.environ.get("EARLY_STOPPING", False))

    feature_list = df.columns.tolist()
    print(feature_list)

    SCHEDULED_SAMPLING_COMPATIBLE_MODELS = ["EncoderLSTM"]
    if MODEL_TYPE in SCHEDULED_SAMPLING_COMPATIBLE_MODELS:
        X, y = window_data(df, input_len=INPUT_SEQ_LEN, output_len=OUTPUT_SEQ_LEN)
    else:
        X, y = window_data(df, TIME_FEATURES, input_len=INPUT_SEQ_LEN, output_len=OUTPUT_SEQ_LEN)

    train_size = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE)

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

    optimizer = "adam"
    schedule = False
    ss = False
    
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

        optimizer = "adamw"
        schedule = True

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
    elif MODEL_TYPE == "EncoderLSTM":
        HIDDEN_SIZE: int = int(env_var("HIDDEN_SIZE"))
        NUM_LAYERS: int = int(env_var("NUM_LAYERS"))

        ss = True

        config.update({
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS
        })

        model = EncoderLSTM(
            input_size=NUM_FEATURES, 
            n_exo_features=N_EXO_FEATURES, 
            hidden_size=HIDDEN_SIZE, 
            output_seq_len=OUTPUT_SEQ_LEN, 
            num_layers=NUM_LAYERS
        ).to(device)
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

        # Log scaler as an artifact
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        model = train(model, train_loader, test_loader,
                          epochs=EPOCHS,
                          optimizer_type=optimizer, # "adam" | "adamw" | "sgd"
                          scheduled_learning=schedule,
                          scheduled_sampling=ss,
                          lr=LEARNING_RATE,
                          criterion="mse", # "mse" | "l1"
                          max_grad_norm=1.0,
                          device=device,
                          early_stopping=True, patience=PATIENCE)
        
        try:
            from captum.attr import IntegratedGradients, GradientShap, DeepLift
            from captum.attr import visualization as viz
            import matplotlib.pyplot as plt
            
            X_test_tensor = torch.from_numpy(X_test).float()
            
            # Get predictions for targets
            model.eval()
            X_subset = X_test_tensor[:min(50, len(X_test_tensor))]
            baseline_dist = torch.randn(10, *X_subset.shape[1:]) * 0.1

            batch_size, seq_len, num_features = X_subset.shape
            
            # Get model output shape
            with torch.no_grad():
                sample_output = model(X_subset[:1])
                print(f"Model output shape: {sample_output.shape}")
            
            _, _, num_targets = sample_output.shape

            print(f"Processing {num_features} input features and {num_targets} output targets...")

            # Method 1: Feature-wise attribution (your original approach)
            feature_attributions = []

            # Loop over input features
            for f in range(num_features):
                print(f"Processing input feature {f+1}/{num_features}")
                
                input_masked = X_subset.clone()
                
                # Optional: zero out other features if you want strict per-feature attribution
                # input_masked[:, :, [i for i in range(num_features) if i != f]] = 0

                target_importances = []

                # Loop over output targets
                for t in range(num_targets):
                    try:
                        # Define a wrapper that returns scalar output per sample
                        def forward_for_target(x):
                            return model(x)[:, -1, t]  # pick last timestep and target t
                        
                        gs = GradientShap(forward_for_target)

                        # Compute attributions for this target
                        attributions = gs.attribute(
                            input_masked, 
                            baselines=baseline_dist, 
                            target=None, 
                            n_samples=50, 
                            stdevs=0.0
                        )
                        
                        # attributions shape: (batch_size, seq_len, num_features)
                        attributions_np = attributions.detach().cpu().numpy()
                        
                        # Focus on the current feature's attributions
                        feature_attr = attributions_np[:, :, f]  # Shape: (batch_size, seq_len)
                        
                        # Aggregate across batch and sequence
                        importance = np.abs(feature_attr).mean()
                        target_importances.append(importance)
                        
                    except Exception as e:
                        print(f"Warning: Failed for feature {f}, target {t}: {e}")
                        target_importances.append(0.0)

                # Average importance across output targets
                avg_importance = np.mean(target_importances) if target_importances else 0.0
                feature_attributions.append(avg_importance)

            feature_attributions = np.array(feature_attributions)

            # Plot feature-wise attributions
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(feature_attributions)), feature_attributions)
            plt.xlabel("Input Feature Index")
            plt.ylabel("Attribution Magnitude")
            plt.title("Input Feature Attributions Averaged Across All Outputs (GradientShap)")
            plt.xticks(range(len(feature_attributions)))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "xai/feature_attributions.png")
            plt.close()

            # Method 2: Target-wise attribution analysis
            print("Computing target-wise attributions...")
            
            target_attributions = []
            
            for t in range(min(num_targets, 10)):  # Limit to first 10 targets to avoid too many plots
                print(f"Processing output target {t+1}/{min(num_targets, 10)}")
                
                def forward_for_target_t(x):
                    return model(x)[:, -1, t]
                
                try:
                    gs = GradientShap(forward_for_target_t)
                    attributions = gs.attribute(
                        X_subset, 
                        baselines=baseline_dist, 
                        target=None, 
                        n_samples=50, 
                        stdevs=0.0
                    )
                    
                    attributions_np = attributions.detach().cpu().numpy()
                    
                    # Average across samples and time steps for each input feature
                    feature_importance = np.abs(attributions_np).mean(axis=(0, 1))  # Shape: (num_features,)
                    target_attributions.append(feature_importance)
                    
                except Exception as e:
                    print(f"Warning: Failed for target {t}: {e}")
                    target_attributions.append(np.zeros(num_features))

            if target_attributions:
                target_attributions = np.array(target_attributions)  # Shape: (num_targets, num_features)
                
                # Create heatmap showing which input features are important for each output
                plt.figure(figsize=(14, 8))
                im = plt.imshow(target_attributions, cmap='YlOrRd', aspect='auto')
                plt.colorbar(im, label='Attribution Magnitude')
                plt.xlabel('Input Features')
                plt.ylabel('Output Targets')
                plt.title('Attribution Heatmap: Input Features vs Output Targets')
                
                # Add text annotations for small matrices
                if target_attributions.shape[0] <= 10 and target_attributions.shape[1] <= 20:
                    for i in range(target_attributions.shape[0]):
                        for j in range(target_attributions.shape[1]):
                            plt.text(j, i, f'{target_attributions[i, j]:.3f}', 
                                    ha='center', va='center', fontsize=8)
                
                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "xai/target_attribution_heatmap.png")
                plt.close()
                
                # Plot top contributing input features for each target
                n_top_features = min(5, num_features)
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                for t in range(min(len(target_attributions), 6)):
                    ax = axes[t]
                    
                    # Get top features for this target
                    top_indices = np.argsort(target_attributions[t])[-n_top_features:]
                    top_values = target_attributions[t][top_indices]
                    
                    ax.barh(range(len(top_values)), top_values)
                    ax.set_yticks(range(len(top_values)))
                    ax.set_yticklabels([f'Feature_{i}' for i in top_indices])
                    ax.set_xlabel('Attribution')
                    ax.set_title(f'Top Features for Output Target {t}')
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for t in range(len(target_attributions), len(axes)):
                    axes[t].set_visible(False)
                
                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "xai/top_features_per_target.png")
                plt.close()

            # Method 3: Temporal attribution analysis
            print("Computing temporal attributions...")
            
            try:
                # Analyze how importance changes across time steps
                def forward_sum_all_targets(x):
                    outputs = model(x)  # Shape: (batch, seq_len, num_targets)
                    return outputs[:, -1, :].sum(dim=1)  # Sum all targets at last timestep
                
                gs_temporal = GradientShap(forward_sum_all_targets)
                temporal_attributions = gs_temporal.attribute(
                    X_subset, 
                    baselines=baseline_dist, 
                    target=None, 
                    n_samples=50, 
                    stdevs=0.0
                )
                
                temporal_attr_np = temporal_attributions.detach().cpu().numpy()
                
                # Average across samples and features to see temporal patterns
                temporal_importance = np.abs(temporal_attr_np).mean(axis=(0, 2))  # Shape: (seq_len,)
                
                plt.figure(figsize=(10, 6))
                plt.plot(range(seq_len), temporal_importance, marker='o', linewidth=2, markersize=6)
                plt.xlabel('Time Step')
                plt.ylabel('Average Attribution Magnitude')
                plt.title('Temporal Attribution Pattern (Importance Across Sequence)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "xai/temporal_attribution.png")
                plt.close()
                
            except Exception as e:
                print(f"Warning: Temporal analysis failed: {e}")

            # Summary statistics
            print("\n=== Attribution Summary ===")
            print(f"Top 3 most important input features:")
            top_3_features = np.argsort(feature_attributions)[-3:][::-1]
            for i, feat_idx in enumerate(top_3_features):
                print(f"  {i+1}. Feature {feat_idx}: {feature_attributions[feat_idx]:.4f}")
            
            if target_attributions:
                print(f"\nMost variable output target (has diverse input dependencies):")
                target_variance = np.var(target_attributions, axis=1)
                most_variable_target = np.argmax(target_variance)
                print(f"  Target {most_variable_target}: variance = {target_variance[most_variable_target]:.4f}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] Could not generate Captum artifacts: {e}")

        producer = create_producer()
        topic = os.environ.get("PRODUCER_TOPIC")
        if not topic:
            raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

        try:
            mlflow.pytorch.log_model(       # type: ignore
                model,
                name=MODEL_TYPE,
                input_example=X_train[:1],
                code_paths=["ml_models.py"],
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



TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

trims = os.environ.get("TRIMS", "[]").strip("[]").split(",")
TRIMS = [item.strip() for item in trims if item.strip()]

message_queue = queue.Queue()

worker_thread = threading.Thread(target=message_handler, daemon=True)
worker_thread.start()

# debug
from random import randint
CONSUMER_GROUP_ID = f"CONSUMER_GROUP_ID{randint(0, 999)}"
consumer = create_consumer(os.environ.get("CONSUMER_TOPIC"), CONSUMER_GROUP_ID)

# consumer = create_consumer(os.environ.get("CONSUMER_TOPIC"), os.environ.get("CONSUMER_GROUP_ID"))
consume_messages(consumer, callback)
