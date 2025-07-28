from client_utils import get_file, post_file
from data_utils import window_data, check_uniform, time_to_feature, subset_scaler
from kafka_utils import produce_message, publish_error
import torch
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import mlflow # type: ignore
import io
from typing import Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore

# Constants - These can remain here if they are only used by the service
TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
MODEL_NAME = "LSTM" # This could be configurable or extracted from MLflow run
OUTPUT_SEQ_LEN = 1 # This could be configurable or extracted from MLflow run
SAMPLE_IDX = 0
INFERENCE_LENGTH = 720

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inferencer:
    def __init__(self, gateway_url: str, producer, dlq_topic: str, output_topic: str):
        self.gateway_url = gateway_url
        self.producer = producer
        self.dlq_topic = dlq_topic
        self.output_topic = output_topic
        self.df = None
        self.current_model = None
        self.current_scaler: Union[MinMaxScaler, StandardScaler, None] = None
        self.current_experiment_name = "Default"
        self.current_run_name = "LSTM" # Default or configurable

    def load_model(self, experiment_name: str, run_name: str, sort: str="Recent"):
        print(f"Attempting to load model for experiment: {experiment_name}, run: {run_name}")

        try:
            if sort == "Recent":
                order = ["start_time desc"]
            elif sort == "Best":
                order = ["mse desc"] # not sure if this is correct
            else:
                raise TypeError("Invalid sort argument")
            
            runs_df = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.mlflow.runName = '{run_name}'", # Filter by run name
                order_by=order,
                max_results=1
            )

            if runs_df.empty:
                raise Exception(f"No runs found for experiment '{experiment_name}' with run name '{run_name}'.")

            run_id = runs_df.loc[0, 'run_id']
            print(f"Found run with ID: {run_id}")

            artifact_path = "model"
            model_uri = f"runs:/{run_id}/{artifact_path}"

            print(f"Loading model from: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            self.current_model = model
            self.current_experiment_name = experiment_name
            self.current_run_name = run_name
            print("✅ Model loaded successfully and updated service model.")

        except Exception as e:
            print(f"Error loading model: {e}")
            publish_error(
                self.producer,
                self.dlq_topic,
                "Model Load",
                "Failure",
                str(e),
                {"experiment": experiment_name, "run_name": run_name}
            )
            print(f"\nListing artifacts for run ID {run_id} to help find the correct path:")
            try:
                artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
                for artifact in artifacts:
                    print(f"- {artifact.path}")
            except Exception as list_e:
                print(f"Could not list artifacts: {list_e}")

    def perform_inference(self, df_eval: pd.DataFrame):
        if self.current_model is None:
            print("Model not loaded yet. Skipping inference.")

            publish_error(
                producer=self.producer,
                dlq_topic=self.dlq_topic,
                operation="Inference",
                status="Failure",
                error_details="Model not loaded",
                payload={"data_shape": df_eval.shape}
            )
            return

        FEATURES = df_eval.columns.difference(TIME_FEATURES, sort=False).tolist()

        X_eval, _ = window_data(df_eval, TIME_FEATURES)

        X_eval_tensor = torch.from_numpy(X_eval).float().to(device)

        timedelta = check_uniform(df_eval)

        remaining_real_data = X_eval.shape[0] - SAMPLE_IDX
        available_future_steps = min(remaining_real_data, INFERENCE_LENGTH)

        df_predictions = pd.DataFrame(
            index=pd.date_range(
                start=df_eval.index[SAMPLE_IDX],
                periods=INFERENCE_LENGTH,
                freq=timedelta
            ),
            columns=df_eval.columns
        )

        df_predictions = time_to_feature(df_predictions)

        current_sequence = X_eval_tensor[SAMPLE_IDX].unsqueeze(0).to(device)

        with torch.no_grad():
            step = 0
            while step < INFERENCE_LENGTH:
                multi_step_pred = self.current_model.predict(current_sequence.cpu().numpy())
                remaining_steps = INFERENCE_LENGTH - step
                steps_to_use = min(OUTPUT_SEQ_LEN, remaining_steps)

                for i in range(steps_to_use):
                    absolute_step = step + i
                    next_step = absolute_step + 1
                    if absolute_step >= INFERENCE_LENGTH:
                        break

                    current_pred = multi_step_pred[:, i, :].flatten()
                    df_predictions.loc[df_predictions.index[absolute_step], FEATURES] = current_pred

                    if next_step <= available_future_steps:
                        current_sequence = X_eval_tensor[SAMPLE_IDX + next_step].unsqueeze(0).to(device)
                    else:
                        extension_idx = next_step - available_future_steps

                        if extension_idx < df_predictions.shape[0]:
                            extension_row = df_predictions.iloc[[extension_idx]][TIME_FEATURES]
                            numpy_extension, _ = window_data(
                                extension_row,
                                TIME_FEATURES,
                                input_len=1, output_len=1
                            )
                            exog_tensor = torch.from_numpy(numpy_extension).float().to(device)
                            
                            pred_tensor = torch.tensor(current_pred).float().view(1, 1, -1).to(device)
                            
                            current_pred_for_seq = torch.cat((pred_tensor, exog_tensor.unsqueeze(0)), dim=-1)
                            current_sequence = torch.cat((current_sequence[:, 1:, :], current_pred_for_seq), dim=1)
                        else:
                            print(f"[Warning] df_predictions extension exhausted at index {extension_idx}. Stopping inference.")
                            break

                step += steps_to_use

        df_predictions = df_predictions.drop(columns=TIME_FEATURES)

        if self.current_scaler is not None:
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()

        print(f"Inference completed:")
        print(f"- Used actual future values for first {min(available_future_steps, INFERENCE_LENGTH)} steps")
        if INFERENCE_LENGTH > available_future_steps:
            print(f"- Switched to recursive mode after step {available_future_steps}")
        print(f"- Model predicts {OUTPUT_SEQ_LEN} step(s) at a time")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")

        output_table = pa.Table.from_pandas(df_transformed_predictions)
        parquet_buffer = io.BytesIO()
        pq.write_table(output_table, parquet_buffer)
        content_bytes = parquet_buffer.getvalue()

        try:
            object_key = f"{MODEL_NAME}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            post_file(self.gateway_url, "predictions", object_key, content_bytes)
            print("✅ Predictions pushed to MinIO")

            produce_message(self.producer, self.output_topic, {
                "operation": "Inference",
                "status": "SUCCESS",
                "bucket": "predictions",
                "object_key": object_key
            })
        except Exception as e:
            print(f"Error pushing predictions to MinIO: {e}")

            publish_error(
                self.producer,
                self.dlq_topic,
                "Push Predictions",
                "Failure",
                str(e),
                {"model_name": MODEL_NAME, "predictions_shape": df_transformed_predictions.shape}
            )