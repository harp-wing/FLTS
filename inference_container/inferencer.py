from client_utils import post_file
from data_utils import window_data, check_uniform, time_to_feature, subset_scaler
from kafka_utils import produce_message, publish_error
import torch
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import mlflow # type: ignore
import io
import os
from typing import Tuple
from druid_utils import DruidIngester
from typing import Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore

# Constants - These should all be defined by the service later
TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
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
        self.output_seq_len = 0
        self.current_model = None
        self.current_scaler: Union[MinMaxScaler, StandardScaler, None] = None
        self.current_experiment_name = "Default"
        self.current_run_name = ""
        self.model_type = ""
        self.model_class = "pytorch"  # "pytorch", "prophet", "statsforecast"

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
                max_results=1,
                output_format="pandas"
            )

            if runs_df.empty:
                raise Exception(f"No runs found for experiment '{experiment_name}' with run name '{run_name}'.")

            run_id = runs_df.loc[0, "run_id"]
            print(runs_df.columns)
            self.output_seq_len = int(runs_df.loc[0, "params.output_sequence_length"])
            
            # Detect model type from experiment name or parameters
            self.model_type, self.model_class = self._detect_model_type(runs_df.loc[0])
            
            print(f"Found run with ID: {run_id}, Model type: {self.model_type}, Model class: {self.model_class}")

            model_uri = f"runs:/{run_id}/{run_name}"

            print(f"Loading model from: {model_uri}")
            mlflow.pyfunc.get_model_dependencies(model_uri)
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

    def _detect_model_type(self, run_row: pd.Series) -> Tuple[str, str]:
        """Detect [model_type, model_class] from MLflow run parameters or tags."""

        # Check for explicit model type parameter
        if "params.model_type" in run_row and pd.notna(run_row["params.model_type"]):
            model_type = run_row["params.model_type"].upper()
            if model_type in ["LSTM", "GRU", "TETS", "TCN"]:
                return model_type, "pytorch"
            elif model_type in ["AUTOARIMA", "AUTOETS", "AUTOTHETA", "AUTOMFLES", "AUTOTBATS"]:
                return model_type, "statsforecast"
            elif model_type == "PROPHET":
                return "PROPHET", "prophet"

        # Check experiment name
        exp_name = self.current_experiment_name.lower()
        if "prophet" in exp_name:
            return "PROPHET", "prophet"

        for sf_model in ["autoarima", "autoets", "autotheta", "automfles", "autotbats"]:
            if sf_model in exp_name:
                return sf_model.upper(), "statsforecast"

        for pt_model in ["lstm", "gru", "tets", "tcn"]:
            if pt_model in exp_name:
                return pt_model.upper(), "pytorch"

        # Check params to infer framework
        if any(param.startswith("params.seasonality") 
            for param in run_row.index if pd.notna(run_row.get(param))):
            return "PROPHET", "prophet"

        if any(param.startswith("params.season_length") 
            for param in run_row.index if pd.notna(run_row.get(param))):
            return "", "statsforecast"  # fallback default for statsforecast

        # Default fallback
        return "", "pytorch"

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
        
        # Prepare evaluation data
        timedelta = check_uniform(df_eval)

        df_predictions = pd.DataFrame(
            index=pd.date_range(
                start=df_eval.index[SAMPLE_IDX],
                periods=INFERENCE_LENGTH,
                freq=timedelta
            ),
            columns=df_eval.columns
        )

        df_predictions = time_to_feature(df_predictions)

        # Route to appropriate inference method based on model type
        if self.model_class == "pytorch":
            df_transformed_predictions = self._perform_pytorch_inference(df_eval, df_predictions)
        elif self.model_class == "prophet":
            df_transformed_predictions = self._perform_prophet_inference(df_eval, df_predictions)
        elif self.model_class == "statsforecast":
            df_transformed_predictions = self._perform_statsforecast_inference(df_eval, df_predictions)
        else:
            raise ValueError(f"Unsupported model class: {self.model_class}")

        # Common post-processing for all model types
        self._save_and_publish_predictions(df_transformed_predictions)

    def _perform_pytorch_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """PyTorch inference logic"""
        FEATURES = df_eval.columns.difference(TIME_FEATURES, sort=False).tolist()

        X_eval, _ = window_data(df_eval, TIME_FEATURES)
        X_eval_tensor = torch.from_numpy(X_eval).float().to(device)

        remaining_real_data = X_eval.shape[0] - SAMPLE_IDX
        available_future_steps = min(remaining_real_data, INFERENCE_LENGTH)

        current_sequence = X_eval_tensor[SAMPLE_IDX].unsqueeze(0).to(device)

        with torch.no_grad():
            step = 0
            while step < INFERENCE_LENGTH:
                multi_step_pred = self.current_model.predict(current_sequence.cpu().numpy()) # type: ignore
                remaining_steps = INFERENCE_LENGTH - step
                steps_to_use = min(self.output_seq_len, remaining_steps)

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

        print(f"PyTorch Inference completed:")
        print(f"- Used actual future values for first {min(available_future_steps, INFERENCE_LENGTH)} steps")
        if INFERENCE_LENGTH > available_future_steps:
            print(f"- Switched to recursive mode after step {available_future_steps}")
        print(f"- Model predicts {self.output_seq_len} step(s) at a time")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")

        return df_transformed_predictions

    def _perform_prophet_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """Prophet inference logic"""
        # Prepare future dataframe for Prophet
        timedelta = check_uniform(df_eval)
        
        future_df = pd.DataFrame(
            index=pd.date_range(
                start=df_eval.index[-1] + timedelta,  # Start from next time point
                periods=INFERENCE_LENGTH,
                freq=timedelta
            )
        )
        future_df['ds'] = future_df.index
        
        # Get predictions from Prophet model
        predictions = self.current_model.predict(future_df) # type: ignore
        
        # Extract forecast columns (those ending with '_yhat')
        forecast_columns = [col for col in predictions.columns if col.endswith('_yhat')]
        
        # Create output dataframe with original feature names
        feature_names = [col.replace('_yhat', '') for col in forecast_columns]
        df_predictions = pd.DataFrame(
            predictions[forecast_columns].values,
            index=future_df.index,
            columns=feature_names
        )
        
        # Apply inverse scaling if scaler is available
        if self.current_scaler is not None:
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()

        print(f"Prophet Inference completed:")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")
        print(f"- Features forecasted: {list(df_transformed_predictions.columns)}")

        return df_transformed_predictions

    def _perform_statsforecast_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """StatsForecast inference logic"""
        # Get predictions from StatsForecast model
        exog_df = df_predictions[TIME_FEATURES] if TIME_FEATURES else None

        input_dict = {
            "h": INFERENCE_LENGTH,
            "X": exog_df,
            "level": None
        }

        df_predictions = self.current_model.predict(input_dict) # type: ignore
        
        # og_features = ["down", "up", "rnti_count", "mcs_down", "mcs_down_var", "mcs_up", "mcs_up_var", "rb_down", "rb_down_var", "rb_up", "rb_up_var"]
        # self.current_scaler = subset_scaler(self.current_scaler, og_features, df_predictions.columns.to_list())
        
        # Apply inverse scaling if scaler is available
        if self.current_scaler is not None:
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()

        print(f"StatsForecast Inference completed:")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")
        print(f"- Features forecasted: {list(df_transformed_predictions.columns)}")

        return df_transformed_predictions

    def _save_and_publish_predictions(self, df_transformed_predictions: pd.DataFrame):
        """Common logic for saving and publishing predictions"""
        # Ingest to Druid
        druid = DruidIngester()
        druid_df = df_transformed_predictions.reset_index(names="time")
        task_id = druid.ingest_dataframe(druid_df, f"{self.current_run_name}", "time")
        if task_id:
            print(f"Data ingested successfully. Task ID: {task_id}")
        else:
            print("Failed to ingest data")

        # Save to MinIO
        # output_table = pa.Table.from_pandas(df_transformed_predictions)
        # parquet_buffer = io.BytesIO()
        # pq.write_table(output_table, parquet_buffer)
        # content_bytes = parquet_buffer.getvalue()

        # try:
        #     object_key = f"{self.current_run_name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.parquet"
        #     post_file(self.gateway_url, "predictions", object_key, content_bytes)
        #     print("✅ Predictions pushed to MinIO")

        #     produce_message(self.producer, self.output_topic, {
        #         "operation": "Inference",
        #         "status": "SUCCESS",
        #         "bucket": "predictions",
        #         "object_key": object_key
        #     })
        # except Exception as e:
        #     print(f"Error pushing predictions to MinIO: {e}")

        #     publish_error(
        #         self.producer,
        #         self.dlq_topic,
        #         "Push Predictions",
        #         "Failure",
        #         str(e),
        #         {"model_name": self.current_run_name, "predictions_shape": df_transformed_predictions.shape}
        #     )