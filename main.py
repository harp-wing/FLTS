# basic_pipeline/main.py

import pandas as pd
import numpy as np
import os
import torch

from eda_container.explore import describe_data, plot_feature_trends, correlation_matrix
from train_container.train import train_model
from train_container.m1 import LSTM
from inference_container.predict import predict, load_model
from eval_container.metrics import compute_metrics, save_metrics, plot_predictions
from shared.data_utils import to_Xy, generate_time_lags, remove_identifiers, to_timeseries_rep

def main():
    os.makedirs("outputs/eda", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    # === Step 1: Load Data ===
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])

    # === Step 2: EDA ===
    describe_data(df)
    plot_feature_trends(df, output_dir="outputs/eda")
    correlation_matrix(df, output_path="outputs/eda/correlation_matrix.png")

    # === Step 3: Preprocessing for Training & Inference ===
    num_lags = 10
    targets = ["down", "up"]
    X_df, y_df = to_Xy(df, targets=targets)
    X_df = generate_time_lags(X_df, num_lags)
    y_df = generate_time_lags(y_df, num_lags, is_y=True)
    X_df, y_df = remove_identifiers(X_df, y_df)

    num_features = X_df.shape[1] // num_lags
    X_np = to_timeseries_rep(X_df.to_numpy(), num_lags=num_lags, num_features=num_features)
    y_np = y_df.to_numpy()

    # === Step 4: Train Model ===
    config = {
        "device": "cpu",
        "num_lags": num_lags,
        "num_features": num_features,
        "output_dim": y_np.shape[1],
        "epochs": 5,
        "batch_size": 32,
        "lr": 0.001
    }
    model = train_model(X_np, y_np, X_np, y_np, config)
    torch.save(model.state_dict(), "outputs/models/lstm.pt")
    print("✅ Model saved")

    # === Step 5: Predict ===
    model = load_model("outputs/models/lstm.pt", num_features, y_np.shape[1], num_lags, device="cpu")
    y_pred = predict(model, X_np)
    np.save("outputs/predictions/elborn_preds.npy", y_pred)
    np.save("outputs/predictions/elborn_ground_truth.npy", y_np)
    print("✅ Predictions saved")

    # === Step 6: Evaluate ===
    metrics = compute_metrics(y_np, y_pred)
    save_metrics(metrics, path="outputs/metrics/metrics.json")
    plot_predictions(y_np, y_pred, path="outputs/metrics/prediction_plot.png")

    print("✅ Evaluation complete")

if __name__ == "__main__":
    main()
