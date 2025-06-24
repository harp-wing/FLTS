# train_container/main.py

import os
import pandas as pd
import numpy as np
from shared.data_utils import (
    generate_time_lags,
    to_Xy,
    to_timeseries_rep
)
from train import train_model
import torch

def run_training():
    # Load raw data
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    
    print("Raw DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample rows:\n", df.head())
    
    # # Add time lags BEFORE splitting X/y
    # X_df, y_df = to_Xy(df, targets=["down", "up"], identifier=None)
    # X_df = generate_time_lags(X_df, n_lags=10, identifier=None, is_y=False)
    # y_df = generate_time_lags(y_df, n_lags=10, identifier=None, is_y=True)

    # print("Time-lagged DF shape:", df.shape)
    # print("Sample rows:\n", df.head())
    # print("Time lag generation complete")


    # # Split into features and targets
    # X_df, y_df = to_Xy(df, targets=["down", "up"], identifier=None)

    # # Parameters
    # num_lags = 10
    # total_features = X_df.shape[1]
    # num_features = total_features // num_lags

    # # Debug check
    # if total_features % num_lags != 0:
    #     raise ValueError(f"Feature count {total_features} not divisible by num_lags={num_lags}!")
####################    
    # Step 1: Split into features and targets
    X_df, y_df = to_Xy(df, targets=["down", "up"], identifier=None)

    # Step 2: Add time lags
    num_lags = 10
    X_df = generate_time_lags(X_df, n_lags=10, identifier=None, is_y=False)
    y_df = generate_time_lags(y_df, n_lags=10, identifier=None, is_y=True)

    # Step 3: Print debug shapes
    print("Lagged X shape:", X_df.shape)
    print("Lagged y shape:", y_df.shape)

    # Step 4: Safety check
    total_features = X_df.shape[1]
    if total_features % num_lags != 0:
        raise ValueError(f"Feature count {total_features} not divisible by num_lags={num_lags}!")
    num_features = total_features // num_lags

##################

    # Convert to model-ready tensors
    X_np = to_timeseries_rep(X_df.to_numpy(), num_lags=num_lags, num_features=num_features)
    y_np = y_df.to_numpy()

    # Config
    config = {
        "device": "cpu",
        "num_lags": num_lags,
        "num_features": num_features,
        "output_dim": y_np.shape[1],
        "epochs": 5,
        "batch_size": 32,
        "lr": 0.001
    }

    # Train
    model = train_model(X_np, y_np, X_np, y_np, config)

    # Save
    os.makedirs("../outputs/models", exist_ok=True)
    torch.save(model.state_dict(), "../outputs/models/lstm.pt")
    print("Model saved to outputs/models/lstm.pt")

if __name__ == "__main__":
    run_training()
