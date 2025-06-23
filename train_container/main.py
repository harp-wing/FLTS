# train_container/main.py

import pandas as pd
import numpy as np
from shared.data_utils import to_Xy, to_timeseries_rep
from train import train_model

def run_training():
    # Load data
    df = pd.read_csv("../data/ElBorn.csv", parse_dates=["time"])
    X_df, y_df = to_Xy(df, targets=["down", "up"])
    
    # Convert to model-ready format
    num_lags = 10
    num_features = X_df.shape[1] // num_lags
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

    # Train model
    model = train_model(X_np, y_np, X_np, y_np, config)

    # Save model
    import os
    os.makedirs("../outputs/models", exist_ok=True)
    torch.save(model.state_dict(), "../outputs/models/lstm.pt")
    print("✅ Model saved to outputs/models/lstm.pt")

if __name__ == "__main__":
    run_training()
