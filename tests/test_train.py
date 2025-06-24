import pandas as pd
import torch
import numpy as np
from shared.data_utils import to_Xy, generate_time_lags, remove_identifiers, to_timeseries_rep
from train_container.train import train_model

def test_train_model_and_save(tmp_path):
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    X_df, y_df = to_Xy(df, targets=["down", "up"])
    X_df = generate_time_lags(X_df, 10)
    y_df = generate_time_lags(y_df, 10, is_y=True)
    X_df, y_df = remove_identifiers(X_df, y_df)

    X_np = to_timeseries_rep(X_df.to_numpy(), 10, X_df.shape[1] // 10)
    y_np = y_df.to_numpy()

    config = {
        "device": "cpu",
        "num_lags": 10,
        "num_features": X_np.shape[2],
        "output_dim": y_np.shape[1],
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.001
    }

    model = train_model(X_np, y_np, X_np, y_np, config)
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    assert model_path.exists()
    assert isinstance(model, torch.nn.Module)
