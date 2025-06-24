import pandas as pd
import numpy as np
import torch
from shared.data_utils import to_Xy, generate_time_lags, remove_identifiers, to_timeseries_rep
from inference_container.predict import predict, load_model
from train_container.train import train_model

def test_predict_output_shape():
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    X_df, y_df = to_Xy(df, ["down", "up"])
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
        "batch_size": 8,
        "lr": 0.001
    }

    model = train_model(X_np, y_np, X_np, y_np, config)
    path = "temp_model.pt"
    torch.save(model.state_dict(), path)

    loaded_model = load_model(path, input_dim=X_np.shape[2], output_dim=y_np.shape[1], num_lags=10, device="cpu")
    y_pred = predict(loaded_model, X_np, device="cpu")

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_np.shape
