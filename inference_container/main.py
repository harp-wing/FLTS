# predict_container/main.py

import pandas as pd
import numpy as np
import os

from predict import predict, load_model
from shared.data_utils import to_Xy, generate_time_lags, remove_identifiers, to_timeseries_rep

def run_inference():
    df = pd.read_csv("../data/ElBorn.csv", parse_dates=["time"])

    num_lags = 10
    targets = ["down", "up"]

    X_df, y_df = to_Xy(df, targets=targets)
    X_df = generate_time_lags(X_df, num_lags)
    y_df = generate_time_lags(y_df, num_lags, is_y=True)
    X_df, y_df = remove_identifiers(X_df, y_df)

    num_features = X_df.shape[1] // num_lags
    X_np = to_timeseries_rep(X_df.to_numpy(), num_lags=num_lags, num_features=num_features)
    y_np = y_df.to_numpy()

    model = load_model("../outputs/models/lstm.pt", input_dim=num_features, output_dim=y_np.shape[1], num_lags=num_lags)

    y_pred = predict(model, X_np)

    os.makedirs("../outputs/predictions", exist_ok=True)
    np.save("../outputs/predictions/elborn_preds.npy", y_pred)
    print("✅ Predictions saved to outputs/predictions/elborn_preds.npy")

if __name__ == "__main__":
    run_inference()
