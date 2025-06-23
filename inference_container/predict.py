# predict_container/predict.py

import torch
import numpy as np
import pandas as pd

from train_container.m1 import LSTM
from shared.data_utils import (
    read_data, to_Xy, generate_time_lags,
    time_to_feature, assign_statistics, remove_identifiers,
    get_exogenous_data_by_area, to_timeseries_rep
)

def load_model(path, input_dim, output_dim, num_lags, device="cpu"):
    model = LSTM(input_dim=input_dim, num_outputs=output_dim, matrix_rep=True)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, X, exogenous_data=None, device="cpu"):
    X_tensor = torch.tensor(X).float().to(device)
    y_hist = torch.zeros((X_tensor.shape[0], X_tensor.shape[2]))  # dummy
    exo_tensor = torch.tensor(exogenous_data).float().to(device) if exogenous_data is not None else None

    with torch.no_grad():
        y_pred = model(X_tensor, exo_tensor, device=device, y_hist=y_hist)

    return y_pred.cpu().numpy()
