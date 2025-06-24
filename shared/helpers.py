"""
Helper functions during training.
"""

import copy
import math
from logging import INFO
from typing import Union, List, Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from shared.logger import log


def get_optim(model,
              optim_name: str = "adam",
              lr: float = 1e-3):
    """Returns the specified optimizer for the model defined as torch module."""
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"The specified optimizer: {optim_name} is not currently supported.")
    return optimizer


def get_criterion(crit_name: str = "mse"):
    """Returns the specified loss function."""
    if crit_name == "mse":
        criterion = torch.nn.MSELoss()
    elif crit_name == "l1":
        criterion = torch.nn.L1Loss()
    else:
        raise NotImplementedError(f"The specified criterion: {crit_name} is not currently supported.")
    return criterion


def log_metrics(y_true: np.ndarray,
                y_pred: np.ndarray) -> Union[None, Dict[int, List[float]]]:
    """Regression metrics per output dimensions."""
    try:
        shape = y_true.shape[1]
    except IndexError:
        return None

    assert y_true.shape == y_pred.shape

    res = dict()
    for dim in range(shape):
        y_true_dim = y_true[:, dim]
        y_pred_dim = y_pred[:, dim]
        mse = mean_squared_error(y_true_dim, y_pred_dim)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true_dim, y_pred_dim)
        r2 = r2_score(y_true_dim, y_pred_dim)
        nrmse = rmse / np.mean(y_true_dim)
        log(INFO, f"Metrics for dimension: {dim}\n"
                  f"\tmse: {mse}, rmse: {rmse}, mae: {mae}, r^2: {r2}, nrmse: {nrmse}")

        res[dim] = [rmse, mae, nrmse]

    return res


def accumulate_metric(y_true: Union[np.ndarray, torch.Tensor],
                      y_pred: Union[np.ndarray, torch.Tensor],
                      log_per_output: bool = False,
                      dims: Optional[List[int]] = None,
                      return_all=False) -> Union[
    Tuple[float, float, float, float, float],
    Tuple[float, float, float, float, float, Union[None, Dict[int, List[float]]]]]:
    """Computes standard regression metrics (MSE, RMSE, MAE, R2, NRMSE).
    Automatically computes NRMSE for all output dimensions if dims is None.
    """

    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Default: use all output dimensions
    if dims is None:
        dims = list(range(y_true.shape[1]))

    # NRMSE: average over selected dimensions
    nrmse_list = []
    for i in dims:
        y_true_dim = y_true[:, i]
        y_pred_dim = y_pred[:, i]
        rmse_dim = math.sqrt(mean_squared_error(y_true_dim, y_pred_dim))
        mean_true = np.mean(y_true_dim)
        if mean_true != 0:
            nrmse_dim = rmse_dim / mean_true
            nrmse_list.append(nrmse_dim)

    nrmse = np.mean(nrmse_list) if nrmse_list else 0.0

    if log_per_output:
        res = log_metrics(y_true, y_pred)
        if return_all:
            return mse, rmse, mae, r2, nrmse, res

    return mse, rmse, mae, r2, nrmse


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, delta=0, trace=True, trace_func=log):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace = trace
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.cache_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace:
                self.trace_func(INFO, f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.cache_checkpoint(val_loss, model)
            self.counter = 0

    def cache_checkpoint(self, val_loss, model):
        """Caches model when validation loss decrease.
        """
        if self.trace:
            self.trace_func(INFO, f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Caching "
                                  f"model ...")
        self.val_loss_min = val_loss
        self.best_model = copy.deepcopy(model)
