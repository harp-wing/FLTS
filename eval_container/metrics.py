# eval_container/metrics.py

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import json
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score # type: ignore

def eval_forecast(y_true, y_pred):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    return metrics

def save_metrics(metrics: dict, path="outputs/metrics/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {path}")

def plot_predictions(x, y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each data series on the same axes
    ax.plot(x, y_true, linestyle='-', label='Actual Data')
    ax.plot(x, y_pred, linestyle='-', label='Model Prediction')

    # --- 3. Chart Customization ---
    # Add a title and labels for clarity
    ax.set_xlabel('Time')
    ax.set_ylabel('Traffic (bps)')

    # Add a legend to identify the lines
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    return fig, title

def test_plot():
    """Generates a random scatter plot."""
    title = "A Scatter Plot"
    fig, ax = plt.subplots()
    x = np.random.rand(50)
    y = np.random.rand(50)
    ax.scatter(x, y)
    ax.set_title(title)

    return fig, title
