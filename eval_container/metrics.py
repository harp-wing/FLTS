# eval_container/metrics.py

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore

def compute_metrics(y_true, y_pred):
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    return metrics

def save_metrics(metrics: dict, path="outputs/metrics/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {path}")

def plot_predictions(y_true, y_pred, path="outputs/metrics/prediction_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:, 0], label="Actual")
    plt.plot(y_pred[:, 0], label="Predicted")
    plt.title("Predicted vs Actual (first target)")
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"📈 Plot saved to {path}")

def test_plot():
    """Generates a random scatter plot."""
    title = "A Scatter Plot"
    fig, ax = plt.subplots()
    x = np.random.rand(50)
    y = np.random.rand(50)
    ax.scatter(x, y)
    ax.set_title(title)

    return fig, title
