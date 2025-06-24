import numpy as np
import os
import json
from eval_container.metrics import compute_metrics, save_metrics, plot_predictions

def test_compute_metrics_output_keys():
    y_true = np.random.rand(50, 2)
    y_pred = y_true + np.random.normal(0, 0.1, y_true.shape)

    metrics = compute_metrics(y_true, y_pred)
    assert "MSE" in metrics
    assert "MAE" in metrics
    assert "R2" in metrics

def test_save_metrics_and_plot(tmp_path):
    y_true = np.random.rand(100, 2)
    y_pred = y_true + np.random.normal(0, 0.05, size=y_true.shape)

    metrics_path = tmp_path / "metrics.json"
    plot_path = tmp_path / "plot.png"

    save_metrics(compute_metrics(y_true, y_pred), metrics_path)
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "MSE" in metrics

    plot_predictions(y_true, y_pred, path=plot_path)
    assert plot_path.exists()
