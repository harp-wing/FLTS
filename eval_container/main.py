# eval_container/main.py

import numpy as np
from metrics import compute_metrics, save_metrics, plot_predictions

def run_evaluation():
    y_true = np.load("../outputs/predictions/elborn_ground_truth.npy")
    y_pred = np.load("../outputs/predictions/elborn_preds.npy")

    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics)
    plot_predictions(y_true, y_pred)

if __name__ == "__main__":
    run_evaluation()
