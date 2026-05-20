"""Evaluation metrics."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, task_type: str
) -> Dict[str, float]:
    if task_type == "classification":
        try:
            roc = roc_auc_score(y_true, y_pred)
        except ValueError:
            roc = float("nan")
        try:
            auprc = average_precision_score(y_true, y_pred)
        except ValueError:
            auprc = float("nan")
        return {"roc_auc": roc, "auprc": auprc}

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    try:
        from scipy.stats import spearmanr
        spearman = spearmanr(y_true, y_pred).correlation
    except Exception:
        spearman = float("nan")
    return {"rmse": rmse, "mae": mae, "spearman": spearman}
