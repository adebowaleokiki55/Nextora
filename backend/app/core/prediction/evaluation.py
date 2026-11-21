"""Forecast accuracy metrics (plain NumPy)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = {
    "baseline_persistence": "Persistence (last observed week)",
    "baseline_seasonal_naive": "Seasonal naive (same week last year)",
    "baseline_hist_mean": "Historical weekly mean (same week, earlier years)",
}


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    if a.size == 0:
        return {"mae": None, "rmse": None, "n": 0}
    return {
        "mae": round(float(np.mean(np.abs(a - p))), 2),
        "rmse": round(float(np.sqrt(np.mean((a - p) ** 2))), 2),
        "n": int(a.size),
    }


def crossing_metrics(actual_cross: np.ndarray, predicted_cross: np.ndarray) -> dict[str, float | int | None]:
    """Precision/recall for 'cases crossed the alert threshold'."""
    a = actual_cross.astype(bool)
    p = predicted_cross.astype(bool)
    tp = int(np.sum(a & p))
    fp = int(np.sum(~a & p))
    fn = int(np.sum(a & ~p))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "actual_crossings": int(a.sum()),
        "weeks_evaluated": int(a.size),
    }


def evaluate_horizon(rows: pd.DataFrame, model_predictions: np.ndarray, model_crossing: np.ndarray) -> dict:
    """Compare the model and the three baselines on the same evaluation rows."""
    actual = rows["target_cases"].to_numpy(dtype=float)
    threshold = rows["threshold"].to_numpy(dtype=float)
    actual_cross = actual >= threshold

    result = {
        "model": {
            **error_metrics(actual, model_predictions),
            "crossing": crossing_metrics(actual_cross, model_crossing),
        }
    }
    for column, label in BASELINES.items():
        predicted = rows[column].to_numpy(dtype=float)
        result[column] = {
            "label": label,
            **error_metrics(actual, predicted),
            "crossing": crossing_metrics(actual_cross, np.nan_to_num(predicted, nan=-1) >= threshold),
        }
    return result
