"""Evaluation metrics for anomaly detection.

Computes AUROC, AUPRC, F1, precision, recall, and operational
metrics (FPR at 95% recall) for anomaly detection models.
All models in the project are evaluated through this single interface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class AnomalyMetrics:
    """Container for anomaly detection evaluation metrics."""

    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    fpr_at_95_recall: float
    optimal_threshold: float


def compute_anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> AnomalyMetrics:
    """Compute all anomaly detection evaluation metrics.

    Args:
        y_true: Binary labels (0 = normal, 1 = anomaly).
        scores: Anomaly scores where HIGHER = more anomalous.

    Returns:
        AnomalyMetrics with all computed metrics.

    Raises:
        ValueError: If y_true contains fewer than 2 classes.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"y_true must contain both classes. Got {n_pos} positives, {n_neg} negatives."
        )

    # AUROC
    auroc = float(roc_auc_score(y_true, scores))

    # AUPRC
    auprc = float(average_precision_score(y_true, scores))

    # F1 / precision / recall at optimal threshold
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="invalid value"
        )
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns n+1 precisions/recalls but n thresholds
    # Last precision=1, recall=0 is a sentinel — exclude it for F1 computation
    precisions_t = precisions[:-1]
    recalls_t = recalls[:-1]
    denom = precisions_t + recalls_t
    denom_safe = np.where(denom > 0, denom, 1.0)
    f1_scores = np.where(
        denom > 0,
        2 * precisions_t * recalls_t / denom_safe,
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions_t[best_idx])
    best_recall = float(recalls_t[best_idx])
    best_threshold = float(thresholds[best_idx])

    # FPR at 95% recall
    fpr_values, tpr_values, _ = roc_curve(y_true, scores)
    # Find the FPR where TPR >= 0.95
    mask = tpr_values >= 0.95
    if mask.any():
        fpr_at_95 = float(fpr_values[mask][0])
    else:
        # If we never reach 95% recall, return 1.0 (worst case)
        fpr_at_95 = 1.0

    return AnomalyMetrics(
        auroc=auroc,
        auprc=auprc,
        f1=best_f1,
        precision=best_precision,
        recall=best_recall,
        fpr_at_95_recall=fpr_at_95,
        optimal_threshold=best_threshold,
    )


def compute_metrics_table(
    results: dict[str, AnomalyMetrics],
) -> pd.DataFrame:
    """Build a comparison table from multiple model results.

    Args:
        results: Dict mapping model names to their AnomalyMetrics.

    Returns:
        DataFrame with models as rows and metrics as columns,
        sorted by AUROC descending. Floats formatted to 4 decimal places.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "AUROC": round(metrics.auroc, 4),
                "AUPRC": round(metrics.auprc, 4),
                "F1": round(metrics.f1, 4),
                "Precision": round(metrics.precision, 4),
                "Recall": round(metrics.recall, 4),
                "FPR@95%Recall": round(metrics.fpr_at_95_recall, 4),
                "Threshold": round(metrics.optimal_threshold, 4),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("AUROC", ascending=False).reset_index(drop=True)
    df = df.set_index("Model")
    return df
