"""Tests for anomaly detection metrics."""

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import AnomalyMetrics, compute_anomaly_metrics, compute_metrics_table


class TestComputeAnomalyMetrics:
    """Tests for compute_anomaly_metrics."""

    def test_perfect_scores_auroc(self):
        """Perfect separation should give AUROC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        metrics = compute_anomaly_metrics(y_true, scores)
        assert metrics.auroc == 1.0

    def test_random_scores_auroc(self):
        """Random scores should give AUROC near 0.5."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = np.concatenate([np.zeros(500), np.ones(500)])
        scores = rng.random(n)
        metrics = compute_anomaly_metrics(y_true, scores)
        assert abs(metrics.auroc - 0.5) < 0.1

    def test_perfect_scores_auprc(self):
        """Perfect separation should give AUPRC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        metrics = compute_anomaly_metrics(y_true, scores)
        assert metrics.auprc == 1.0

    def test_metrics_in_valid_range(self):
        """All metrics should be in [0, 1]."""
        rng = np.random.default_rng(123)
        y_true = np.concatenate([np.zeros(80), np.ones(20)])
        scores = rng.random(100)
        metrics = compute_anomaly_metrics(y_true, scores)
        assert 0 <= metrics.auroc <= 1
        assert 0 <= metrics.auprc <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.fpr_at_95_recall <= 1

    def test_f1_precision_recall_consistency(self):
        """F1 should equal 2*P*R/(P+R) at optimal threshold."""
        rng = np.random.default_rng(99)
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        scores = rng.random(100)
        metrics = compute_anomaly_metrics(y_true, scores)
        expected_f1 = (
            2 * metrics.precision * metrics.recall
            / (metrics.precision + metrics.recall)
        )
        assert abs(metrics.f1 - expected_f1) < 1e-10

    def test_fpr_at_95_recall_perfect(self):
        """With perfect scores, FPR at 95% recall should be 0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        metrics = compute_anomaly_metrics(y_true, scores)
        assert metrics.fpr_at_95_recall == 0.0

    def test_single_class_raises(self):
        """Should raise ValueError if only one class present."""
        y_true = np.array([0, 0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        with pytest.raises(ValueError, match="must contain both classes"):
            compute_anomaly_metrics(y_true, scores)

    def test_returns_anomaly_metrics_dataclass(self):
        """Should return an AnomalyMetrics instance."""
        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = compute_anomaly_metrics(y_true, scores)
        assert isinstance(metrics, AnomalyMetrics)

    def test_optimal_threshold_is_float(self):
        """Optimal threshold should be a float."""
        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = compute_anomaly_metrics(y_true, scores)
        assert isinstance(metrics.optimal_threshold, float)


class TestComputeMetricsTable:
    """Tests for compute_metrics_table."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        results = {
            "Model A": AnomalyMetrics(0.9, 0.8, 0.7, 0.75, 0.65, 0.1, 0.5),
            "Model B": AnomalyMetrics(0.85, 0.75, 0.65, 0.7, 0.6, 0.15, 0.45),
        }
        table = compute_metrics_table(results)
        assert isinstance(table, pd.DataFrame)

    def test_sorted_by_auroc_descending(self):
        """Models should be sorted by AUROC descending."""
        results = {
            "Worse": AnomalyMetrics(0.7, 0.6, 0.5, 0.55, 0.45, 0.3, 0.4),
            "Better": AnomalyMetrics(0.9, 0.8, 0.7, 0.75, 0.65, 0.1, 0.5),
            "Middle": AnomalyMetrics(0.8, 0.7, 0.6, 0.65, 0.55, 0.2, 0.45),
        }
        table = compute_metrics_table(results)
        aurocs = table["AUROC"].tolist()
        assert aurocs == sorted(aurocs, reverse=True)

    def test_all_columns_present(self):
        """Table should have all expected columns."""
        results = {
            "Model A": AnomalyMetrics(0.9, 0.8, 0.7, 0.75, 0.65, 0.1, 0.5),
        }
        table = compute_metrics_table(results)
        expected_cols = {"AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR@95%Recall", "Threshold"}
        assert set(table.columns) == expected_cols

    def test_model_names_as_index(self):
        """Model names should be the index."""
        results = {
            "Model A": AnomalyMetrics(0.9, 0.8, 0.7, 0.75, 0.65, 0.1, 0.5),
            "Model B": AnomalyMetrics(0.85, 0.75, 0.65, 0.7, 0.6, 0.15, 0.45),
        }
        table = compute_metrics_table(results)
        assert table.index.name == "Model"
        assert set(table.index) == {"Model A", "Model B"}

    def test_values_rounded_to_4_decimals(self):
        """Values should be rounded to 4 decimal places."""
        results = {
            "Model": AnomalyMetrics(0.912345, 0.8, 0.7, 0.75, 0.65, 0.1, 0.5),
        }
        table = compute_metrics_table(results)
        assert table.loc["Model", "AUROC"] == 0.9123
