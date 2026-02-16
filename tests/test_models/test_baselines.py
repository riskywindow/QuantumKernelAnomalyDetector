"""Tests for classical baseline anomaly detectors."""

import numpy as np
import pytest

from src.models.baselines import AutoencoderBaseline, IsolationForestBaseline, LOFBaseline
from src.utils.metrics import AnomalyMetrics


@pytest.fixture
def synthetic_data():
    """Generate synthetic normal/anomaly data for baseline testing.

    Normal data is clustered around origin, anomalies are far away.
    """
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (50, 5))
    X_test_normal = rng.normal(0, 1, (15, 5))
    X_test_anomaly = rng.normal(5, 1, (5, 5))
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * 15 + [1] * 5)
    return X_train, X_test, y_test


class TestIsolationForestBaseline:
    """Tests for IsolationForestBaseline."""

    def test_fit_returns_self(self, synthetic_data):
        """fit() should return self."""
        X_train, _, _ = synthetic_data
        model = IsolationForestBaseline(seed=42)
        result = model.fit(X_train)
        assert result is model

    def test_predict_scores_shape(self, synthetic_data):
        """predict_scores should return one score per sample."""
        X_train, X_test, _ = synthetic_data
        model = IsolationForestBaseline(seed=42).fit(X_train)
        scores = model.predict_scores(X_test)
        assert scores.shape == (len(X_test),)

    def test_scores_negated(self, synthetic_data):
        """Scores should be negated decision_function."""
        X_train, X_test, _ = synthetic_data
        model = IsolationForestBaseline(seed=42).fit(X_train)
        scores = model.predict_scores(X_test)
        raw = model._model.decision_function(X_test)
        np.testing.assert_array_equal(scores, -raw)

    def test_evaluate_returns_metrics(self, synthetic_data):
        """evaluate() should return AnomalyMetrics."""
        X_train, X_test, y_test = synthetic_data
        model = IsolationForestBaseline(seed=42).fit(X_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)
        assert 0 <= metrics.auroc <= 1

    def test_predict_before_fit_raises(self):
        """predict_scores before fit should raise RuntimeError."""
        model = IsolationForestBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_scores(np.zeros((5, 3)))

    def test_detects_obvious_anomalies(self, synthetic_data):
        """Should detect obviously anomalous points."""
        X_train, X_test, y_test = synthetic_data
        model = IsolationForestBaseline(seed=42).fit(X_train)
        metrics = model.evaluate(X_test, y_test)
        assert metrics.auroc > 0.7


class TestAutoencoderBaseline:
    """Tests for AutoencoderBaseline."""

    def test_fit_returns_self(self, synthetic_data):
        """fit() should return self."""
        X_train, _, _ = synthetic_data
        model = AutoencoderBaseline(epochs=5, seed=42)
        result = model.fit(X_train)
        assert result is model

    def test_predict_scores_shape(self, synthetic_data):
        """predict_scores should return one score per sample."""
        X_train, X_test, _ = synthetic_data
        model = AutoencoderBaseline(epochs=5, seed=42).fit(X_train)
        scores = model.predict_scores(X_test)
        assert scores.shape == (len(X_test),)

    def test_scores_nonnegative(self, synthetic_data):
        """MSE reconstruction error should be non-negative."""
        X_train, X_test, _ = synthetic_data
        model = AutoencoderBaseline(epochs=5, seed=42).fit(X_train)
        scores = model.predict_scores(X_test)
        assert np.all(scores >= 0)

    def test_scores_are_float64(self, synthetic_data):
        """Scores should be float64."""
        X_train, X_test, _ = synthetic_data
        model = AutoencoderBaseline(epochs=5, seed=42).fit(X_train)
        scores = model.predict_scores(X_test)
        assert scores.dtype == np.float64

    def test_evaluate_returns_metrics(self, synthetic_data):
        """evaluate() should return AnomalyMetrics."""
        X_train, X_test, y_test = synthetic_data
        model = AutoencoderBaseline(epochs=10, seed=42).fit(X_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)

    def test_predict_before_fit_raises(self):
        """predict_scores before fit should raise RuntimeError."""
        model = AutoencoderBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_scores(np.zeros((5, 3)))


class TestLOFBaseline:
    """Tests for LOFBaseline."""

    def test_fit_returns_self(self, synthetic_data):
        """fit() should return self."""
        X_train, _, _ = synthetic_data
        model = LOFBaseline()
        result = model.fit(X_train)
        assert result is model

    def test_predict_scores_shape(self, synthetic_data):
        """predict_scores should return one score per sample."""
        X_train, X_test, _ = synthetic_data
        model = LOFBaseline().fit(X_train)
        scores = model.predict_scores(X_test)
        assert scores.shape == (len(X_test),)

    def test_scores_negated(self, synthetic_data):
        """Scores should be negated decision_function."""
        X_train, X_test, _ = synthetic_data
        model = LOFBaseline().fit(X_train)
        scores = model.predict_scores(X_test)
        raw = model._model.decision_function(X_test)
        np.testing.assert_array_equal(scores, -raw)

    def test_evaluate_returns_metrics(self, synthetic_data):
        """evaluate() should return AnomalyMetrics."""
        X_train, X_test, y_test = synthetic_data
        model = LOFBaseline().fit(X_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)

    def test_predict_before_fit_raises(self):
        """predict_scores before fit should raise RuntimeError."""
        model = LOFBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_scores(np.zeros((5, 3)))

    def test_n_neighbors_clamped(self, synthetic_data):
        """n_neighbors should be clamped to n_train - 1."""
        X_train, X_test, _ = synthetic_data
        model = LOFBaseline(n_neighbors=1000).fit(X_train)
        scores = model.predict_scores(X_test)
        assert len(scores) == len(X_test)


class TestBaselineInterface:
    """Tests that all baselines follow the same interface."""

    @pytest.mark.parametrize(
        "model_cls,kwargs",
        [
            (IsolationForestBaseline, {"seed": 42}),
            (AutoencoderBaseline, {"epochs": 5, "seed": 42}),
            (LOFBaseline, {}),
        ],
    )
    def test_common_interface(self, model_cls, kwargs, synthetic_data):
        """All baselines should support fit/predict_scores/evaluate."""
        X_train, X_test, y_test = synthetic_data
        model = model_cls(**kwargs)

        # fit
        result = model.fit(X_train)
        assert result is model

        # predict_scores
        scores = model.predict_scores(X_test)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(X_test),)

        # evaluate
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)
        assert 0 <= metrics.auroc <= 1
