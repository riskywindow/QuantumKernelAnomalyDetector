"""Tests for One-Class SVM with precomputed kernels."""

import numpy as np
import pytest

from src.models.ocsvm import QuantumOCSVM
from src.utils.metrics import AnomalyMetrics


class TestQuantumOCSVM:
    """Tests for QuantumOCSVM."""

    @pytest.fixture
    def simple_kernel_matrices(self):
        """Create simple synthetic kernel matrices for testing.

        Normal points are clustered (high similarity), anomaly points
        have lower similarity to normals.
        """
        rng = np.random.default_rng(42)
        n_train = 30
        n_test = 20

        # K_train: 30x30 symmetric PSD matrix with unit diagonal
        # Normal points are similar to each other
        K_train = np.eye(n_train)
        for i in range(n_train):
            for j in range(i + 1, n_train):
                K_train[i, j] = 0.7 + 0.3 * rng.random()
                K_train[j, i] = K_train[i, j]

        # K_test: 20x30 cross-kernel matrix
        # First 15 are normal (high similarity), last 5 are anomalies (low similarity)
        K_test = np.zeros((n_test, n_train))
        for i in range(15):
            K_test[i] = 0.6 + 0.3 * rng.random(n_train)
        for i in range(15, 20):
            K_test[i] = 0.1 + 0.2 * rng.random(n_train)

        y_test = np.array([0] * 15 + [1] * 5)

        return K_train, K_test, y_test

    def test_fit_returns_self(self, simple_kernel_matrices):
        """fit() should return self for chaining."""
        K_train, _, _ = simple_kernel_matrices
        model = QuantumOCSVM(nu=0.1)
        result = model.fit(K_train)
        assert result is model

    def test_predict_scores_shape(self, simple_kernel_matrices):
        """predict_scores should return one score per test sample."""
        K_train, K_test, _ = simple_kernel_matrices
        model = QuantumOCSVM(nu=0.1).fit(K_train)
        scores = model.predict_scores(K_test)
        assert scores.shape == (K_test.shape[0],)

    def test_scores_negated(self, simple_kernel_matrices):
        """Scores should be negated decision_function (higher = anomalous)."""
        K_train, K_test, _ = simple_kernel_matrices
        model = QuantumOCSVM(nu=0.1).fit(K_train)
        scores = model.predict_scores(K_test)
        # Raw sklearn decision_function: positive = normal
        raw = model._model.decision_function(K_test)
        np.testing.assert_array_equal(scores, -raw)

    def test_evaluate_returns_metrics(self, simple_kernel_matrices):
        """evaluate() should return AnomalyMetrics."""
        K_train, K_test, y_test = simple_kernel_matrices
        model = QuantumOCSVM(nu=0.1).fit(K_train)
        metrics = model.evaluate(K_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)
        assert 0 <= metrics.auroc <= 1

    def test_predict_before_fit_raises(self):
        """predict_scores before fit should raise RuntimeError."""
        model = QuantumOCSVM()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_scores(np.eye(5))

    def test_kernel_name_stored(self):
        """kernel_name should be stored."""
        model = QuantumOCSVM(kernel_name="ZZ")
        assert model.kernel_name == "ZZ"

    def test_identity_kernel(self):
        """Identity kernel matrix (all samples identical) should work."""
        K_train = np.ones((10, 10))
        K_test = np.ones((5, 10))
        y_test = np.array([0, 0, 0, 1, 1])
        model = QuantumOCSVM(nu=0.5).fit(K_train)
        scores = model.predict_scores(K_test)
        assert len(scores) == 5
