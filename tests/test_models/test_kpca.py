"""Tests for Kernel PCA anomaly detector."""

import numpy as np
import pytest

from src.models.kpca import KernelPCAAnomalyDetector
from src.utils.metrics import AnomalyMetrics


class TestKernelPCAAnomalyDetector:
    """Tests for KernelPCAAnomalyDetector."""

    @pytest.fixture
    def simple_kernel_matrices(self):
        """Create kernel matrices from feature-space data where anomalies are clear.

        Uses RBF kernel on synthetic data: normals clustered at origin,
        anomalies far away. This guarantees realistic kernel structure.
        """
        from sklearn.metrics.pairwise import rbf_kernel

        rng = np.random.default_rng(42)

        # Normal training data: tight cluster at origin
        X_train = rng.normal(0, 1, (30, 5))

        # Test data: 15 normals near origin, 5 anomalies shifted
        X_test_normal = rng.normal(0, 1, (15, 5))
        X_test_anomaly = rng.normal(3, 1, (5, 5))
        X_test = np.vstack([X_test_normal, X_test_anomaly])
        y_test = np.array([0] * 15 + [1] * 5)

        gamma = 0.1
        K_train = rbf_kernel(X_train, gamma=gamma)
        K_test = rbf_kernel(X_test, X_train, gamma=gamma)

        return K_train, K_test, y_test

    def test_fit_returns_self(self, simple_kernel_matrices):
        """fit() should return self for chaining."""
        K_train, _, _ = simple_kernel_matrices
        model = KernelPCAAnomalyDetector(n_components=5)
        result = model.fit(K_train)
        assert result is model

    def test_predict_scores_shape(self, simple_kernel_matrices):
        """predict_scores should return one score per test sample."""
        K_train, K_test, _ = simple_kernel_matrices
        model = KernelPCAAnomalyDetector(n_components=5).fit(K_train)
        scores = model.predict_scores(K_test)
        assert scores.shape == (K_test.shape[0],)

    def test_scores_are_finite(self, simple_kernel_matrices):
        """Scores should be finite real numbers."""
        K_train, K_test, _ = simple_kernel_matrices
        model = KernelPCAAnomalyDetector(n_components=5).fit(K_train)
        scores = model.predict_scores(K_test)
        assert np.all(np.isfinite(scores))

    def test_auroc_above_chance(self, simple_kernel_matrices):
        """On easy synthetic data, AUROC should be better than random."""
        K_train, K_test, y_test = simple_kernel_matrices
        model = KernelPCAAnomalyDetector(n_components=5).fit(K_train)
        metrics = model.evaluate(K_test, y_test)
        # Should be better than random (0.5)
        assert metrics.auroc > 0.5

    def test_evaluate_returns_metrics(self, simple_kernel_matrices):
        """evaluate() should return AnomalyMetrics."""
        K_train, K_test, y_test = simple_kernel_matrices
        model = KernelPCAAnomalyDetector(n_components=5).fit(K_train)
        metrics = model.evaluate(K_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)
        assert 0 <= metrics.auroc <= 1

    def test_predict_before_fit_raises(self):
        """predict_scores before fit should raise RuntimeError."""
        model = KernelPCAAnomalyDetector()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_scores(np.eye(5))

    def test_n_components_clamped(self):
        """n_components should be clamped to matrix size."""
        K_train = np.eye(5)
        for i in range(5):
            for j in range(i + 1, 5):
                K_train[i, j] = K_train[j, i] = 0.5
        model = KernelPCAAnomalyDetector(n_components=100).fit(K_train)
        assert model._is_fitted

    def test_small_matrix(self):
        """Should work with very small kernel matrices."""
        K_train = np.array([[1.0, 0.5, 0.3],
                            [0.5, 1.0, 0.4],
                            [0.3, 0.4, 1.0]])
        K_test = np.array([[0.6, 0.5, 0.4],
                           [0.1, 0.1, 0.1]])
        y_test = np.array([0, 1])
        model = KernelPCAAnomalyDetector(n_components=2).fit(K_train)
        metrics = model.evaluate(K_test, y_test)
        assert isinstance(metrics, AnomalyMetrics)
