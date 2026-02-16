"""One-Class SVM with precomputed kernel matrices.

Learns a boundary around normal data in kernel space, then flags
points outside the boundary as anomalies. Primary model for
quantum kernel anomaly detection.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from src.utils.metrics import AnomalyMetrics, compute_anomaly_metrics


class QuantumOCSVM:
    """One-Class SVM using precomputed kernel matrices.

    Trains ONLY on normal data and detects anomalies as points
    that fall outside the learned boundary in kernel space.

    Args:
        nu: Upper bound on fraction of training errors / lower bound
            on fraction of support vectors. 0.05 is a good default
            for ~5% contamination expectation.
        kernel_name: Label for tracking/display purposes.
    """

    def __init__(self, nu: float = 0.05, kernel_name: str = "quantum") -> None:
        self.nu = nu
        self.kernel_name = kernel_name
        self._model: OneClassSVM | None = None

    def fit(self, K_train: np.ndarray) -> QuantumOCSVM:
        """Fit the One-Class SVM on a precomputed kernel matrix.

        Args:
            K_train: Precomputed kernel matrix of shape (n_train, n_train)
                where ALL samples are normal.

        Returns:
            self for method chaining.
        """
        K_train = np.asarray(K_train, dtype=np.float64)
        self._model = OneClassSVM(kernel="precomputed", nu=self.nu)
        self._model.fit(K_train)
        return self

    def predict_scores(self, K_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for test data.

        Args:
            K_test: Precomputed cross-kernel matrix of shape (n_test, n_train).

        Returns:
            Anomaly scores where HIGHER = more anomalous.

        Raises:
            RuntimeError: If called before fit.
        """
        if self._model is None:
            raise RuntimeError("QuantumOCSVM must be fitted before predict_scores.")

        K_test = np.asarray(K_test, dtype=np.float64)
        # sklearn OneClassSVM: positive = normal, negative = anomaly
        # Negate so higher = more anomalous
        return -self._model.decision_function(K_test)

    def evaluate(
        self, K_test: np.ndarray, y_test: np.ndarray
    ) -> AnomalyMetrics:
        """Compute anomaly scores and evaluate against ground truth.

        Args:
            K_test: Precomputed cross-kernel matrix of shape (n_test, n_train).
            y_test: Binary labels (0 = normal, 1 = anomaly).

        Returns:
            AnomalyMetrics with all evaluation metrics.
        """
        scores = self.predict_scores(K_test)
        return compute_anomaly_metrics(y_test, scores)
