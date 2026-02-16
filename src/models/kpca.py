"""Kernel PCA anomaly detector with precomputed kernels.

Projects data into kernel principal component space and uses
reconstruction error as anomaly score. Points that project
weakly onto the training principal components have high
reconstruction error and are flagged as anomalies.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer

from src.utils.metrics import AnomalyMetrics, compute_anomaly_metrics


class KernelPCAAnomalyDetector:
    """Anomaly detection via Kernel PCA reconstruction error.

    Projects data into kernel principal component space. Normal data
    projects strongly onto the learned PCs (high projection norm),
    while anomalies project weakly (low norm). The anomaly score is
    the negative projection norm: higher = more anomalous.

    Args:
        n_components: Number of principal components to retain.
        kernel_name: Label for tracking/display purposes.
    """

    def __init__(
        self, n_components: int = 10, kernel_name: str = "quantum"
    ) -> None:
        self.n_components = n_components
        self.kernel_name = kernel_name
        self._kpca: KernelPCA | None = None
        self._centerer: KernelCenterer | None = None
        self._train_max_norm_sq: float = 0.0
        self._is_fitted = False

    def fit(self, K_train: np.ndarray) -> KernelPCAAnomalyDetector:
        """Fit Kernel PCA on a precomputed kernel matrix.

        Args:
            K_train: Precomputed kernel matrix of shape (n_train, n_train).

        Returns:
            self for method chaining.
        """
        K_train = np.asarray(K_train, dtype=np.float64)

        # Center the kernel matrix
        self._centerer = KernelCenterer()
        K_centered = self._centerer.fit_transform(K_train)

        # Fit Kernel PCA
        # Clamp n_components to available dimensions
        n_components = min(self.n_components, K_train.shape[0])
        self._kpca = KernelPCA(
            n_components=n_components,
            kernel="precomputed",
            fit_inverse_transform=False,
        )
        Z_train = self._kpca.fit_transform(K_centered)

        # Store max squared norm from training for score normalization
        train_norms_sq = np.sum(Z_train ** 2, axis=1)
        self._train_max_norm_sq = float(train_norms_sq.max())
        self._is_fitted = True
        return self

    def predict_scores(self, K_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for test data.

        Uses reconstruction error as anomaly score: normal points project
        strongly onto training PCs (high squared norm), anomalies project
        weakly (low squared norm). Score = max_train_norm_sq - test_norm_sq,
        so higher score = weaker projection = more anomalous.

        Args:
            K_test: Precomputed cross-kernel matrix of shape (n_test, n_train).

        Returns:
            Anomaly scores (non-negative), one per test sample.

        Raises:
            RuntimeError: If called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "KernelPCAAnomalyDetector must be fitted before predict_scores."
            )

        K_test = np.asarray(K_test, dtype=np.float64)

        # Center using the fitted centerer
        K_test_centered = self._centerer.transform(K_test)

        # Project into PCA space
        Z_test = self._kpca.transform(K_test_centered)

        # Reconstruction error proxy: points that project weakly
        # onto the training PCs have low squared norm → high anomaly score
        test_norms_sq = np.sum(Z_test ** 2, axis=1)
        scores = self._train_max_norm_sq - test_norms_sq

        return scores

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
