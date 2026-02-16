"""Quantum preprocessing pipeline: StandardScale -> PCA -> angle encoding."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class QuantumPreprocessor:
    """Preprocessing pipeline for quantum feature map encoding.

    Transforms raw features into rotation angles suitable for
    parameterized quantum circuits. Pipeline:
        1. StandardScale: zero mean, unit variance
        2. PCA: reduce to n_features dimensions
        3. MinMax rescale: map to [range_min, range_max] for angle encoding

    Follows the sklearn fit/transform pattern for consistent
    train -> test transformation.

    Args:
        n_features: Number of output features (= number of qubits).
        dim_reduction: Dimensionality reduction method. Currently supports 'pca'.
        range_min: Minimum angle value (default 0).
        range_max: Maximum angle value (default 2*pi).
    """

    def __init__(
        self,
        n_features: int,
        dim_reduction: str = "pca",
        range_min: float = 0.0,
        range_max: float = 2 * np.pi,
    ) -> None:
        self.n_features = n_features
        self.dim_reduction = dim_reduction
        self.range_min = range_min
        self.range_max = range_max

        self._scaler: StandardScaler | None = None
        self._reducer: PCA | None = None
        self._feature_min: np.ndarray | None = None
        self._feature_max: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> QuantumPreprocessor:
        """Fit the preprocessing pipeline on training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_original_features).

        Returns:
            self for method chaining.
        """
        X = np.asarray(X, dtype=np.float64)

        # Step 1: Standard scaling
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Step 2: Dimensionality reduction
        X_reduced = self._reduce_dimensions_fit(X_scaled)

        # Step 3: Compute min/max for angle rescaling
        self._feature_min = X_reduced.min(axis=0)
        self._feature_max = X_reduced.max(axis=0)

        # Handle constant features (avoid division by zero)
        constant_mask = self._feature_max == self._feature_min
        self._feature_max[constant_mask] = self._feature_min[constant_mask] + 1.0

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted preprocessing pipeline to data.

        Args:
            X: Feature matrix of shape (n_samples, n_original_features).

        Returns:
            Transformed array of shape (n_samples, n_features) with values
            in [range_min, range_max].

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError("QuantumPreprocessor must be fitted before transform.")

        X = np.asarray(X, dtype=np.float64)

        # Step 1: Standard scaling
        X_scaled = self._scaler.transform(X)

        # Step 2: Dimensionality reduction
        X_reduced = self._reduce_dimensions_transform(X_scaled)

        # Step 3: Rescale to angle range
        X_angles = self._rescale_to_angles(X_reduced)

        return X_angles

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the pipeline and transform the data in one step.

        Args:
            X: Training feature matrix of shape (n_samples, n_original_features).

        Returns:
            Transformed array of shape (n_samples, n_features) with values
            in [range_min, range_max].
        """
        self.fit(X)
        return self.transform(X)

    def _reduce_dimensions_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit and apply dimensionality reduction.

        Args:
            X: Scaled feature matrix.

        Returns:
            Reduced feature matrix of shape (n_samples, n_features).
        """
        if self.dim_reduction == "pca":
            self._reducer = PCA(n_components=self.n_features)
            return self._reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown dim_reduction method: {self.dim_reduction}")

    def _reduce_dimensions_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted dimensionality reduction to new data.

        Args:
            X: Scaled feature matrix.

        Returns:
            Reduced feature matrix of shape (n_samples, n_features).
        """
        if self.dim_reduction == "pca":
            return self._reducer.transform(X)
        else:
            raise ValueError(f"Unknown dim_reduction method: {self.dim_reduction}")

    def _rescale_to_angles(self, X: np.ndarray) -> np.ndarray:
        """Rescale features to [range_min, range_max] using fitted min/max.

        Args:
            X: Reduced feature matrix.

        Returns:
            Rescaled array with values in [range_min, range_max].
        """
        # Min-max normalize to [0, 1]
        X_normalized = (X - self._feature_min) / (self._feature_max - self._feature_min)

        # Clip to handle values slightly outside training range
        X_normalized = np.clip(X_normalized, 0.0, 1.0)

        # Scale to [range_min, range_max]
        return self.range_min + X_normalized * (self.range_max - self.range_min)
