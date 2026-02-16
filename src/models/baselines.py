"""Classical anomaly detection baselines.

These baselines operate directly on preprocessed feature vectors (not kernel
matrices). They provide context for how well the quantum/classical kernel
approaches perform relative to standard methods.

All baselines follow the same interface: fit(X_train), predict_scores(X_test),
evaluate(X_test, y_test).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.utils.metrics import AnomalyMetrics, compute_anomaly_metrics


class IsolationForestBaseline:
    """Isolation Forest anomaly detector.

    Args:
        n_estimators: Number of isolation trees.
        contamination: Expected fraction of anomalies in training data.
            Since our training set is 100% normal, use a very small value.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self._model: IsolationForest | None = None

    def fit(self, X_train: np.ndarray) -> IsolationForestBaseline:
        """Fit Isolation Forest on training data.

        Args:
            X_train: Training features of shape (n_train, n_features).

        Returns:
            self for method chaining.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self._model.fit(X_train)
        return self

    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for test data.

        Args:
            X_test: Test features of shape (n_test, n_features).

        Returns:
            Anomaly scores where HIGHER = more anomalous.

        Raises:
            RuntimeError: If called before fit.
        """
        if self._model is None:
            raise RuntimeError(
                "IsolationForestBaseline must be fitted before predict_scores."
            )
        X_test = np.asarray(X_test, dtype=np.float64)
        # sklearn IsolationForest: positive = normal, negative = anomaly
        return -self._model.decision_function(X_test)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> AnomalyMetrics:
        """Compute anomaly scores and evaluate against ground truth.

        Args:
            X_test: Test features of shape (n_test, n_features).
            y_test: Binary labels (0 = normal, 1 = anomaly).

        Returns:
            AnomalyMetrics with all evaluation metrics.
        """
        scores = self.predict_scores(X_test)
        return compute_anomaly_metrics(y_test, scores)


class _Autoencoder(nn.Module):
    """Simple autoencoder for reconstruction-based anomaly detection."""

    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderBaseline:
    """Autoencoder-based anomaly detector using reconstruction error.

    Trains ONLY on normal data. Anomalies are detected as points
    with high reconstruction error (MSE per sample).

    Args:
        encoding_dim: Dimension of the bottleneck layer.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        encoding_dim: int = 3,
        epochs: int = 50,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._model: _Autoencoder | None = None
        self._is_fitted = False

    def fit(self, X_train: np.ndarray) -> AutoencoderBaseline:
        """Train the autoencoder on normal data.

        Args:
            X_train: Training features of shape (n_train, n_features).

        Returns:
            self for method chaining.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_train = np.asarray(X_train, dtype=np.float32)
        input_dim = X_train.shape[1]

        self._model = _Autoencoder(input_dim, self.encoding_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X_tensor = torch.from_numpy(X_train)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self._model.train()
        for _ in range(self.epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()

        self._is_fitted = True
        return self

    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores as reconstruction error (MSE per sample).

        Args:
            X_test: Test features of shape (n_test, n_features).

        Returns:
            Anomaly scores where HIGHER = more anomalous.

        Raises:
            RuntimeError: If called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "AutoencoderBaseline must be fitted before predict_scores."
            )

        X_test = np.asarray(X_test, dtype=np.float32)
        X_tensor = torch.from_numpy(X_test)

        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(X_tensor)

        # MSE per sample
        mse = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()
        return mse.astype(np.float64)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> AnomalyMetrics:
        """Compute anomaly scores and evaluate against ground truth.

        Args:
            X_test: Test features of shape (n_test, n_features).
            y_test: Binary labels (0 = normal, 1 = anomaly).

        Returns:
            AnomalyMetrics with all evaluation metrics.
        """
        scores = self.predict_scores(X_test)
        return compute_anomaly_metrics(y_test, scores)


class LOFBaseline:
    """Local Outlier Factor anomaly detector.

    Uses novelty detection mode (fit on normal training data, score new test data).

    Args:
        n_neighbors: Number of neighbors for LOF computation.
        contamination: Expected fraction of anomalies.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.001,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._model: LocalOutlierFactor | None = None

    def fit(self, X_train: np.ndarray) -> LOFBaseline:
        """Fit LOF in novelty detection mode.

        Args:
            X_train: Training features of shape (n_train, n_features).

        Returns:
            self for method chaining.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        self._model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X_train) - 1),
            contamination=self.contamination,
            novelty=True,
        )
        self._model.fit(X_train)
        return self

    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for test data.

        Args:
            X_test: Test features of shape (n_test, n_features).

        Returns:
            Anomaly scores where HIGHER = more anomalous.

        Raises:
            RuntimeError: If called before fit.
        """
        if self._model is None:
            raise RuntimeError(
                "LOFBaseline must be fitted before predict_scores."
            )
        X_test = np.asarray(X_test, dtype=np.float64)
        # sklearn LOF: positive = normal, negative = anomaly
        return -self._model.decision_function(X_test)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> AnomalyMetrics:
        """Compute anomaly scores and evaluate against ground truth.

        Args:
            X_test: Test features of shape (n_test, n_features).
            y_test: Binary labels (0 = normal, 1 = anomaly).

        Returns:
            AnomalyMetrics with all evaluation metrics.
        """
        scores = self.predict_scores(X_test)
        return compute_anomaly_metrics(y_test, scores)
