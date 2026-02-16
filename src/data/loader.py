"""Dataset loading and preprocessing for credit card fraud detection."""

import ssl
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def _ensure_ssl_certificates() -> None:
    """Fix SSL certificate verification on macOS with Python.org installs."""
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )


def load_credit_card_fraud(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Credit Card Fraud Detection dataset from OpenML.

    Downloads via sklearn.datasets.fetch_openml on first call and caches
    to a parquet file for instant subsequent loads.

    Args:
        data_dir: Directory to cache the downloaded dataset.

    Returns:
        Tuple of (features DataFrame, target Series) where target is
        0 for normal transactions and 1 for fraud.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    features_path = data_dir / "creditcard_features.parquet"
    target_path = data_dir / "creditcard_target.parquet"

    if features_path.exists() and target_path.exists():
        X = pd.read_parquet(features_path)
        y = pd.read_parquet(target_path).squeeze()
        return X, y

    # Fix SSL for macOS environments
    _ensure_ssl_certificates()

    # Download from OpenML (no authentication required)
    data = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    X = data.data
    y = data.target

    # Ensure target is integer (0/1)
    y = y.astype(int)

    # Cache to parquet for fast subsequent loads
    X.to_parquet(features_path)
    y.to_frame().to_parquet(target_path)

    return X, y


def prepare_anomaly_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    normal_train_size: int = 1000,
    test_size: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test split for one-class anomaly detection.

    Training set contains ONLY normal samples (class 0).
    Test set contains both normal and anomaly samples (stratified).

    Args:
        X: Feature matrix.
        y: Binary labels (0 = normal, 1 = anomaly/fraud).
        normal_train_size: Number of normal samples for training.
        test_size: Total number of samples in test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_test) where X_train contains only
        normal samples and X_test contains both normal and anomaly samples.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    rng = np.random.default_rng(seed)

    # Separate normal and anomaly indices
    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]

    # Shuffle indices
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    # Reserve normal samples for training
    train_idx = normal_idx[:normal_train_size]
    remaining_normal_idx = normal_idx[normal_train_size:]

    # Determine test set composition (stratified)
    # Keep the same anomaly ratio as the full dataset, but ensure we have anomalies
    n_anomaly_test = min(len(anomaly_idx), max(1, int(test_size * 0.1)))
    n_normal_test = test_size - n_anomaly_test

    if n_normal_test > len(remaining_normal_idx):
        n_normal_test = len(remaining_normal_idx)
        n_anomaly_test = min(len(anomaly_idx), test_size - n_normal_test)

    test_normal_idx = remaining_normal_idx[:n_normal_test]
    test_anomaly_idx = anomaly_idx[:n_anomaly_test]

    test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Shuffle test set
    shuffle_order = rng.permutation(len(X_test))
    X_test = X_test[shuffle_order]
    y_test = y_test[shuffle_order]

    return X_train, X_test, y_test
