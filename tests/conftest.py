"""Shared test fixtures for quantum kernel anomaly detection."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic dataset for testing (10 samples, 5 features)."""
    X = rng.random((10, 5))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def angle_data_2q(rng: np.random.Generator) -> np.ndarray:
    """Random angle data for 2-qubit tests, shape (5, 2) in [0, 2*pi]."""
    return rng.random((5, 2)) * 2 * np.pi


@pytest.fixture
def angle_data_3q(rng: np.random.Generator) -> np.ndarray:
    """Random angle data for 3-qubit tests, shape (5, 3) in [0, 2*pi]."""
    return rng.random((5, 3)) * 2 * np.pi
