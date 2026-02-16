"""Tests for synthetic advantage dataset generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.synthetic import (
    generate_quantum_advantage_dataset,
    generate_quantum_advantage_split,
)
from src.kernels.feature_maps.zz import ZZFeatureMap


@pytest.fixture
def zz_3qubit() -> ZZFeatureMap:
    """ZZ feature map with 3 qubits for fast tests."""
    return ZZFeatureMap(n_qubits=3, reps=1, entanglement="linear")


class TestGenerateQuantumAdvantageDataset:
    """Tests for the main dataset generation function."""

    def test_correct_shapes(self, zz_3qubit: ZZFeatureMap) -> None:
        """Output shapes should match n_samples and n_features."""
        X, y = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=50, n_features=3, seed=42
        )
        assert X.shape == (50, 3)
        assert y.shape == (50,)

    def test_data_in_range(self, zz_3qubit: ZZFeatureMap) -> None:
        """Data should be in [0, 2*pi]."""
        X, _ = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=50, n_features=3, seed=42
        )
        assert np.all(X >= 0)
        assert np.all(X <= 2 * np.pi)

    def test_labels_binary(self, zz_3qubit: ZZFeatureMap) -> None:
        """Labels should be 0 or 1."""
        _, y = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=50, n_features=3, seed=42
        )
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_approximately_balanced(self, zz_3qubit: ZZFeatureMap) -> None:
        """Labels should be approximately balanced (within noise)."""
        _, y = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=200, n_features=3, noise_rate=0.0, seed=42
        )
        balance = y.mean()
        # Median split -> approximately 50%, allow some variance
        assert 0.3 <= balance <= 0.7

    def test_zero_noise_deterministic(self, zz_3qubit: ZZFeatureMap) -> None:
        """With noise_rate=0, labels should be deterministic given the seed."""
        X1, y1 = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=30, n_features=3, noise_rate=0.0, seed=42
        )
        X2, y2 = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=30, n_features=3, noise_rate=0.0, seed=42
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_high_noise_approximately_random(self, zz_3qubit: ZZFeatureMap) -> None:
        """With noise_rate=0.5, labels should be approximately random."""
        _, y_no_noise = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=200, n_features=3, noise_rate=0.0, seed=42
        )
        _, y_high_noise = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=200, n_features=3, noise_rate=0.5, seed=42
        )
        # Many labels should differ between noisy and clean versions
        diff_rate = np.mean(y_no_noise != y_high_noise)
        assert diff_rate > 0.2  # Substantial difference

    def test_feature_mismatch_raises(self, zz_3qubit: ZZFeatureMap) -> None:
        """Should raise if n_features doesn't match feature map n_qubits."""
        with pytest.raises(ValueError, match="n_features"):
            generate_quantum_advantage_dataset(
                zz_3qubit, n_samples=10, n_features=5, seed=42
            )

    def test_different_seeds_different_data(self, zz_3qubit: ZZFeatureMap) -> None:
        """Different seeds should produce different datasets."""
        X1, _ = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=30, n_features=3, seed=42
        )
        X2, _ = generate_quantum_advantage_dataset(
            zz_3qubit, n_samples=30, n_features=3, seed=99
        )
        assert not np.allclose(X1, X2)


class TestGenerateQuantumAdvantageSplit:
    """Tests for the train/test split function."""

    def test_correct_split_sizes(self, zz_3qubit: ZZFeatureMap) -> None:
        """Train and test sizes should match requested."""
        X_train, X_test, y_train, y_test = generate_quantum_advantage_split(
            zz_3qubit, n_train=30, n_test=20, n_features=3, seed=42
        )
        assert X_train.shape == (30, 3)
        assert X_test.shape == (20, 3)
        assert y_train.shape == (30,)
        assert y_test.shape == (20,)

    def test_no_data_leakage(self, zz_3qubit: ZZFeatureMap) -> None:
        """Train and test data should not overlap."""
        X_train, X_test, _, _ = generate_quantum_advantage_split(
            zz_3qubit, n_train=30, n_test=20, n_features=3, seed=42
        )
        # Check no rows are exactly the same
        for i in range(len(X_test)):
            for j in range(len(X_train)):
                assert not np.allclose(X_test[i], X_train[j])

    def test_labels_binary(self, zz_3qubit: ZZFeatureMap) -> None:
        """All labels should be 0 or 1."""
        _, _, y_train, y_test = generate_quantum_advantage_split(
            zz_3qubit, n_train=30, n_test=20, n_features=3, seed=42
        )
        assert set(np.unique(np.concatenate([y_train, y_test]))).issubset({0.0, 1.0})
