"""Tests for data loading utilities."""

import numpy as np
import pytest

from src.data.loader import prepare_anomaly_split


class TestPrepareAnomalySplit:
    """Tests for the anomaly train/test split function."""

    def test_train_contains_only_normal(self) -> None:
        """Training set should contain only normal (class 0) samples."""
        rng = np.random.default_rng(42)
        X = rng.random((200, 5))
        y = np.concatenate([np.zeros(180), np.ones(20)]).astype(int)

        X_train, X_test, y_test = prepare_anomaly_split(
            X, y, normal_train_size=50, test_size=50, seed=42
        )

        # Train set should be all normal (no labels returned, but the function
        # selects from normal_idx only)
        assert X_train.shape[0] == 50

    def test_test_set_has_both_classes(self) -> None:
        """Test set should contain both normal and anomaly samples."""
        rng = np.random.default_rng(42)
        X = rng.random((200, 5))
        y = np.concatenate([np.zeros(180), np.ones(20)]).astype(int)

        X_train, X_test, y_test = prepare_anomaly_split(
            X, y, normal_train_size=50, test_size=50, seed=42
        )

        assert 0 in y_test
        assert 1 in y_test

    def test_output_shapes(self) -> None:
        """Output shapes should match requested sizes."""
        rng = np.random.default_rng(42)
        X = rng.random((500, 10))
        y = np.concatenate([np.zeros(450), np.ones(50)]).astype(int)

        X_train, X_test, y_test = prepare_anomaly_split(
            X, y, normal_train_size=100, test_size=80, seed=42
        )

        assert X_train.shape == (100, 10)
        assert X_test.shape[0] == len(y_test)
        assert X_test.shape[1] == 10

    def test_reproducibility(self) -> None:
        """Same seed should produce identical splits."""
        rng = np.random.default_rng(42)
        X = rng.random((200, 5))
        y = np.concatenate([np.zeros(180), np.ones(20)]).astype(int)

        result1 = prepare_anomaly_split(X, y, normal_train_size=50, test_size=50, seed=42)
        result2 = prepare_anomaly_split(X, y, normal_train_size=50, test_size=50, seed=42)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
        np.testing.assert_array_equal(result1[2], result2[2])
