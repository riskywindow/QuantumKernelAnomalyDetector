"""Tests for the quantum preprocessing pipeline."""

import numpy as np
import pytest

from src.data.transforms import QuantumPreprocessor


class TestQuantumPreprocessor:
    """Tests for QuantumPreprocessor."""

    def test_output_shape(self) -> None:
        """fit_transform should reduce dimensions to n_features."""
        X = np.random.default_rng(42).random((50, 20))
        preprocessor = QuantumPreprocessor(n_features=5)
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape == (50, 5)

    def test_output_range(self) -> None:
        """Transformed features should be in [0, 2*pi]."""
        X = np.random.default_rng(42).random((100, 10))
        preprocessor = QuantumPreprocessor(n_features=3)
        X_transformed = preprocessor.fit_transform(X)

        assert np.all(X_transformed >= 0.0)
        assert np.all(X_transformed <= 2 * np.pi + 1e-10)

    def test_fit_transform_consistency(self) -> None:
        """fit_transform on training data should match fit then transform."""
        rng = np.random.default_rng(42)
        X_train = rng.random((100, 10))

        pp1 = QuantumPreprocessor(n_features=4)
        result1 = pp1.fit_transform(X_train)

        pp2 = QuantumPreprocessor(n_features=4)
        pp2.fit(X_train)
        result2 = pp2.transform(X_train)

        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_transform_on_new_data(self) -> None:
        """transform on new data should produce correct shape and range."""
        rng = np.random.default_rng(42)
        X_train = rng.random((100, 10))
        X_test = rng.random((20, 10))

        preprocessor = QuantumPreprocessor(n_features=5)
        preprocessor.fit(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        assert X_test_transformed.shape == (20, 5)
        # Test data may extend slightly outside [0, 2pi] before clipping
        # but our implementation clips, so should be in range
        assert np.all(X_test_transformed >= 0.0)
        assert np.all(X_test_transformed <= 2 * np.pi + 1e-10)

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform before fit should raise RuntimeError."""
        preprocessor = QuantumPreprocessor(n_features=3)
        X = np.random.default_rng(42).random((10, 5))
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(X)

    def test_custom_angle_range(self) -> None:
        """Custom range_min and range_max should be respected."""
        X = np.random.default_rng(42).random((50, 10))
        preprocessor = QuantumPreprocessor(
            n_features=3, range_min=0.0, range_max=np.pi
        )
        X_transformed = preprocessor.fit_transform(X)

        assert np.all(X_transformed >= 0.0)
        assert np.all(X_transformed <= np.pi + 1e-10)

    def test_training_data_spans_full_range(self) -> None:
        """On training data, min/max per feature should be range endpoints."""
        X = np.random.default_rng(42).random((100, 10))
        preprocessor = QuantumPreprocessor(n_features=3)
        X_transformed = preprocessor.fit_transform(X)

        # Training data min-max should map to [0, 2*pi]
        for col in range(3):
            assert np.isclose(X_transformed[:, col].min(), 0.0, atol=1e-10)
            assert np.isclose(X_transformed[:, col].max(), 2 * np.pi, atol=1e-10)
