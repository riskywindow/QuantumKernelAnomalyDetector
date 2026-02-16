"""Tests for classical kernel baselines (RBF and Polynomial)."""

import numpy as np
import pytest

from src.kernels.base import validate_kernel_matrix
from src.kernels.classical import PolynomialKernel, RBFKernel


class TestRBFKernel:
    """Tests for the RBF (Gaussian) kernel."""

    def test_self_kernel_is_one(self) -> None:
        """RBF K(x, x) should always be 1.0."""
        rbf = RBFKernel(gamma=0.5)
        x = np.array([1.0, 2.0, 3.0])
        k = rbf.compute_entry(x, x)
        assert np.isclose(k, 1.0, atol=1e-12)

    def test_matrix_diagonal_is_one(self) -> None:
        """RBF kernel matrix should have unit diagonal."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        rbf = RBFKernel(gamma=0.5)
        K = rbf.compute_matrix(X)
        np.testing.assert_allclose(np.diag(K), 1.0)

    def test_matrix_symmetric(self) -> None:
        """RBF kernel matrix should be symmetric."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        rbf = RBFKernel(gamma=0.5)
        K = rbf.compute_matrix(X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_matrix_psd(self) -> None:
        """RBF kernel matrix should be positive semi-definite."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        rbf = RBFKernel(gamma=0.5)
        K = rbf.compute_matrix(X)
        results = validate_kernel_matrix(K)
        assert results["positive_semidefinite"]

    def test_matrix_bounded_0_1(self) -> None:
        """RBF kernel values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        rbf = RBFKernel(gamma=0.5)
        K = rbf.compute_matrix(X)
        assert np.all(K >= 0.0) and np.all(K <= 1.0)

    def test_gamma_scale_works(self) -> None:
        """gamma='scale' should work without error."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        rbf = RBFKernel(gamma="scale")
        K = rbf.compute_matrix(X)
        assert K.shape == (10, 10)

    def test_gamma_scale_adapts_to_data(self) -> None:
        """gamma='scale' should produce different results for different data."""
        rng = np.random.default_rng(42)
        X_narrow = rng.random((10, 5)) * 0.01
        X_wide = rng.random((10, 5)) * 100.0

        rbf = RBFKernel(gamma="scale")
        K_narrow = rbf.compute_matrix(X_narrow)
        K_wide = rbf.compute_matrix(X_wide)

        # Both should be valid but different
        assert K_narrow.shape == K_wide.shape
        # They shouldn't be identical since gamma adapts
        assert not np.allclose(K_narrow, K_wide)

    def test_compute_entry_consistent_with_matrix(self) -> None:
        """compute_entry should match corresponding compute_matrix element."""
        rng = np.random.default_rng(42)
        X = rng.random((5, 3))
        rbf = RBFKernel(gamma=0.5)

        K = rbf.compute_matrix(X)
        k_entry = rbf.compute_entry(X[1], X[3])
        assert np.isclose(K[1, 3], k_entry, atol=1e-10)

    def test_non_symmetric_matrix(self) -> None:
        """K(X1, X2) should have correct shape."""
        rng = np.random.default_rng(42)
        X1 = rng.random((3, 5))
        X2 = rng.random((4, 5))
        rbf = RBFKernel(gamma=0.5)
        K = rbf.compute_matrix(X1, X2)
        assert K.shape == (3, 4)

    def test_name_property(self) -> None:
        """name should be descriptive."""
        rbf_scale = RBFKernel(gamma="scale")
        assert "RBF" in rbf_scale.name
        assert "scale" in rbf_scale.name

        rbf_fixed = RBFKernel(gamma=0.5)
        assert "RBF" in rbf_fixed.name
        assert "0.5" in rbf_fixed.name


class TestPolynomialKernel:
    """Tests for the Polynomial kernel."""

    def test_known_computation(self) -> None:
        """Test polynomial kernel for known input values."""
        poly = PolynomialKernel(degree=2, gamma=1.0, coef0=0.0)
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])
        # (1.0 * 0 + 0.0)^2 = 0.0
        k = poly.compute_entry(x1, x2)
        assert np.isclose(k, 0.0)

        # Same point: (1.0 * 1 + 0.0)^2 = 1.0
        k_self = poly.compute_entry(x1, x1)
        assert np.isclose(k_self, 1.0)

    def test_matrix_symmetric(self) -> None:
        """Polynomial kernel matrix should be symmetric."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        poly = PolynomialKernel(degree=3, gamma="scale")
        K = poly.compute_matrix(X)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_matrix_psd(self) -> None:
        """Polynomial kernel matrix should be PSD."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        poly = PolynomialKernel(degree=2, gamma="scale", coef0=1.0)
        K = poly.compute_matrix(X)
        results = validate_kernel_matrix(K)
        assert results["symmetric"]
        assert results["positive_semidefinite"]

    def test_gamma_scale_works(self) -> None:
        """gamma='scale' should work without error."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        poly = PolynomialKernel(degree=3, gamma="scale")
        K = poly.compute_matrix(X)
        assert K.shape == (10, 10)

    def test_compute_entry_consistent_with_matrix(self) -> None:
        """compute_entry should match corresponding compute_matrix element."""
        rng = np.random.default_rng(42)
        X = rng.random((5, 3))
        poly = PolynomialKernel(degree=2, gamma=1.0, coef0=1.0)

        K = poly.compute_matrix(X)
        k_entry = poly.compute_entry(X[0], X[2])
        assert np.isclose(K[0, 2], k_entry, atol=1e-10)

    def test_non_symmetric_matrix(self) -> None:
        """K(X1, X2) should have correct shape."""
        rng = np.random.default_rng(42)
        X1 = rng.random((3, 5))
        X2 = rng.random((4, 5))
        poly = PolynomialKernel(degree=2, gamma=1.0)
        K = poly.compute_matrix(X1, X2)
        assert K.shape == (3, 4)

    def test_name_property(self) -> None:
        """name should be descriptive."""
        poly = PolynomialKernel(degree=3, gamma="scale")
        assert "Polynomial" in poly.name
        assert "d=3" in poly.name
        assert "scale" in poly.name

    def test_degree_affects_result(self) -> None:
        """Different degrees should produce different results."""
        rng = np.random.default_rng(42)
        X = rng.random((5, 3))
        K2 = PolynomialKernel(degree=2, gamma=1.0).compute_matrix(X)
        K3 = PolynomialKernel(degree=3, gamma=1.0).compute_matrix(X)
        assert not np.allclose(K2, K3)
