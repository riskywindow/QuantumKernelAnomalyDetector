"""Tests for the quantum kernel implementation."""

import numpy as np
import pytest

from src.kernels.base import validate_kernel_matrix
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


class TestQuantumKernel:
    """Tests for QuantumKernel."""

    @pytest.fixture
    def kernel_2q(self) -> QuantumKernel:
        """2-qubit quantum kernel with default settings."""
        fm = ZZFeatureMap(n_qubits=2, reps=2, entanglement="linear")
        return QuantumKernel(fm, backend="statevector")

    def test_self_kernel_is_one(self, kernel_2q: QuantumKernel) -> None:
        """K(x, x) should be exactly 1.0 for any data point."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.random(2) * 2 * np.pi
            k = kernel_2q.compute_entry(x, x)
            assert np.isclose(k, 1.0, atol=1e-12), f"K(x,x) = {k}, expected 1.0"

    def test_different_points_less_than_one(self, kernel_2q: QuantumKernel) -> None:
        """K(x1, x2) should be in [0, 1) for different data points."""
        rng = np.random.default_rng(42)
        x1 = rng.random(2) * 2 * np.pi
        x2 = rng.random(2) * 2 * np.pi
        k = kernel_2q.compute_entry(x1, x2)
        assert 0.0 <= k <= 1.0, f"K(x1, x2) = {k}, should be in [0, 1]"

    def test_kernel_is_symmetric(self, kernel_2q: QuantumKernel) -> None:
        """K(x1, x2) should equal K(x2, x1)."""
        rng = np.random.default_rng(42)
        x1 = rng.random(2) * 2 * np.pi
        x2 = rng.random(2) * 2 * np.pi
        k12 = kernel_2q.compute_entry(x1, x2)
        k21 = kernel_2q.compute_entry(x2, x1)
        assert np.isclose(k12, k21, atol=1e-12)

    def test_matrix_5x5_all_properties(self, kernel_2q: QuantumKernel) -> None:
        """5x5 kernel matrix should pass all validation checks."""
        rng = np.random.default_rng(42)
        X = rng.random((5, 2)) * 2 * np.pi
        K = kernel_2q.compute_matrix(X, show_progress=False)

        assert K.shape == (5, 5)
        results = validate_kernel_matrix(K)
        assert results["symmetric"], "Kernel matrix is not symmetric"
        assert results["positive_semidefinite"], "Kernel matrix is not PSD"
        assert results["unit_diagonal"], "Kernel matrix diagonal is not 1"
        assert results["bounded_0_1"], "Kernel matrix values not in [0,1]"

    def test_matrix_symmetric_mode(self, kernel_2q: QuantumKernel) -> None:
        """K(X, None) should produce a symmetric matrix."""
        rng = np.random.default_rng(42)
        X = rng.random((4, 2)) * 2 * np.pi
        K = kernel_2q.compute_matrix(X, show_progress=False)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_matrix_non_symmetric_mode(self, kernel_2q: QuantumKernel) -> None:
        """K(X1, X2) for different datasets should have correct shape."""
        rng = np.random.default_rng(42)
        X1 = rng.random((3, 2)) * 2 * np.pi
        X2 = rng.random((4, 2)) * 2 * np.pi
        K = kernel_2q.compute_matrix(X1, X2, show_progress=False)
        assert K.shape == (3, 4)

    def test_3qubit_kernel(self) -> None:
        """3-qubit kernel should also produce valid matrices."""
        fm = ZZFeatureMap(n_qubits=3, reps=1, entanglement="linear")
        qk = QuantumKernel(fm, backend="statevector")

        rng = np.random.default_rng(42)
        X = rng.random((4, 3)) * 2 * np.pi
        K = qk.compute_matrix(X, show_progress=False)

        results = validate_kernel_matrix(K)
        assert results["symmetric"]
        assert results["positive_semidefinite"]
        assert results["unit_diagonal"]
        assert results["bounded_0_1"]

    def test_invalid_backend(self) -> None:
        """Invalid backend should raise ValueError."""
        fm = ZZFeatureMap(n_qubits=2)
        with pytest.raises(ValueError, match="Unsupported backend"):
            QuantumKernel(fm, backend="invalid")

    def test_name_property(self, kernel_2q: QuantumKernel) -> None:
        """name property should return a descriptive string."""
        assert "QuantumKernel" in kernel_2q.name
        assert "2q" in kernel_2q.name
        assert "statevector" in kernel_2q.name
