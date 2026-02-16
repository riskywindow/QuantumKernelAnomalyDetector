"""Tests for shot-based (sampler) quantum kernel estimation."""

import numpy as np
import pytest

from src.kernels.base import validate_kernel_matrix
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


class TestQuantumKernelSampler:
    """Tests for the sampler backend in QuantumKernel."""

    @pytest.fixture
    def kernel_sampler(self) -> QuantumKernel:
        """2-qubit quantum kernel with sampler backend."""
        fm = ZZFeatureMap(n_qubits=2, reps=2, entanglement="linear")
        return QuantumKernel(fm, backend="sampler", n_shots=4096)

    def test_self_kernel_approximately_one(
        self, kernel_sampler: QuantumKernel
    ) -> None:
        """K(x, x) should be approximately 1.0 with shot noise."""
        x = np.array([1.5, 0.7])
        k = kernel_sampler.compute_entry(x, x)
        # With 4096 shots, should be very close to 1.0
        assert np.isclose(k, 1.0, atol=0.05), f"K(x,x) = {k}, expected ~1.0"

    def test_different_points_bounded(
        self, kernel_sampler: QuantumKernel
    ) -> None:
        """K(x1, x2) should be in [0, 1] for different points."""
        rng = np.random.default_rng(42)
        x1 = rng.random(2) * 2 * np.pi
        x2 = rng.random(2) * 2 * np.pi
        k = kernel_sampler.compute_entry(x1, x2)
        assert 0.0 <= k <= 1.0

    def test_matrix_approximately_symmetric(
        self, kernel_sampler: QuantumKernel
    ) -> None:
        """Kernel matrix should be approximately symmetric."""
        rng = np.random.default_rng(42)
        X = rng.random((4, 2)) * 2 * np.pi
        K = kernel_sampler.compute_matrix(X, show_progress=False)

        # Diagonal should be 1.0 (set explicitly by compute_matrix)
        np.testing.assert_allclose(np.diag(K), 1.0)
        # Off-diagonal should be approximately symmetric
        np.testing.assert_allclose(K, K.T, atol=0.0)  # exact due to symmetry exploitation

    def test_sampler_close_to_statevector(self) -> None:
        """Sampler results should be close to exact statevector results."""
        fm = ZZFeatureMap(n_qubits=2, reps=1, entanglement="linear")

        qk_exact = QuantumKernel(fm, backend="statevector")
        qk_sampler = QuantumKernel(fm, backend="sampler", n_shots=8192)

        rng = np.random.default_rng(42)
        x1 = rng.random(2) * 2 * np.pi
        x2 = rng.random(2) * 2 * np.pi

        k_exact = qk_exact.compute_entry(x1, x2)
        k_sampler = qk_sampler.compute_entry(x1, x2)

        # With 8192 shots, should be within ~0.05 of exact value
        assert abs(k_exact - k_sampler) < 0.1, \
            f"Sampler {k_sampler:.4f} too far from exact {k_exact:.4f}"

    def test_more_shots_reduces_variance(self) -> None:
        """Higher shot counts should produce more consistent estimates."""
        fm = ZZFeatureMap(n_qubits=2, reps=1)

        x1 = np.array([1.5, 0.7])
        x2 = np.array([0.3, 2.1])

        # Collect multiple estimates at low and high shot counts
        low_shots = []
        high_shots = []
        for seed in range(10):
            np.random.seed(seed)
            qk_low = QuantumKernel(fm, backend="sampler", n_shots=100)
            low_shots.append(qk_low.compute_entry(x1, x2))

            qk_high = QuantumKernel(fm, backend="sampler", n_shots=8192)
            high_shots.append(qk_high.compute_entry(x1, x2))

        std_low = np.std(low_shots)
        std_high = np.std(high_shots)

        # High shots should have lower variance than low shots
        assert std_high < std_low, \
            f"Expected std_high ({std_high:.4f}) < std_low ({std_low:.4f})"

    def test_name_includes_shots(self) -> None:
        """Sampler kernel name should include shot count."""
        fm = ZZFeatureMap(n_qubits=2)
        qk = QuantumKernel(fm, backend="sampler", n_shots=2048)
        assert "sampler" in qk.name
        assert "2048" in qk.name

    def test_statevector_name_no_shots(self) -> None:
        """Statevector kernel name should NOT include shot count."""
        fm = ZZFeatureMap(n_qubits=2)
        qk = QuantumKernel(fm, backend="statevector")
        assert "shots" not in qk.name
