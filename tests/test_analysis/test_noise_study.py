"""Tests for noise impact study."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.noise import (
    compute_kernel_fidelity,
    compute_noisy_kernel_matrix,
    run_noise_sweep,
)
from src.hardware.noise_models import build_depolarizing_noise_model, build_noise_sweep


class TestComputeKernelFidelity:
    """Tests for compute_kernel_fidelity."""

    def test_identical_matrices(self) -> None:
        K = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
        result = compute_kernel_fidelity(K, K)
        assert result["frobenius_error"] == pytest.approx(0.0, abs=1e-10)
        assert result["mean_abs_error"] == pytest.approx(0.0, abs=1e-10)
        assert result["max_abs_error"] == pytest.approx(0.0, abs=1e-10)
        assert result["correlation"] == pytest.approx(1.0, abs=1e-6)
        assert result["is_psd"] is True

    def test_different_matrices(self) -> None:
        K_ideal = np.array([[1.0, 0.5], [0.5, 1.0]])
        K_noisy = np.array([[0.9, 0.4], [0.4, 0.9]])
        result = compute_kernel_fidelity(K_noisy, K_ideal)
        assert result["frobenius_error"] > 0
        assert result["mean_abs_error"] > 0
        assert result["max_abs_error"] > 0

    def test_psd_detection(self) -> None:
        K_ideal = np.eye(3)
        K_noisy = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = compute_kernel_fidelity(K_noisy, K_ideal)
        assert result["is_psd"] is False

    def test_correlation_with_noise(self) -> None:
        rng = np.random.default_rng(42)
        K_ideal = np.eye(5) + 0.3 * rng.random((5, 5))
        K_ideal = (K_ideal + K_ideal.T) / 2
        K_noisy = K_ideal + 0.01 * rng.standard_normal((5, 5))
        K_noisy = (K_noisy + K_noisy.T) / 2

        result = compute_kernel_fidelity(K_noisy, K_ideal)
        # Small noise should give high correlation
        assert result["correlation"] > 0.9


class TestComputeNoisyKernelMatrix:
    """Tests for compute_noisy_kernel_matrix."""

    def test_produces_correct_shape(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        noise_model = build_depolarizing_noise_model(0.001, 0.01, 0.02)
        K = compute_noisy_kernel_matrix(fm, X, noise_model, shots=512)

        assert K.shape == (3, 3)

    def test_symmetric(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        noise_model = build_depolarizing_noise_model(0.001, 0.01, 0.02)
        K = compute_noisy_kernel_matrix(fm, X, noise_model, shots=512)

        assert np.allclose(K, K.T)

    def test_values_in_range(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        noise_model = build_depolarizing_noise_model(0.001, 0.01, 0.02)
        K = compute_noisy_kernel_matrix(fm, X, noise_model, shots=1024)

        assert np.all(K >= 0)
        assert np.all(K <= 1)


class TestRunNoiseSweep:
    """Tests for run_noise_sweep."""

    def test_completes_without_error(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        # Compute ideal kernel
        from src.kernels.quantum import QuantumKernel

        qk = QuantumKernel(fm, backend="statevector")
        ideal_K = qk.compute_matrix(X, show_progress=False)

        # Small sweep
        noise_models = build_noise_sweep(n_levels=2, min_error=0.001, max_error=0.01)

        result = run_noise_sweep(
            fm, X, noise_models, ideal_K, shots=512, show_progress=False
        )

        assert len(result["error_rates"]) == 2
        assert len(result["kernel_matrices"]) == 2
        assert len(result["fidelities"]) == 2
        assert len(result["mean_abs_errors"]) == 2

    def test_increasing_noise_increases_error(self) -> None:
        """Higher noise should generally lead to larger fidelity error."""
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        from src.kernels.quantum import QuantumKernel

        qk = QuantumKernel(fm, backend="statevector")
        ideal_K = qk.compute_matrix(X, show_progress=False)

        # Two noise levels: low and high
        noise_models = [
            (0.001, build_depolarizing_noise_model(0.001, 0.01, 0.002)),
            (0.05, build_depolarizing_noise_model(0.05, 0.3, 0.1)),
        ]

        result = run_noise_sweep(
            fm, X, noise_models, ideal_K, shots=2048, show_progress=False
        )

        # Higher noise should give larger error (on average)
        assert result["fidelities"][1] > result["fidelities"][0]
