"""Tests for QuantumKernel with noise model support."""

from __future__ import annotations

import numpy as np
import pytest

from src.hardware.noise_models import build_depolarizing_noise_model
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


class TestQuantumKernelNoisy:
    """Tests for QuantumKernel with noise_model parameter."""

    def test_accepts_noise_model(self) -> None:
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        noise = build_depolarizing_noise_model(0.01, 0.05, 0.02)
        qk = QuantumKernel(fm, backend="sampler", noise_model=noise)
        assert qk.noise_model is not None

    def test_none_noise_model_default(self) -> None:
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        qk = QuantumKernel(fm, backend="sampler")
        assert qk.noise_model is None

    def test_noisy_kernel_entry(self) -> None:
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        noise = build_depolarizing_noise_model(0.01, 0.05, 0.02)
        qk = QuantumKernel(fm, backend="sampler", n_shots=2048, noise_model=noise)

        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.5, 2.5])
        k_val = qk.compute_entry(x1, x2)

        assert 0 <= k_val <= 1

    def test_noisy_kernel_matrix(self) -> None:
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        noise = build_depolarizing_noise_model(0.01, 0.05, 0.02)
        qk = QuantumKernel(fm, backend="sampler", n_shots=1024, noise_model=noise)

        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi
        K = qk.compute_matrix(X, show_progress=False)

        assert K.shape == (3, 3)
        assert np.allclose(K, K.T)
        assert np.all(K >= 0)
        assert np.all(K <= 1)

    def test_noisy_differs_from_ideal(self) -> None:
        """Noisy kernel matrix should differ from statevector."""
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((4, 2)) * 2 * np.pi

        # Ideal
        qk_ideal = QuantumKernel(fm, backend="statevector")
        K_ideal = qk_ideal.compute_matrix(X, show_progress=False)

        # Noisy (high noise to ensure difference)
        noise = build_depolarizing_noise_model(0.05, 0.2, 0.05)
        qk_noisy = QuantumKernel(fm, backend="sampler", n_shots=2048, noise_model=noise)
        K_noisy = qk_noisy.compute_matrix(X, show_progress=False)

        # They should differ
        diff = np.abs(K_ideal - K_noisy)
        assert np.mean(diff) > 0.01

    def test_higher_noise_larger_deviation(self) -> None:
        """Higher noise should produce larger deviation from ideal."""
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        qk_ideal = QuantumKernel(fm, backend="statevector")
        K_ideal = qk_ideal.compute_matrix(X, show_progress=False)

        # Low noise
        noise_low = build_depolarizing_noise_model(0.001, 0.005, 0.002)
        qk_low = QuantumKernel(fm, backend="sampler", n_shots=4096, noise_model=noise_low)
        K_low = qk_low.compute_matrix(X, show_progress=False)

        # High noise
        noise_high = build_depolarizing_noise_model(0.05, 0.3, 0.1)
        qk_high = QuantumKernel(fm, backend="sampler", n_shots=4096, noise_model=noise_high)
        K_high = qk_high.compute_matrix(X, show_progress=False)

        error_low = np.mean(np.abs(K_ideal - K_low))
        error_high = np.mean(np.abs(K_ideal - K_high))

        assert error_high > error_low

    def test_name_includes_noisy(self) -> None:
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        noise = build_depolarizing_noise_model(0.01, 0.05, 0.02)
        qk = QuantumKernel(fm, backend="sampler", noise_model=noise)
        assert "noisy" in qk.name

    def test_name_without_noise(self) -> None:
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        qk = QuantumKernel(fm, backend="sampler")
        assert "noisy" not in qk.name

    def test_statevector_ignores_noise(self) -> None:
        """Statevector backend should work even if noise_model is provided."""
        fm = ZZFeatureMap(n_qubits=2, reps=1)
        noise = build_depolarizing_noise_model(0.05, 0.2, 0.05)
        qk = QuantumKernel(fm, backend="statevector", noise_model=noise)

        x = np.array([1.0, 2.0])
        k_val = qk.compute_entry(x, x)
        # Statevector always gives exact 1.0 for K(x,x)
        assert k_val == pytest.approx(1.0, abs=1e-10)
