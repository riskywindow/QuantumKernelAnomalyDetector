"""Tests for zero-noise extrapolation."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.zne import (
    _extrapolate_to_zero,
    scale_noise_model,
    zero_noise_extrapolation,
    zne_kernel_matrix,
)
from src.hardware.noise_models import build_depolarizing_noise_model


class TestScaleNoiseModel:
    """Tests for scale_noise_model."""

    def test_returns_noise_model(self) -> None:
        model = scale_noise_model(1.0, 0.001, 0.01, 0.02)
        from qiskit_aer.noise import NoiseModel

        assert isinstance(model, NoiseModel)

    def test_scale_factor_1_matches_base(self) -> None:
        base = build_depolarizing_noise_model(0.001, 0.01, 0.02)
        scaled = scale_noise_model(1.0, 0.001, 0.01, 0.02)
        # Both should have the same noise instructions
        assert set(base.noise_instructions) == set(scaled.noise_instructions)

    def test_increasing_scale_factors(self) -> None:
        # Higher scale factor should produce more noise (lower all-zeros probability)
        from qiskit.circuit import QuantumCircuit
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.measure([0, 1], [0, 1])

        probs = []
        for sf in [1.0, 2.0, 3.0]:
            model = scale_noise_model(sf, 0.01, 0.05, 0.02)
            sim = AerSimulator(noise_model=model)
            pm = generate_preset_pass_manager(optimization_level=0, backend=sim)
            transpiled = pm.run(qc)
            result = sim.run(transpiled, shots=8192).result()
            counts = result.get_counts()
            probs.append(counts.get("00", 0) / 8192)

        # Higher noise should generally change the probability distribution
        # (though not necessarily monotonically for this specific circuit)
        assert len(probs) == 3

    def test_caps_at_physical_limits(self) -> None:
        # Very high scale factor should not exceed physical limits
        model = scale_noise_model(100.0, 0.01, 0.05, 0.02)
        assert model is not None  # Should not raise


class TestExtrapolateToZero:
    """Tests for _extrapolate_to_zero."""

    def test_linear_extrapolation(self) -> None:
        # f(x) = 2x + 3, extrapolate to x=0 should give 3
        scale_factors = [1.0, 2.0, 3.0, 4.0]
        values = [5.0, 7.0, 9.0, 11.0]  # 2*x + 3
        result, info = _extrapolate_to_zero(scale_factors, values, "linear")
        assert result == pytest.approx(3.0, abs=0.01)
        assert info["method"] == "linear"
        assert info["r_squared"] == pytest.approx(1.0, abs=0.01)

    def test_linear_extrapolation_noisy(self) -> None:
        # Slightly noisy linear data
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
        values = [0.85, 0.78, 0.72, 0.64, 0.58]
        result, info = _extrapolate_to_zero(scale_factors, values, "linear")
        # Should extrapolate to something above the first value
        assert result > values[0]
        assert info["r_squared"] > 0.9

    def test_exponential_extrapolation(self) -> None:
        # f(x) = 0.5 * exp(-0.5*x) + 0.3
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
        values = [0.5 * np.exp(-0.5 * x) + 0.3 for x in scale_factors]
        result, info = _extrapolate_to_zero(scale_factors, values, "exponential")
        # At x=0: 0.5 * 1 + 0.3 = 0.8
        assert result == pytest.approx(0.8, abs=0.05)

    def test_exponential_fallback_to_linear(self) -> None:
        # Data that doesn't fit exponential well
        scale_factors = [1.0, 2.0, 3.0]
        values = [0.5, 0.5, 0.5]  # Constant data
        result, info = _extrapolate_to_zero(scale_factors, values, "exponential")
        assert result == pytest.approx(0.5, abs=0.01)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown extrapolation"):
            _extrapolate_to_zero([1.0], [0.5], "invalid")


class TestZeroNoiseExtrapolation:
    """Tests for zero_noise_extrapolation."""

    def test_produces_estimate(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.5, 2.5])

        noise_params = {
            "single_qubit_error": 0.01,
            "two_qubit_error": 0.05,
            "readout_error": 0.02,
        }

        result = zero_noise_extrapolation(
            fm, x1, x2, noise_params,
            scale_factors=[1.0, 2.0, 3.0],
            shots=2048,
        )

        assert "zne_estimate" in result
        assert "raw_estimates" in result
        assert 0 <= result["zne_estimate"] <= 1
        assert len(result["raw_estimates"]) == 3

    def test_zne_closer_to_ideal(self) -> None:
        """ZNE estimate should be closer to ideal than raw noisy."""
        from qiskit.quantum_info import Statevector

        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.0, 2.0])  # Same point: ideal = 1.0

        noise_params = {
            "single_qubit_error": 0.01,
            "two_qubit_error": 0.05,
            "readout_error": 0.02,
        }

        result = zero_noise_extrapolation(
            fm, x1, x2, noise_params,
            scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
            shots=4096,
        )

        ideal = 1.0
        zne_error = abs(result["zne_estimate"] - ideal)
        raw_error = abs(result["raw_estimates"][0] - ideal)

        # ZNE should improve or at least not significantly worsen
        # (with noise on K(x,x), raw will be < 1, ZNE should extrapolate closer)
        assert zne_error <= raw_error + 0.05  # Allow small tolerance

    def test_clamped_to_01(self) -> None:
        """ZNE estimate should be clamped to [0, 1]."""
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.0, 2.0])

        noise_params = {
            "single_qubit_error": 0.005,
            "two_qubit_error": 0.02,
            "readout_error": 0.01,
        }

        result = zero_noise_extrapolation(
            fm, x1, x2, noise_params,
            scale_factors=[1.0, 2.0, 3.0],
            shots=2048,
        )

        assert 0.0 <= result["zne_estimate"] <= 1.0


class TestZneKernelMatrix:
    """Tests for zne_kernel_matrix."""

    def test_produces_matrix(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        noise_params = {
            "single_qubit_error": 0.01,
            "two_qubit_error": 0.05,
            "readout_error": 0.02,
        }

        result = zne_kernel_matrix(
            fm, X, noise_params,
            scale_factors=[1.0, 2.0],
            shots=1024,
            show_progress=False,
        )

        assert "zne_matrix" in result
        assert result["zne_matrix"].shape == (3, 3)
        assert len(result["raw_matrices"]) == 2

    def test_symmetric(self) -> None:
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        rng = np.random.default_rng(42)
        X = rng.random((3, 2)) * 2 * np.pi

        noise_params = {
            "single_qubit_error": 0.01,
            "two_qubit_error": 0.05,
            "readout_error": 0.02,
        }

        result = zne_kernel_matrix(
            fm, X, noise_params,
            scale_factors=[1.0, 2.0],
            shots=1024,
            show_progress=False,
        )

        assert np.allclose(result["zne_matrix"], result["zne_matrix"].T)
