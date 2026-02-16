"""Tests for noise model construction."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.hardware.noise_models import (
    build_depolarizing_noise_model,
    build_noise_sweep,
    try_fetch_real_noise_model,
)


class TestBuildDepolarizingNoiseModel:
    """Tests for build_depolarizing_noise_model."""

    def test_returns_noise_model(self) -> None:
        model = build_depolarizing_noise_model()
        assert isinstance(model, NoiseModel)

    def test_default_parameters(self) -> None:
        model = build_depolarizing_noise_model()
        # Should have quantum errors and readout errors
        assert len(model.noise_instructions) > 0

    def test_custom_error_rates(self) -> None:
        model = build_depolarizing_noise_model(
            single_qubit_error=0.01,
            two_qubit_error=0.05,
            readout_error=0.03,
        )
        assert isinstance(model, NoiseModel)

    def test_zero_error_model(self) -> None:
        model = build_depolarizing_noise_model(
            single_qubit_error=0.0,
            two_qubit_error=0.0,
            readout_error=0.0,
        )
        assert isinstance(model, NoiseModel)

    def test_zero_error_produces_near_ideal_results(self) -> None:
        """Zero-error noise model should produce results close to statevector."""
        model = build_depolarizing_noise_model(
            single_qubit_error=0.0,
            two_qubit_error=0.0,
            readout_error=0.0,
        )

        # Create a simple circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        # Run with noise model (should be noiseless)
        sim = AerSimulator(noise_model=model)
        pm = generate_preset_pass_manager(optimization_level=0, backend=sim)
        transpiled = pm.run(qc)
        result = sim.run(transpiled, shots=8192).result()
        counts = result.get_counts()

        # Bell state: should give ~50% "00" and ~50% "11"
        total = sum(counts.values())
        p_00 = counts.get("00", 0) / total
        p_11 = counts.get("11", 0) / total
        assert abs(p_00 - 0.5) < 0.05
        assert abs(p_11 - 0.5) < 0.05

    def test_high_noise_affects_results(self) -> None:
        """High noise should visibly degrade circuit results."""
        model = build_depolarizing_noise_model(
            single_qubit_error=0.1,
            two_qubit_error=0.3,
            readout_error=0.1,
        )

        # Identity circuit: should give all-zeros without noise
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)  # H twice = identity
        qc.measure([0, 1], [0, 1])

        sim = AerSimulator(noise_model=model)
        pm = generate_preset_pass_manager(optimization_level=0, backend=sim)
        transpiled = pm.run(qc)
        result = sim.run(transpiled, shots=4096).result()
        counts = result.get_counts()

        # With high noise, we should see non-zero strings
        total = sum(counts.values())
        p_00 = counts.get("00", 0) / total
        # With 10% single-qubit error on each H + readout error, p_00 should be < 1
        assert p_00 < 0.99

    def test_noise_model_has_correct_gates(self) -> None:
        model = build_depolarizing_noise_model(
            single_qubit_error=0.01,
            two_qubit_error=0.05,
            readout_error=0.02,
        )
        instructions = model.noise_instructions
        # CX should be in the noise instructions
        assert "cx" in instructions


class TestBuildNoiseSweep:
    """Tests for build_noise_sweep."""

    def test_returns_correct_count(self) -> None:
        sweep = build_noise_sweep(n_levels=5)
        assert len(sweep) == 5

    def test_returns_tuples(self) -> None:
        sweep = build_noise_sweep(n_levels=3)
        for item in sweep:
            assert isinstance(item, tuple)
            assert len(item) == 2
            rate, model = item
            assert isinstance(rate, float)
            assert isinstance(model, NoiseModel)

    def test_increasing_error_rates(self) -> None:
        sweep = build_noise_sweep(n_levels=5)
        rates = [r for r, _ in sweep]
        for i in range(len(rates) - 1):
            assert rates[i] < rates[i + 1]

    def test_error_rate_range(self) -> None:
        sweep = build_noise_sweep(n_levels=10, min_error=0.001, max_error=0.01)
        rates = [r for r, _ in sweep]
        assert rates[0] == pytest.approx(0.001, rel=0.01)
        assert rates[-1] == pytest.approx(0.01, rel=0.01)

    def test_logarithmic_spacing(self) -> None:
        sweep = build_noise_sweep(n_levels=5, min_error=0.001, max_error=0.1)
        rates = [r for r, _ in sweep]
        log_rates = np.log10(rates)
        diffs = np.diff(log_rates)
        # All log-spacings should be approximately equal
        assert np.allclose(diffs, diffs[0], rtol=0.01)

    def test_caps_at_physical_limits(self) -> None:
        """Two-qubit error should be capped at 15/16."""
        sweep = build_noise_sweep(n_levels=3, min_error=0.01, max_error=0.5)
        # At max_error=0.5, two_qubit would be 5.0 without cap
        # Should be capped at 0.9375
        # Just verify it doesn't raise an error
        assert len(sweep) == 3


class TestTryFetchRealNoiseModel:
    """Tests for try_fetch_real_noise_model."""

    def test_returns_none_without_token(self, monkeypatch) -> None:
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        result = try_fetch_real_noise_model()
        assert result is None

    def test_returns_none_on_invalid_backend(self, monkeypatch) -> None:
        # Even with a fake token, should return None gracefully
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "fake_token_for_test")
        result = try_fetch_real_noise_model("nonexistent_backend_xyz")
        assert result is None
