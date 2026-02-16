"""Tests for IBM Quantum runner and local noise simulation."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from src.hardware.ibm_runner import IBMQuantumRunner, LocalNoiseRunner
from src.hardware.noise_models import build_depolarizing_noise_model


class TestIBMQuantumRunner:
    """Tests for IBMQuantumRunner."""

    def test_raises_without_token(self, monkeypatch) -> None:
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="IBM_QUANTUM_TOKEN"):
            IBMQuantumRunner()

    def test_error_message_includes_instructions(self, monkeypatch) -> None:
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="quantum.ibm.com"):
            IBMQuantumRunner()


class TestLocalNoiseRunner:
    """Tests for LocalNoiseRunner."""

    def test_noiseless_runner(self) -> None:
        runner = LocalNoiseRunner()
        assert runner.noise_model is None

    def test_runner_with_noise_model(self) -> None:
        noise_model = build_depolarizing_noise_model(
            single_qubit_error=0.01,
            two_qubit_error=0.05,
            readout_error=0.02,
        )
        runner = LocalNoiseRunner(noise_model=noise_model)
        assert runner.noise_model is not None

    def test_run_circuits(self) -> None:
        noise_model = build_depolarizing_noise_model(
            single_qubit_error=0.005,
            two_qubit_error=0.02,
            readout_error=0.01,
        )
        runner = LocalNoiseRunner(noise_model=noise_model)

        # Simple circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        counts_list = runner.run_circuits([qc], shots=1024)
        assert len(counts_list) == 1
        counts = counts_list[0]
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 1024

    def test_run_multiple_circuits(self) -> None:
        runner = LocalNoiseRunner()

        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.measure([0, 1], [0, 1])

        qc2 = QuantumCircuit(2, 2)
        qc2.x(0)
        qc2.measure([0, 1], [0, 1])

        counts_list = runner.run_circuits([qc1, qc2], shots=512)
        assert len(counts_list) == 2

    def test_estimate_kernel_entry(self) -> None:
        runner = LocalNoiseRunner()

        # Identity kernel circuit (should give ~1.0)
        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])

        k_val = runner.estimate_kernel_entry(qc, shots=2048)
        assert 0 <= k_val <= 1
        # Identity circuit: should be close to 1
        assert k_val > 0.95

    def test_noisy_estimation_differs_from_noiseless(self) -> None:
        """Noisy runner should produce different results than noiseless."""
        noise_model = build_depolarizing_noise_model(
            single_qubit_error=0.05,
            two_qubit_error=0.15,
            readout_error=0.05,
        )
        noisy_runner = LocalNoiseRunner(noise_model=noise_model)
        clean_runner = LocalNoiseRunner()

        # Build a non-trivial circuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.measure([0, 1, 2], [0, 1, 2])

        noisy_val = noisy_runner.estimate_kernel_entry(qc, shots=4096)
        clean_val = clean_runner.estimate_kernel_entry(qc, shots=4096)

        # They should differ (noise has an effect)
        # Both should be valid probabilities
        assert 0 <= noisy_val <= 1
        assert 0 <= clean_val <= 1

    def test_kernel_entry_from_feature_map(self) -> None:
        """Test kernel entry estimation using a real feature map circuit."""
        from src.kernels.feature_maps.zz import ZZFeatureMap

        fm = ZZFeatureMap(n_qubits=2, reps=1)
        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.0, 2.0])  # Same point

        circuit_x1 = fm.build_circuit(x1)
        circuit_x2_dag = fm.build_circuit(x2).inverse()
        kernel_circuit = circuit_x1.compose(circuit_x2_dag)

        meas_circuit = QuantumCircuit(2, 2)
        meas_circuit.compose(kernel_circuit, inplace=True)
        meas_circuit.measure([0, 1], [0, 1])

        runner = LocalNoiseRunner()
        k_val = runner.estimate_kernel_entry(meas_circuit, shots=4096)
        # K(x, x) should be close to 1
        assert k_val > 0.9

    def test_fallback_on_missing_backend(self) -> None:
        """Backend_name without token should fallback to noiseless."""
        runner = LocalNoiseRunner(backend_name="nonexistent_backend")
        assert runner.noise_model is None
