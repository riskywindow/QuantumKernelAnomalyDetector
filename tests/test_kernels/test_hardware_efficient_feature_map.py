"""Tests for the Hardware-Efficient feature map implementation."""

import numpy as np
import pytest

from src.kernels.feature_maps.hardware_efficient import HardwareEfficientFeatureMap


class TestHardwareEfficientFeatureMap:
    """Tests for Hardware-Efficient feature map circuit construction."""

    def test_circuit_num_qubits(self) -> None:
        """Circuit should have the correct number of qubits."""
        fm = HardwareEfficientFeatureMap(n_qubits=3, reps=1)
        x = np.array([1.0, 2.0, 3.0])
        qc = fm.build_circuit(x)
        assert qc.num_qubits == 3

    def test_only_native_gates(self) -> None:
        """Circuit should use ONLY RZ, SX, and CX gates."""
        fm = HardwareEfficientFeatureMap(n_qubits=4, reps=2)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        qc = fm.build_circuit(x)

        allowed_gates = {"rz", "sx", "cx"}
        actual_gates = set(qc.count_ops().keys())
        assert actual_gates.issubset(allowed_gates), \
            f"Non-native gates found: {actual_gates - allowed_gates}"

    def test_gate_counts_3q_1rep(self) -> None:
        """Check gate counts for 3 qubits, 1 rep."""
        fm = HardwareEfficientFeatureMap(n_qubits=3, reps=1)
        x = np.array([0.5, 1.0, 1.5])
        qc = fm.build_circuit(x)

        ops = dict(qc.count_ops())
        # Per rep: 3*(RZ+SX+RZ) + 2 CX = 6 RZ + 3 SX + 2 CX
        assert ops.get("rz", 0) == 6
        assert ops.get("sx", 0) == 3
        assert ops.get("cx", 0) == 2

    def test_linear_nearest_neighbor_connectivity(self) -> None:
        """CX gates should only connect nearest neighbors."""
        fm = HardwareEfficientFeatureMap(n_qubits=4, reps=1)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        qc = fm.build_circuit(x)

        # Check all CX gates have adjacent qubit pairs
        for instruction in qc.data:
            if instruction.operation.name == "cx":
                qubits = [qc.find_bit(q).index for q in instruction.qubits]
                assert abs(qubits[0] - qubits[1]) == 1, \
                    f"Non-adjacent CX: {qubits}"

    def test_reps_multiplies_gates(self) -> None:
        """More reps should multiply the gate count."""
        fm1 = HardwareEfficientFeatureMap(n_qubits=3, reps=1)
        fm2 = HardwareEfficientFeatureMap(n_qubits=3, reps=2)
        x = np.array([0.5, 1.0, 1.5])

        qc1 = fm1.build_circuit(x)
        qc2 = fm2.build_circuit(x)

        assert qc2.count_ops()["rz"] == 2 * qc1.count_ops()["rz"]
        assert qc2.count_ops()["sx"] == 2 * qc1.count_ops()["sx"]
        assert qc2.count_ops()["cx"] == 2 * qc1.count_ops()["cx"]

    def test_1qubit_no_cx(self) -> None:
        """1-qubit map should have no CX gates."""
        fm = HardwareEfficientFeatureMap(n_qubits=1, reps=1)
        x = np.array([1.0])
        qc = fm.build_circuit(x)
        ops = dict(qc.count_ops())
        assert ops.get("cx", 0) == 0
        assert ops.get("rz", 0) == 2
        assert ops.get("sx", 0) == 1

    def test_name_property(self) -> None:
        """name should include key parameters."""
        fm = HardwareEfficientFeatureMap(n_qubits=5, reps=2)
        assert "HWEfficient" in fm.name
        assert "5q" in fm.name
        assert "2reps" in fm.name

    def test_kernel_self_overlap_is_one(self) -> None:
        """K(x, x) using HW-Efficient should be 1.0."""
        from src.kernels.quantum import QuantumKernel

        fm = HardwareEfficientFeatureMap(n_qubits=2, reps=2)
        qk = QuantumKernel(fm, backend="statevector")

        x = np.array([1.0, 2.0])
        k = qk.compute_entry(x, x)
        assert np.isclose(k, 1.0, atol=1e-12)

    def test_total_gate_count_property(self) -> None:
        """total_gate_count should match actual circuit gate count."""
        fm = HardwareEfficientFeatureMap(n_qubits=3, reps=2)
        x = np.array([0.5, 1.0, 1.5])
        qc = fm.build_circuit(x)

        actual_total = sum(qc.count_ops().values())
        assert fm.total_gate_count == actual_total
