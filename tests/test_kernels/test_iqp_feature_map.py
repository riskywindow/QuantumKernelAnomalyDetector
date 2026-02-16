"""Tests for the IQP feature map implementation."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.kernels.feature_maps.iqp import IQPFeatureMap
from src.kernels.feature_maps.zz import ZZFeatureMap


class TestIQPFeatureMap:
    """Tests for IQP feature map circuit construction."""

    def test_circuit_num_qubits_2q(self) -> None:
        """Circuit should have the correct number of qubits."""
        fm = IQPFeatureMap(n_qubits=2, reps=1)
        x = np.array([1.0, 2.0])
        qc = fm.build_circuit(x)
        assert qc.num_qubits == 2

    def test_circuit_num_qubits_3q(self) -> None:
        """Circuit should have 3 qubits for 3-qubit map."""
        fm = IQPFeatureMap(n_qubits=3, reps=1)
        x = np.array([1.0, 2.0, 3.0])
        qc = fm.build_circuit(x)
        assert qc.num_qubits == 3

    def test_gate_structure_2q_1rep(self) -> None:
        """Check gate types for 2 qubits, 1 rep, linear entanglement."""
        fm = IQPFeatureMap(n_qubits=2, reps=1, entanglement="linear")
        x = np.array([0.5, 1.0])
        qc = fm.build_circuit(x)

        ops = dict(qc.count_ops())
        # Per rep: 2 H + 2 RZ(single) + 1*(CX+RZ+CX) = 2+2+3 = 7
        assert ops.get("h", 0) == 2
        assert ops.get("rz", 0) == 3  # 2 single + 1 in RZZ block
        assert ops.get("cx", 0) == 2  # 2 CX per pair

    def test_gate_structure_contains_h_rz_cx(self) -> None:
        """IQP circuits should contain H, RZ, and CX gates."""
        fm = IQPFeatureMap(n_qubits=3, reps=1)
        x = np.array([0.5, 1.0, 1.5])
        qc = fm.build_circuit(x)
        ops = set(qc.count_ops().keys())
        assert "h" in ops
        assert "rz" in ops
        assert "cx" in ops

    def test_iqp_different_from_zz(self) -> None:
        """IQP should produce DIFFERENT circuits than ZZ for the same input."""
        x = np.array([1.5, 0.7])

        iqp = IQPFeatureMap(n_qubits=2, reps=2, entanglement="linear")
        zz = ZZFeatureMap(n_qubits=2, reps=2, entanglement="linear")

        iqp_circ = iqp.build_circuit(x)
        zz_circ = zz.build_circuit(x)

        iqp_sv = Statevector.from_instruction(iqp_circ)
        zz_sv = Statevector.from_instruction(zz_circ)

        # Statevectors should NOT be the same (up to global phase)
        overlap = abs(np.vdot(iqp_sv.data, zz_sv.data))
        assert not np.isclose(overlap, 1.0, atol=1e-6), \
            "IQP and ZZ should produce different states"

    def test_reps_multiplies_gates(self) -> None:
        """More reps should multiply the number of gates."""
        fm1 = IQPFeatureMap(n_qubits=2, reps=1)
        fm2 = IQPFeatureMap(n_qubits=2, reps=2)
        x = np.array([0.5, 1.0])

        qc1 = fm1.build_circuit(x)
        qc2 = fm2.build_circuit(x)

        assert qc2.count_ops()["h"] == 2 * qc1.count_ops()["h"]
        assert qc2.count_ops()["rz"] == 2 * qc1.count_ops()["rz"]

    def test_full_entanglement(self) -> None:
        """Full entanglement should produce more CX gates."""
        fm_lin = IQPFeatureMap(n_qubits=3, reps=1, entanglement="linear")
        fm_full = IQPFeatureMap(n_qubits=3, reps=1, entanglement="full")
        x = np.array([0.5, 1.0, 1.5])

        qc_lin = fm_lin.build_circuit(x)
        qc_full = fm_full.build_circuit(x)

        # Linear: 2 pairs → 4 CX, Full: 3 pairs → 6 CX
        assert qc_full.count_ops()["cx"] > qc_lin.count_ops()["cx"]

    def test_name_property(self) -> None:
        """name should include key parameters."""
        fm = IQPFeatureMap(n_qubits=5, reps=2, entanglement="linear")
        assert "IQP" in fm.name
        assert "5q" in fm.name
        assert "2reps" in fm.name

    def test_kernel_self_overlap_is_one(self) -> None:
        """K(x, x) using IQP should be 1.0."""
        from src.kernels.quantum import QuantumKernel

        fm = IQPFeatureMap(n_qubits=2, reps=2)
        qk = QuantumKernel(fm, backend="statevector")

        x = np.array([1.0, 2.0])
        k = qk.compute_entry(x, x)
        assert np.isclose(k, 1.0, atol=1e-12)
