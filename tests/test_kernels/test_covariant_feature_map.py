"""Tests for the Covariant feature map implementation."""

import numpy as np
import pytest

from src.kernels.feature_maps.covariant import CovariantFeatureMap


class TestCovariantFeatureMap:
    """Tests for Covariant feature map circuit construction."""

    def test_circuit_num_qubits(self) -> None:
        """Circuit should have the correct number of qubits."""
        fm = CovariantFeatureMap(n_qubits=3, reps=1)
        x = np.array([1.0, 2.0, 3.0])
        qc = fm.build_circuit(x)
        assert qc.num_qubits == 3

    def test_gate_structure_3q_1rep(self) -> None:
        """Check gate types for 3 qubits, 1 rep."""
        fm = CovariantFeatureMap(n_qubits=3, reps=1)
        x = np.array([0.5, 1.0, 1.5])
        qc = fm.build_circuit(x)

        ops = dict(qc.count_ops())
        # Per rep: 3 RY + 3 CX (ring) + 3 RZ
        assert ops.get("ry", 0) == 3
        assert ops.get("cx", 0) == 3  # ring: (0,1), (1,2), (2,0)
        assert ops.get("rz", 0) == 3

    def test_ring_entanglement_includes_wrap_around(self) -> None:
        """Ring entanglement should include CX(n-1, 0) wrap-around."""
        fm = CovariantFeatureMap(n_qubits=3, reps=1)
        ring_pairs = fm._get_ring_pairs()
        assert (2, 0) in ring_pairs, "Ring should wrap around: CX(n-1, 0)"

    def test_ring_entanglement_4q(self) -> None:
        """Ring entanglement for 4 qubits."""
        fm = CovariantFeatureMap(n_qubits=4, reps=1)
        ring_pairs = fm._get_ring_pairs()
        assert ring_pairs == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_ring_entanglement_2q(self) -> None:
        """Ring for 2 qubits: (0,1) and (1,0)."""
        fm = CovariantFeatureMap(n_qubits=2, reps=1)
        ring_pairs = fm._get_ring_pairs()
        assert ring_pairs == [(0, 1), (1, 0)]

    def test_two_rotation_axes(self) -> None:
        """Covariant should use both RY and RZ gates."""
        fm = CovariantFeatureMap(n_qubits=3, reps=1)
        x = np.array([0.5, 1.0, 1.5])
        qc = fm.build_circuit(x)
        ops = set(qc.count_ops().keys())
        assert "ry" in ops, "Should contain RY gates"
        assert "rz" in ops, "Should contain RZ gates"

    def test_reps_multiplies_gates(self) -> None:
        """More reps should multiply the gate count."""
        fm1 = CovariantFeatureMap(n_qubits=3, reps=1)
        fm2 = CovariantFeatureMap(n_qubits=3, reps=2)
        x = np.array([0.5, 1.0, 1.5])

        qc1 = fm1.build_circuit(x)
        qc2 = fm2.build_circuit(x)

        assert qc2.count_ops()["ry"] == 2 * qc1.count_ops()["ry"]
        assert qc2.count_ops()["rz"] == 2 * qc1.count_ops()["rz"]
        assert qc2.count_ops()["cx"] == 2 * qc1.count_ops()["cx"]

    def test_name_property(self) -> None:
        """name should include key parameters."""
        fm = CovariantFeatureMap(n_qubits=5, reps=2)
        assert "Covariant" in fm.name
        assert "5q" in fm.name
        assert "ring" in fm.name

    def test_kernel_self_overlap_is_one(self) -> None:
        """K(x, x) using Covariant should be 1.0."""
        from src.kernels.quantum import QuantumKernel

        fm = CovariantFeatureMap(n_qubits=2, reps=2)
        qk = QuantumKernel(fm, backend="statevector")

        x = np.array([1.0, 2.0])
        k = qk.compute_entry(x, x)
        assert np.isclose(k, 1.0, atol=1e-12)

    def test_1qubit_no_entanglement(self) -> None:
        """1-qubit covariant should have no CX gates."""
        fm = CovariantFeatureMap(n_qubits=1, reps=1)
        x = np.array([1.0])
        qc = fm.build_circuit(x)
        ops = dict(qc.count_ops())
        assert ops.get("cx", 0) == 0
        assert ops.get("ry", 0) == 1
        assert ops.get("rz", 0) == 1
