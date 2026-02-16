"""Tests for the ZZ feature map implementation."""

import numpy as np
import pytest
from qiskit.circuit.library import zz_feature_map as qiskit_zz_feature_map
from qiskit.quantum_info import Statevector

from src.kernels.feature_maps.zz import ZZFeatureMap


class TestZZFeatureMap:
    """Tests for ZZ feature map circuit construction."""

    def test_circuit_num_qubits(self) -> None:
        """Circuit should have the correct number of qubits."""
        fm = ZZFeatureMap(n_qubits=3, reps=1)
        x = np.array([1.0, 2.0, 3.0])
        qc = fm.build_circuit(x)
        assert qc.num_qubits == 3

    def test_wrong_dimension_raises(self) -> None:
        """Passing wrong number of features should raise ValueError."""
        fm = ZZFeatureMap(n_qubits=2)
        x = np.array([1.0, 2.0, 3.0])  # 3 features, but 2 qubits
        with pytest.raises(ValueError, match="Expected data point"):
            fm.build_circuit(x)

    def test_gate_count_2q_1rep_linear(self) -> None:
        """Check gate counts for 2 qubits, 1 rep, linear entanglement."""
        fm = ZZFeatureMap(n_qubits=2, reps=1, entanglement="linear")
        x = np.array([0.5, 1.0])
        qc = fm.build_circuit(x)

        # Per rep: 2 H + 2 P(single) + 1*(CX+P+CX) = 2+2+3 = 7
        ops = dict(qc.count_ops())
        assert ops.get("h", 0) == 2
        assert ops.get("p", 0) == 3  # 2 single + 1 in ZZ block
        assert ops.get("cx", 0) == 2  # 2 CX per pair

    def test_gate_count_3q_1rep_full(self) -> None:
        """Check gate counts for 3 qubits, 1 rep, full entanglement."""
        fm = ZZFeatureMap(n_qubits=3, reps=1, entanglement="full")
        x = np.array([0.5, 1.0, 1.5])
        qc = fm.build_circuit(x)

        # 3 pairs for full: (0,1), (0,2), (1,2)
        # Per rep: 3 H + 3 P(single) + 3*(CX+P+CX) = 3+3+9 = 15
        ops = dict(qc.count_ops())
        assert ops.get("h", 0) == 3
        assert ops.get("p", 0) == 6  # 3 single + 3 in ZZ blocks
        assert ops.get("cx", 0) == 6  # 2 CX per pair * 3 pairs

    def test_reps_multiplies_gates(self) -> None:
        """More reps should multiply the number of gates."""
        fm1 = ZZFeatureMap(n_qubits=2, reps=1)
        fm2 = ZZFeatureMap(n_qubits=2, reps=2)
        x = np.array([0.5, 1.0])

        qc1 = fm1.build_circuit(x)
        qc2 = fm2.build_circuit(x)

        assert qc2.count_ops()["h"] == 2 * qc1.count_ops()["h"]
        assert qc2.count_ops()["p"] == 2 * qc1.count_ops()["p"]
        assert qc2.count_ops()["cx"] == 2 * qc1.count_ops()["cx"]

    def test_cross_validate_with_qiskit_2q(self) -> None:
        """Our implementation should match Qiskit's built-in ZZFeatureMap."""
        x = np.array([1.5, 0.7])

        # Our circuit
        our_fm = ZZFeatureMap(n_qubits=2, reps=2, entanglement="linear")
        our_circ = our_fm.build_circuit(x)
        our_sv = Statevector.from_instruction(our_circ)

        # Qiskit circuit
        qiskit_circ = qiskit_zz_feature_map(
            feature_dimension=2, reps=2, entanglement="linear"
        )
        qiskit_bound = qiskit_circ.assign_parameters(x)
        qiskit_sv = Statevector.from_instruction(qiskit_bound)

        # Statevectors should match (up to global phase)
        overlap = abs(np.vdot(our_sv.data, qiskit_sv.data))
        assert np.isclose(overlap, 1.0, atol=1e-10)

    def test_cross_validate_with_qiskit_3q(self) -> None:
        """Cross-validate 3-qubit case against Qiskit."""
        x = np.array([0.5, 1.2, 2.8])

        our_fm = ZZFeatureMap(n_qubits=3, reps=1, entanglement="linear")
        our_circ = our_fm.build_circuit(x)
        our_sv = Statevector.from_instruction(our_circ)

        qiskit_circ = qiskit_zz_feature_map(
            feature_dimension=3, reps=1, entanglement="linear"
        )
        qiskit_bound = qiskit_circ.assign_parameters(x)
        qiskit_sv = Statevector.from_instruction(qiskit_bound)

        overlap = abs(np.vdot(our_sv.data, qiskit_sv.data))
        assert np.isclose(overlap, 1.0, atol=1e-10)

    def test_cross_validate_kernel_values(self) -> None:
        """Kernel values from our circuits should match Qiskit-based kernels."""
        x1 = np.array([1.5, 0.7])
        x2 = np.array([0.3, 2.1])

        # Our kernel value
        our_fm = ZZFeatureMap(n_qubits=2, reps=2, entanglement="linear")
        our_c1 = our_fm.build_circuit(x1)
        our_c2_dag = our_fm.build_circuit(x2).inverse()
        our_kernel_circ = our_c1.compose(our_c2_dag)
        our_sv = Statevector.from_instruction(our_kernel_circ)
        our_k = abs(our_sv.data[0]) ** 2

        # Qiskit kernel value
        qc1 = qiskit_zz_feature_map(
            feature_dimension=2, reps=2, entanglement="linear"
        ).assign_parameters(x1)
        qc2_dag = qiskit_zz_feature_map(
            feature_dimension=2, reps=2, entanglement="linear"
        ).assign_parameters(x2).inverse()
        qiskit_kernel_circ = qc1.compose(qc2_dag)
        qiskit_sv = Statevector.from_instruction(qiskit_kernel_circ)
        qiskit_k = abs(qiskit_sv.data[0]) ** 2

        assert np.isclose(our_k, qiskit_k, atol=1e-10)

    def test_invalid_params(self) -> None:
        """Invalid constructor params should raise ValueError."""
        with pytest.raises(ValueError):
            ZZFeatureMap(n_qubits=0)
        with pytest.raises(ValueError):
            ZZFeatureMap(n_qubits=2, reps=0)
        with pytest.raises(ValueError):
            ZZFeatureMap(n_qubits=2, entanglement="invalid")

    def test_entangling_pairs_linear(self) -> None:
        """Linear entanglement should produce adjacent pairs."""
        fm = ZZFeatureMap(n_qubits=4, entanglement="linear")
        assert fm._entangling_pairs == [(0, 1), (1, 2), (2, 3)]

    def test_entangling_pairs_full(self) -> None:
        """Full entanglement should produce all pairs."""
        fm = ZZFeatureMap(n_qubits=3, entanglement="full")
        assert fm._entangling_pairs == [(0, 1), (0, 2), (1, 2)]
