"""ZZ Feature Map for quantum kernel methods.

Implements the ZZ feature map from Havlíček et al. (2019) "Supervised learning
with quantum-enhanced feature spaces". The circuit encodes classical data points
into quantum states using a combination of single-qubit rotations and two-qubit
ZZ entangling gates.

Circuit structure per repetition:
    1. H (Hadamard) on all qubits
    2. P(2*x_i) phase rotation on qubit i
    3. CX-P(2*(pi - x_i)*(pi - x_j))-CX entangling on connected pairs (i, j)
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from src.kernels.feature_maps.base import BaseFeatureMap


class ZZFeatureMap(BaseFeatureMap):
    """ZZ feature map circuit for quantum kernel methods.

    Encodes a classical data point x into a quantum state |ψ(x)⟩ using
    the Havlíček et al. (2019) circuit structure with ZZ entanglement.

    Args:
        n_qubits: Number of qubits (must equal the number of input features).
        reps: Number of circuit repetitions (layers). Default 2.
        entanglement: Entanglement pattern, either 'linear' or 'full'.
            'linear': pairs (0,1), (1,2), ..., (n-2, n-1)
            'full': all pairs (i, j) for i < j
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        entanglement: str = "linear",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}")
        if entanglement not in ("linear", "full"):
            raise ValueError(f"entanglement must be 'linear' or 'full', got {entanglement!r}")

        self._n_qubits = n_qubits
        self._reps = reps
        self.entanglement = entanglement
        self._entangling_pairs = self._get_entangling_pairs()

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the feature map circuit."""
        return self._n_qubits

    @property
    def name(self) -> str:
        """Human-readable name of the feature map."""
        return f"ZZ({self._n_qubits}q, {self._reps}reps, {self.entanglement})"

    @property
    def reps(self) -> int:
        """Number of circuit repetitions."""
        return self._reps

    def _get_entangling_pairs(self) -> list[tuple[int, int]]:
        """Get the list of qubit pairs for entangling gates.

        Returns:
            List of (control, target) pairs.
        """
        if self.entanglement == "linear":
            return [(i, i + 1) for i in range(self._n_qubits - 1)]
        else:  # full
            return [(i, j) for i in range(self._n_qubits) for j in range(i + 1, self._n_qubits)]

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build the feature map circuit for a given data point.

        Constructs the full parameterized circuit U_Φ(x) that maps
        the classical data point x to a quantum state.

        Args:
            x: Data point of shape (n_qubits,). Each feature is used
                as a rotation angle parameter.

        Returns:
            Qiskit QuantumCircuit encoding the data point.

        Raises:
            ValueError: If x has wrong dimension.
        """
        x = self._validate_input(x)

        qc = QuantumCircuit(self._n_qubits)

        for _ in range(self._reps):
            # Layer 1: Hadamard on all qubits
            for q in range(self._n_qubits):
                qc.h(q)

            # Layer 2: Single-qubit phase rotations P(2*x_i)
            for q in range(self._n_qubits):
                qc.p(2 * x[q], q)

            # Layer 3: Two-qubit ZZ entangling gates
            # CX(i,j) - P(2*(π - x_i)(π - x_j)) on j - CX(i,j)
            for i, j in self._entangling_pairs:
                angle = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                qc.cx(i, j)
                qc.p(angle, j)
                qc.cx(i, j)

        return qc

    @property
    def num_parameters_per_rep(self) -> int:
        """Number of parameterized gates per repetition."""
        n_single = self._n_qubits  # P gates
        n_entangling = len(self._entangling_pairs)  # ZZ gates
        return n_single + n_entangling

    @property
    def total_gate_count(self) -> int:
        """Total number of gates in the circuit (per data point).

        Counts: H gates + P gates + CX gates + P gates in ZZ blocks.
        """
        n_pairs = len(self._entangling_pairs)
        per_rep = (
            self._n_qubits  # H gates
            + self._n_qubits  # P(2*x_i) gates
            + n_pairs * 3  # CX + P + CX per pair
        )
        return per_rep * self._reps
