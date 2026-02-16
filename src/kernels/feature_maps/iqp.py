"""IQP (Instantaneous Quantum Polynomial) Feature Map.

Implements the IQP feature map from Havlíček et al. (2019). IQP circuits
produce kernels that are classically hard to simulate, providing the
theoretical basis for quantum advantage in kernel methods.

Circuit structure per repetition:
    1. H (Hadamard) on all qubits
    2. RZ(x_i) single-qubit encoding on qubit i
    3. RZZ(x_i * x_j) two-qubit entangling on connected pairs (i, j)
       implemented as CX(i,j) → RZ(x_i * x_j) on j → CX(i,j)

Key difference from ZZ: IQP uses RZ gates (not P gates), and the two-qubit
interaction uses the raw product x_i * x_j (not the (π - x_i)(π - x_j) form).
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from src.kernels.feature_maps.base import BaseFeatureMap


class IQPFeatureMap(BaseFeatureMap):
    """IQP feature map circuit for quantum kernel methods.

    Encodes a classical data point x into a quantum state using
    Instantaneous Quantum Polynomial circuits. These circuits consist
    of Hadamard layers interleaved with diagonal unitaries encoding
    data as RZ and RZZ rotations.

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
        return f"IQP({self._n_qubits}q, {self._reps}reps, {self.entanglement})"

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
            return [
                (i, j)
                for i in range(self._n_qubits)
                for j in range(i + 1, self._n_qubits)
            ]

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build the IQP feature map circuit for a given data point.

        Args:
            x: Data point of shape (n_qubits,).

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

            # Layer 2: Single-qubit RZ(x_i) encoding
            for q in range(self._n_qubits):
                qc.rz(x[q], q)

            # Layer 3: Two-qubit RZZ(x_i * x_j) entangling
            # Implemented as CX(i,j) → RZ(x_i * x_j) on j → CX(i,j)
            for i, j in self._entangling_pairs:
                angle = x[i] * x[j]
                qc.cx(i, j)
                qc.rz(angle, j)
                qc.cx(i, j)

        return qc

    @property
    def total_gate_count(self) -> int:
        """Total number of gates in the circuit (per data point)."""
        n_pairs = len(self._entangling_pairs)
        per_rep = (
            self._n_qubits  # H gates
            + self._n_qubits  # RZ(x_i) gates
            + n_pairs * 3  # CX + RZ + CX per pair
        )
        return per_rep * self._reps
