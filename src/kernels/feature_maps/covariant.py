"""Covariant quantum feature map.

Covariant quantum kernels exploit symmetry in the data using a circuit
structure with two different rotation axes (RY and RZ) and ring entanglement.

Circuit structure per repetition:
    1. Data encoding: RY(x_i) on qubit i
    2. Ring entanglement: CX(0,1), CX(1,2), ..., CX(n-2,n-1), CX(n-1,0)
    3. Mixing layer: RZ(x_i) on qubit i

This feature map uses two different rotation axes (RY and RZ), accessing
more of the Bloch sphere than ZZ or IQP. The ring entanglement creates
a different correlation structure, and the design has a group-theoretic
motivation (data encoding respects rotation symmetry).
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from src.kernels.feature_maps.base import BaseFeatureMap


class CovariantFeatureMap(BaseFeatureMap):
    """Covariant feature map circuit for quantum kernel methods.

    Uses RY and RZ data-encoding layers interleaved with ring
    entanglement to exploit symmetry structure in the data.

    Args:
        n_qubits: Number of qubits (must equal the number of input features).
        reps: Number of circuit repetitions (layers). Default 2.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}")

        self._n_qubits = n_qubits
        self._reps = reps

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the feature map circuit."""
        return self._n_qubits

    @property
    def name(self) -> str:
        """Human-readable name of the feature map."""
        return f"Covariant({self._n_qubits}q, {self._reps}reps, ring)"

    @property
    def reps(self) -> int:
        """Number of circuit repetitions."""
        return self._reps

    def _get_ring_pairs(self) -> list[tuple[int, int]]:
        """Get the ring entanglement pairs.

        Ring entanglement: (0,1), (1,2), ..., (n-2,n-1), (n-1,0).
        For n=1, no entangling gates.

        Returns:
            List of (control, target) pairs forming a ring.
        """
        if self._n_qubits < 2:
            return []
        pairs = [(i, (i + 1) % self._n_qubits) for i in range(self._n_qubits)]
        return pairs

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build the covariant feature map circuit for a given data point.

        Args:
            x: Data point of shape (n_qubits,).

        Returns:
            Qiskit QuantumCircuit encoding the data point.

        Raises:
            ValueError: If x has wrong dimension.
        """
        x = self._validate_input(x)

        qc = QuantumCircuit(self._n_qubits)
        ring_pairs = self._get_ring_pairs()

        for _ in range(self._reps):
            # Layer 1: Data encoding with RY(x_i)
            for q in range(self._n_qubits):
                qc.ry(x[q], q)

            # Layer 2: Ring entanglement CX gates
            for i, j in ring_pairs:
                qc.cx(i, j)

            # Layer 3: Mixing layer with RZ(x_i)
            for q in range(self._n_qubits):
                qc.rz(x[q], q)

        return qc

    @property
    def total_gate_count(self) -> int:
        """Total number of gates in the circuit (per data point)."""
        n_cx = len(self._get_ring_pairs())
        per_rep = (
            self._n_qubits  # RY gates
            + n_cx  # CX gates in ring
            + self._n_qubits  # RZ gates
        )
        return per_rep * self._reps
