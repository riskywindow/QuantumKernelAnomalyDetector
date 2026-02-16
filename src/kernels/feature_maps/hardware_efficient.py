"""Hardware-efficient feature map using only IBM native gates.

Designed to minimize circuit depth on real IBM quantum hardware by using
only native gates from the IBM gate set: {RZ, SX, CX}. This means zero
transpilation overhead on IBM Eagle/Heron processors.

Circuit structure per repetition:
    1. Data encoding: RZ(x_i) → SX → RZ(x_i) on each qubit i
       (the RZ-SX-RZ sandwich accesses arbitrary Bloch sphere points
       using only native gates)
    2. Entangling layer: Linear nearest-neighbor CX gates
       CX(0,1), CX(1,2), ..., CX(n-2, n-1)
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from src.kernels.feature_maps.base import BaseFeatureMap


class HardwareEfficientFeatureMap(BaseFeatureMap):
    """Hardware-efficient feature map using only IBM native gates.

    Uses ONLY gates from IBM's native gate set {RZ, SX, CX}, requiring
    zero transpilation on IBM Eagle/Heron processors. This minimizes
    noise from gate decomposition.

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
        return f"HWEfficient({self._n_qubits}q, {self._reps}reps)"

    @property
    def reps(self) -> int:
        """Number of circuit repetitions."""
        return self._reps

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build the hardware-efficient feature map circuit.

        Args:
            x: Data point of shape (n_qubits,).

        Returns:
            Qiskit QuantumCircuit using only RZ, SX, CX gates.

        Raises:
            ValueError: If x has wrong dimension.
        """
        x = self._validate_input(x)

        qc = QuantumCircuit(self._n_qubits)

        for _ in range(self._reps):
            # Layer 1: Data encoding with RZ-SX-RZ sandwich per qubit
            for q in range(self._n_qubits):
                qc.rz(x[q], q)
                qc.sx(q)
                qc.rz(x[q], q)

            # Layer 2: Linear nearest-neighbor CX entanglement
            for q in range(self._n_qubits - 1):
                qc.cx(q, q + 1)

        return qc

    @property
    def total_gate_count(self) -> int:
        """Total number of gates in the circuit (per data point)."""
        n_cx = max(0, self._n_qubits - 1)
        per_rep = (
            self._n_qubits * 3  # RZ + SX + RZ per qubit
            + n_cx  # CX gates
        )
        return per_rep * self._reps
