"""Abstract base class for quantum feature maps."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit.circuit import QuantumCircuit


class BaseFeatureMap(ABC):
    """Abstract base class for quantum feature maps.

    All quantum feature map implementations must inherit from this class
    and implement the required interface. The contract is: every feature map
    takes a 1D numpy array of shape (n_qubits,) and returns a QuantumCircuit.
    """

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Number of qubits in the feature map circuit."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the feature map."""

    @property
    @abstractmethod
    def reps(self) -> int:
        """Number of circuit repetitions (layers)."""

    @property
    @abstractmethod
    def total_gate_count(self) -> int:
        """Total number of gates in the circuit (per data point)."""

    @abstractmethod
    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build the feature map circuit for a given data point.

        Args:
            x: Data point of shape (n_qubits,). Each feature is used
                as a rotation angle parameter.

        Returns:
            Qiskit QuantumCircuit encoding the data point.

        Raises:
            ValueError: If x has wrong dimension.
        """

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Validate and convert input data point.

        Args:
            x: Data point to validate.

        Returns:
            Validated numpy array of shape (n_qubits,).

        Raises:
            ValueError: If x has wrong dimension.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_qubits,):
            raise ValueError(
                f"Expected data point of shape ({self.n_qubits},), got {x.shape}"
            )
        return x
