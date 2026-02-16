"""Quantum kernel computation using parameterized feature maps.

Computes kernel entries as K(x1, x2) = |⟨0...0|U†(x2) U(x1)|0...0⟩|²
using statevector simulation for exact results.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector
from tqdm import tqdm

from src.kernels.base import BaseKernel
from src.kernels.feature_maps.zz import ZZFeatureMap


class QuantumKernel(BaseKernel):
    """Quantum kernel using parameterized feature maps.

    Computes the fidelity kernel K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|² where
    |ψ(x)⟩ = U(x)|0...0⟩ is the quantum state produced by the feature map.

    This is computed as the probability of measuring |0...0⟩ after applying
    the circuit U†(x2) U(x1) to the |0...0⟩ state.

    Args:
        feature_map: Feature map circuit builder (e.g., ZZFeatureMap).
        backend: Simulation backend. Currently only 'statevector' is supported.
    """

    def __init__(
        self,
        feature_map: ZZFeatureMap,
        backend: str = "statevector",
    ) -> None:
        if backend != "statevector":
            raise ValueError(f"Unsupported backend: {backend!r}. Use 'statevector'.")

        self.feature_map = feature_map
        self.backend = backend

    @property
    def name(self) -> str:
        """Human-readable name of the kernel."""
        return (
            f"QuantumKernel(ZZ, {self.feature_map.n_qubits}q, "
            f"{self.feature_map.reps}reps, {self.backend})"
        )

    def compute_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute a single kernel entry K(x1, x2).

        Builds the circuit U(x1) followed by U†(x2), then computes the
        probability of measuring |0...0⟩ using statevector simulation.

        Args:
            x1: First data point, shape (n_features,).
            x2: Second data point, shape (n_features,).

        Returns:
            Kernel value |⟨ψ(x1)|ψ(x2)⟩|² in [0, 1].
        """
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)

        # Build U(x1)
        circuit_x1 = self.feature_map.build_circuit(x1)

        # Build U†(x2) = inverse of U(x2)
        circuit_x2_dag = self.feature_map.build_circuit(x2).inverse()

        # Full circuit: U†(x2) @ U(x1)
        n_qubits = self.feature_map.n_qubits
        kernel_circuit = circuit_x1.compose(circuit_x2_dag)

        # Get statevector and extract |0...0⟩ probability
        sv = Statevector.from_instruction(kernel_circuit)
        # Probability of |0...0⟩ state (index 0)
        prob_zero = abs(sv.data[0]) ** 2

        return float(prob_zero)

    def compute_matrix(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute the kernel matrix.

        When X2 is None, exploits symmetry to only compute the upper
        triangle of the matrix (K[i,j] = K[j,i]).

        Args:
            X1: First set of data points, shape (n1, n_features).
            X2: Second set of data points, shape (n2, n_features).
                If None, computes K(X1, X1).
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            Kernel matrix of shape (n1, n2).
        """
        X1 = np.asarray(X1, dtype=np.float64)
        symmetric = X2 is None

        if symmetric:
            X2 = X1

        X2 = np.asarray(X2, dtype=np.float64)
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        if symmetric:
            # Exploit symmetry: only compute upper triangle
            total_entries = n1 * (n1 + 1) // 2
            pbar = tqdm(total=total_entries, desc="Kernel matrix", disable=not show_progress)

            for i in range(n1):
                K[i, i] = 1.0  # K(x, x) = 1 always
                pbar.update(1)
                for j in range(i + 1, n1):
                    K[i, j] = self.compute_entry(X1[i], X1[j])
                    K[j, i] = K[i, j]  # Symmetry
                    pbar.update(1)

            pbar.close()
        else:
            # General case: compute all entries
            total_entries = n1 * n2
            pbar = tqdm(total=total_entries, desc="Kernel matrix", disable=not show_progress)

            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.compute_entry(X1[i], X2[j])
                    pbar.update(1)

            pbar.close()

        return K
