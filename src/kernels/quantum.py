"""Quantum kernel computation using parameterized feature maps.

Computes kernel entries as K(x1, x2) = |⟨0...0|U†(x2) U(x1)|0...0⟩|²
using either exact statevector simulation or shot-based sampling,
with optional noise model support.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel
from tqdm import tqdm

from src.kernels.base import BaseKernel
from src.kernels.feature_maps.base import BaseFeatureMap


class QuantumKernel(BaseKernel):
    """Quantum kernel using parameterized feature maps.

    Computes the fidelity kernel K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|² where
    |ψ(x)⟩ = U(x)|0...0⟩ is the quantum state produced by the feature map.

    This is computed as the probability of measuring |0...0⟩ after applying
    the circuit U†(x2) U(x1) to the |0...0⟩ state.

    Args:
        feature_map: Feature map circuit builder (any BaseFeatureMap subclass).
        backend: Simulation backend. 'statevector' for exact simulation,
            'sampler' for shot-based estimation.
        n_shots: Number of measurement shots for sampler backend.
            Ignored when backend='statevector'.
        noise_model: Optional Qiskit noise model for noisy simulation.
            When provided with backend='sampler', uses AerSimulator with
            this noise model. Ignored when backend='statevector'.
    """

    def __init__(
        self,
        feature_map: BaseFeatureMap,
        backend: str = "statevector",
        n_shots: int = 1024,
        noise_model: NoiseModel | None = None,
    ) -> None:
        if backend not in ("statevector", "sampler"):
            raise ValueError(
                f"Unsupported backend: {backend!r}. Use 'statevector' or 'sampler'."
            )

        self.feature_map = feature_map
        self.backend = backend
        self.n_shots = n_shots
        self.noise_model = noise_model

    @property
    def name(self) -> str:
        """Human-readable name of the kernel."""
        if self.backend == "sampler":
            noise_str = ", noisy" if self.noise_model is not None else ""
            return (
                f"QuantumKernel({self.feature_map.name}, "
                f"{self.backend}, {self.n_shots}shots{noise_str})"
            )
        return f"QuantumKernel({self.feature_map.name}, {self.backend})"

    def _build_kernel_circuit(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> QuantumCircuit:
        """Build the kernel circuit U†(x2) U(x1).

        Args:
            x1: First data point.
            x2: Second data point.

        Returns:
            The kernel evaluation circuit.
        """
        circuit_x1 = self.feature_map.build_circuit(x1)
        circuit_x2_dag = self.feature_map.build_circuit(x2).inverse()
        return circuit_x1.compose(circuit_x2_dag)

    def _compute_entry_statevector(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> float:
        """Compute kernel entry using exact statevector simulation.

        Args:
            x1: First data point.
            x2: Second data point.

        Returns:
            Exact kernel value |⟨ψ(x1)|ψ(x2)⟩|².
        """
        kernel_circuit = self._build_kernel_circuit(x1, x2)
        sv = Statevector.from_instruction(kernel_circuit)
        prob_zero = abs(sv.data[0]) ** 2
        return float(prob_zero)

    def _compute_entry_sampler(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> float:
        """Compute kernel entry using shot-based sampling.

        Adds measurement gates to all qubits and uses AerSimulator
        to estimate the |0...0⟩ probability from finite shots.

        Args:
            x1: First data point.
            x2: Second data point.

        Returns:
            Estimated kernel value from shot statistics.
        """
        from qiskit_aer import AerSimulator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        kernel_circuit = self._build_kernel_circuit(x1, x2)

        # Add measurements for sampler
        n_qubits = self.feature_map.n_qubits
        meas_circuit = QuantumCircuit(n_qubits, n_qubits)
        meas_circuit.compose(kernel_circuit, inplace=True)
        meas_circuit.measure(range(n_qubits), range(n_qubits))

        # Transpile and run (with optional noise model)
        if self.noise_model is not None:
            sim = AerSimulator(noise_model=self.noise_model)
        else:
            sim = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=0, backend=sim)
        transpiled = pm.run(meas_circuit)
        result = sim.run(transpiled, shots=self.n_shots).result()
        counts = result.get_counts()

        # Extract probability of all-zeros bitstring
        zero_key = "0" * n_qubits
        zero_count = counts.get(zero_key, 0)
        return float(zero_count / self.n_shots)

    def compute_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute a single kernel entry K(x1, x2).

        Builds the circuit U(x1) followed by U†(x2), then computes the
        probability of measuring |0...0⟩ using the configured backend.

        Args:
            x1: First data point, shape (n_features,).
            x2: Second data point, shape (n_features,).

        Returns:
            Kernel value |⟨ψ(x1)|ψ(x2)⟩|² in [0, 1].
        """
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)

        if self.backend == "statevector":
            return self._compute_entry_statevector(x1, x2)
        else:
            return self._compute_entry_sampler(x1, x2)

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
