"""Noise impact study for quantum kernel matrices.

Studies how hardware noise degrades kernel matrix quality by computing
noisy kernel matrices across a range of noise levels and comparing
against the ideal (statevector) result.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from tqdm import tqdm

from src.kernels.feature_maps.base import BaseFeatureMap


def compute_noisy_kernel_matrix(
    feature_map: BaseFeatureMap,
    X: np.ndarray,
    noise_model: NoiseModel,
    shots: int = 4096,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute a kernel matrix using noisy shot-based simulation.

    Builds kernel circuits U†(x2) U(x1) for all pairs, adds measurements,
    and runs through AerSimulator with the given noise model.

    Args:
        feature_map: Quantum feature map for circuit construction.
        X: Data points of shape (n_samples, n_features).
        noise_model: Qiskit noise model for simulation.
        shots: Number of measurement shots per circuit.
        show_progress: Whether to show progress bar.

    Returns:
        Noisy kernel matrix of shape (n_samples, n_samples).
    """
    n = len(X)
    K = np.zeros((n, n))
    n_qubits = feature_map.n_qubits

    # Set up noisy simulator
    sim = AerSimulator(noise_model=noise_model)
    pm = generate_preset_pass_manager(optimization_level=0, backend=sim)

    # Build all kernel circuits (upper triangle only)
    circuits = []
    indices = []
    for i in range(n):
        for j in range(i, n):
            circuit_x1 = feature_map.build_circuit(X[i])
            circuit_x2_dag = feature_map.build_circuit(X[j]).inverse()
            kernel_circuit = circuit_x1.compose(circuit_x2_dag)

            # Add measurements
            meas_circuit = QuantumCircuit(n_qubits, n_qubits)
            meas_circuit.compose(kernel_circuit, inplace=True)
            meas_circuit.measure(range(n_qubits), range(n_qubits))

            circuits.append(meas_circuit)
            indices.append((i, j))

    # Transpile all circuits
    transpiled = pm.run(circuits)

    # Run circuits and extract kernel values
    zero_key = "0" * n_qubits
    iterator = enumerate(transpiled)
    if show_progress:
        iterator = tqdm(iterator, total=len(transpiled), desc="Noisy kernel")

    for idx, circ in iterator:
        result = sim.run(circ, shots=shots).result()
        counts = result.get_counts()
        zero_count = counts.get(zero_key, 0)
        k_val = float(zero_count / shots)

        i, j = indices[idx]
        K[i, j] = k_val
        if i != j:
            K[j, i] = k_val

    return K


def compute_kernel_fidelity(K_noisy: np.ndarray, K_ideal: np.ndarray) -> dict:
    """Compute multiple fidelity metrics between noisy and ideal kernels.

    Args:
        K_noisy: Noisy kernel matrix.
        K_ideal: Ideal (statevector) kernel matrix.

    Returns:
        Dictionary with:
        - frobenius_error: ||K_noisy - K_ideal||_F / ||K_ideal||_F
        - mean_abs_error: mean(|K_noisy - K_ideal|)
        - max_abs_error: max(|K_noisy - K_ideal|)
        - correlation: Pearson correlation of off-diagonal elements.
        - is_psd: Whether K_noisy is PSD.
    """
    diff = K_noisy - K_ideal
    frobenius_norm_ideal = np.linalg.norm(K_ideal, "fro")

    frobenius_error = float(
        np.linalg.norm(diff, "fro") / frobenius_norm_ideal
        if frobenius_norm_ideal > 0
        else np.linalg.norm(diff, "fro")
    )
    mean_abs_error = float(np.mean(np.abs(diff)))
    max_abs_error = float(np.max(np.abs(diff)))

    # Pearson correlation of off-diagonal elements
    n = K_noisy.shape[0]
    mask = ~np.eye(n, dtype=bool)
    noisy_off = K_noisy[mask]
    ideal_off = K_ideal[mask]

    if np.std(noisy_off) > 1e-12 and np.std(ideal_off) > 1e-12:
        correlation = float(np.corrcoef(noisy_off, ideal_off)[0, 1])
    else:
        correlation = 1.0 if np.allclose(noisy_off, ideal_off) else 0.0

    # PSD check
    min_eig = float(np.min(np.linalg.eigvalsh(K_noisy)))
    is_psd = min_eig >= -1e-10

    return {
        "frobenius_error": frobenius_error,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "correlation": correlation,
        "is_psd": is_psd,
    }


def run_noise_sweep(
    feature_map: BaseFeatureMap,
    X: np.ndarray,
    noise_models: list[tuple[float, NoiseModel]],
    ideal_K: np.ndarray,
    shots: int = 4096,
    show_progress: bool = True,
) -> dict:
    """Run kernel estimation across a sweep of noise levels.

    Args:
        feature_map: Quantum feature map for circuit construction.
        X: Data points to compute kernel on.
        noise_models: List of (error_rate, noise_model) from build_noise_sweep.
        ideal_K: Ideal (statevector) kernel matrix for comparison.
        shots: Number of measurement shots per circuit.
        show_progress: Whether to show progress.

    Returns:
        Dictionary with:
        - error_rates: List of error rates.
        - kernel_matrices: List of noisy kernel matrices.
        - fidelities: Frobenius error for each level.
        - mean_abs_errors: Mean absolute error for each level.
        - psd_violations: Whether each noisy K is PSD.
        - diagonal_errors: Mean |diag(K_noisy) - 1.0| for each level.
        - correlations: Pearson correlation for each level.
    """
    error_rates = []
    kernel_matrices = []
    fidelities = []
    mean_abs_errors = []
    psd_violations = []
    diagonal_errors = []
    correlations = []

    for rate, noise_model in noise_models:
        if show_progress:
            print(f"  Noise level: single_qubit_error={rate:.6f}")

        K_noisy = compute_noisy_kernel_matrix(
            feature_map, X, noise_model, shots=shots, show_progress=False
        )

        metrics = compute_kernel_fidelity(K_noisy, ideal_K)

        error_rates.append(rate)
        kernel_matrices.append(K_noisy)
        fidelities.append(metrics["frobenius_error"])
        mean_abs_errors.append(metrics["mean_abs_error"])
        psd_violations.append(not metrics["is_psd"])
        diagonal_errors.append(float(np.mean(np.abs(np.diag(K_noisy) - 1.0))))
        correlations.append(metrics["correlation"])

    return {
        "error_rates": error_rates,
        "kernel_matrices": kernel_matrices,
        "fidelities": fidelities,
        "mean_abs_errors": mean_abs_errors,
        "psd_violations": psd_violations,
        "diagonal_errors": diagonal_errors,
        "correlations": correlations,
    }
