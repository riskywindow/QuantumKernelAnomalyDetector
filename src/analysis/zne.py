"""Zero-noise extrapolation (ZNE) for error mitigation.

Runs kernel circuits at artificially increased noise levels, then
extrapolates back to the zero-noise point to estimate the ideal
kernel value. This is a real error mitigation technique used in
production quantum computing.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from src.analysis.noise import compute_noisy_kernel_matrix
from src.hardware.noise_models import build_depolarizing_noise_model
from src.kernels.feature_maps.base import BaseFeatureMap


def scale_noise_model(
    scale_factor: float,
    base_single_qubit_error: float,
    base_two_qubit_error: float,
    base_readout_error: float,
):
    """Create a noise model with scaled error rates.

    Multiplies all error rates by scale_factor, capping at physical limits.

    Args:
        scale_factor: Noise amplification factor (1.0 = base noise).
        base_single_qubit_error: Baseline single-qubit error rate.
        base_two_qubit_error: Baseline two-qubit error rate.
        base_readout_error: Baseline readout error rate.

    Returns:
        NoiseModel with scaled error rates.
    """
    # Cap at physical limits: 3/4 for single-qubit depolarizing, 15/16 for two-qubit
    scaled_1q = min(base_single_qubit_error * scale_factor, 0.75)
    scaled_2q = min(base_two_qubit_error * scale_factor, 0.9375)
    scaled_ro = min(base_readout_error * scale_factor, 0.5)

    return build_depolarizing_noise_model(
        single_qubit_error=scaled_1q,
        two_qubit_error=scaled_2q,
        readout_error=scaled_ro,
    )


def _linear_fit(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear model: f(x) = a*x + b."""
    return a * x + b


def _exponential_fit(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential model: f(x) = a * exp(-c*x) + b."""
    return a * np.exp(-c * x) + b


def _extrapolate_to_zero(
    scale_factors: list[float],
    values: list[float],
    method: str = "linear",
) -> tuple[float, dict]:
    """Extrapolate a sequence of values to zero noise.

    Args:
        scale_factors: Noise amplification factors.
        values: Measured values at each noise level.
        method: 'linear' or 'exponential'.

    Returns:
        Tuple of (extrapolated_value, fit_info).
    """
    x = np.array(scale_factors)
    y = np.array(values)

    if method == "linear":
        popt, _ = curve_fit(_linear_fit, x, y)
        zne_value = float(_linear_fit(0.0, *popt))
        # R-squared
        y_pred = _linear_fit(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        return zne_value, {"method": "linear", "params": list(popt), "r_squared": r_squared}

    elif method == "exponential":
        try:
            # Initial guess: a = y[0] - y[-1], b = y[-1], c = 1.0
            p0 = [y[0] - y[-1], y[-1], 1.0]
            popt, _ = curve_fit(_exponential_fit, x, y, p0=p0, maxfev=5000)
            zne_value = float(_exponential_fit(0.0, *popt))
            y_pred = _exponential_fit(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            return zne_value, {
                "method": "exponential",
                "params": list(popt),
                "r_squared": r_squared,
            }
        except (RuntimeError, ValueError):
            # Fall back to linear if exponential fails
            return _extrapolate_to_zero(scale_factors, values, method="linear")
    else:
        raise ValueError(f"Unknown extrapolation method: {method!r}. Use 'linear' or 'exponential'.")


def zero_noise_extrapolation(
    feature_map: BaseFeatureMap,
    x1: np.ndarray,
    x2: np.ndarray,
    base_noise_params: dict,
    scale_factors: list[float] | None = None,
    shots: int = 4096,
    extrapolation: str = "linear",
) -> dict:
    """Estimate a kernel entry using zero-noise extrapolation.

    Runs the kernel circuit at multiple noise levels and extrapolates
    to the zero-noise limit.

    Args:
        feature_map: Quantum feature map.
        x1, x2: Data points.
        base_noise_params: Dict with 'single_qubit_error', 'two_qubit_error',
            'readout_error' for the base noise level.
        scale_factors: Noise amplification factors. Default [1.0, 1.5, 2.0, 2.5, 3.0].
        shots: Shots per circuit per noise level.
        extrapolation: Fitting method — 'linear' or 'exponential'.

    Returns:
        Dictionary with:
        - zne_estimate: Extrapolated kernel value at zero noise.
        - raw_estimates: Kernel values at each noise level.
        - scale_factors: The noise scale factors used.
        - fit_info: Parameters of the extrapolation fit.
    """
    if scale_factors is None:
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    from qiskit.circuit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    n_qubits = feature_map.n_qubits
    base_1q = base_noise_params["single_qubit_error"]
    base_2q = base_noise_params["two_qubit_error"]
    base_ro = base_noise_params["readout_error"]

    # Build kernel circuit once
    circuit_x1 = feature_map.build_circuit(x1)
    circuit_x2_dag = feature_map.build_circuit(x2).inverse()
    kernel_circuit = circuit_x1.compose(circuit_x2_dag)

    meas_circuit = QuantumCircuit(n_qubits, n_qubits)
    meas_circuit.compose(kernel_circuit, inplace=True)
    meas_circuit.measure(range(n_qubits), range(n_qubits))

    raw_estimates = []
    zero_key = "0" * n_qubits

    for sf in scale_factors:
        noise_model = scale_noise_model(sf, base_1q, base_2q, base_ro)
        sim = AerSimulator(noise_model=noise_model)
        pm = generate_preset_pass_manager(optimization_level=0, backend=sim)
        transpiled = pm.run(meas_circuit)
        result = sim.run(transpiled, shots=shots).result()
        counts = result.get_counts()
        k_val = float(counts.get(zero_key, 0) / shots)
        raw_estimates.append(k_val)

    zne_value, fit_info = _extrapolate_to_zero(scale_factors, raw_estimates, extrapolation)

    # Clamp to [0, 1] since kernel values are probabilities
    zne_value = float(np.clip(zne_value, 0.0, 1.0))

    return {
        "zne_estimate": zne_value,
        "raw_estimates": raw_estimates,
        "scale_factors": scale_factors,
        "fit_info": fit_info,
    }


def zne_kernel_matrix(
    feature_map: BaseFeatureMap,
    X: np.ndarray,
    base_noise_params: dict,
    scale_factors: list[float] | None = None,
    shots: int = 4096,
    extrapolation: str = "linear",
    show_progress: bool = True,
) -> dict:
    """Compute a full ZNE-corrected kernel matrix.

    Computes the full kernel matrix at each noise level, then
    extrapolates each entry independently.

    Args:
        feature_map: Quantum feature map.
        X: Data points of shape (n_samples, n_features).
        base_noise_params: Dict with 'single_qubit_error', 'two_qubit_error',
            'readout_error' for the base noise level.
        scale_factors: Noise amplification factors. Default [1.0, 1.5, 2.0, 2.5, 3.0].
        shots: Shots per circuit per noise level.
        extrapolation: Fitting method — 'linear' or 'exponential'.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with:
        - zne_matrix: ZNE-corrected kernel matrix.
        - raw_matrices: Kernel matrix at each noise level.
        - scale_factors: Noise levels used.
        - mean_correction: Average |zne_value - base_value| across entries.
    """
    if scale_factors is None:
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    base_1q = base_noise_params["single_qubit_error"]
    base_2q = base_noise_params["two_qubit_error"]
    base_ro = base_noise_params["readout_error"]

    n = len(X)
    raw_matrices = []

    # Compute kernel matrix at each noise level
    for sf_idx, sf in enumerate(scale_factors):
        noise_model = scale_noise_model(sf, base_1q, base_2q, base_ro)
        if show_progress:
            print(f"  Scale factor {sf:.1f} ({sf_idx + 1}/{len(scale_factors)})")
        K = compute_noisy_kernel_matrix(
            feature_map, X, noise_model, shots=shots, show_progress=False
        )
        raw_matrices.append(K)

    # Extrapolate each entry independently
    zne_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            values = [K_sf[i, j] for K_sf in raw_matrices]
            zne_val, _ = _extrapolate_to_zero(scale_factors, values, extrapolation)
            zne_val = float(np.clip(zne_val, 0.0, 1.0))
            zne_matrix[i, j] = zne_val
            zne_matrix[j, i] = zne_val

    # Mean correction magnitude
    base_matrix = raw_matrices[0]  # scale_factor=1.0
    mean_correction = float(np.mean(np.abs(zne_matrix - base_matrix)))

    return {
        "zne_matrix": zne_matrix,
        "raw_matrices": raw_matrices,
        "scale_factors": scale_factors,
        "mean_correction": mean_correction,
    }
