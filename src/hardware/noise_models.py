"""Noise model construction for studying hardware noise effects on quantum kernels.

Provides configurable depolarizing noise models for systematic study of how
different noise levels affect kernel quality, plus utilities for fetching
real noise models from IBM Quantum backends.
"""

from __future__ import annotations

import numpy as np
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def build_depolarizing_noise_model(
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.02,
) -> NoiseModel:
    """Build a simple depolarizing noise model.

    Args:
        single_qubit_error: Depolarizing error rate for single-qubit gates.
        two_qubit_error: Depolarizing error rate for two-qubit (CX) gates.
        readout_error: Probability of bit-flip on measurement.

    Returns:
        Qiskit NoiseModel with depolarizing errors on all gates.
    """
    noise_model = NoiseModel()

    # Single-qubit depolarizing error on all single-qubit gates
    if single_qubit_error > 0:
        error_1q = depolarizing_error(single_qubit_error, 1)
        for gate in ["rz", "sx", "h", "p", "ry"]:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)

    # Two-qubit depolarizing error on CX gates
    if two_qubit_error > 0:
        error_2q = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, "cx")

    # Readout error on all qubits (symmetric bit-flip)
    if readout_error > 0:
        ro_error = ReadoutError(
            [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
        )
        noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model


def build_noise_sweep(
    n_levels: int = 10,
    min_error: float = 0.0001,
    max_error: float = 0.05,
) -> list[tuple[float, NoiseModel]]:
    """Build a sweep of noise models from low to high noise.

    Error rates are logarithmically spaced. Two-qubit error is 10x the
    single-qubit error, and readout error is 2x the single-qubit error.

    Args:
        n_levels: Number of noise levels to generate.
        min_error: Minimum single-qubit error rate.
        max_error: Maximum single-qubit error rate.

    Returns:
        List of (single_qubit_error_rate, noise_model) tuples,
        sorted from lowest to highest noise.
    """
    error_rates = np.logspace(np.log10(min_error), np.log10(max_error), n_levels)

    sweep = []
    for rate in error_rates:
        noise_model = build_depolarizing_noise_model(
            single_qubit_error=float(rate),
            two_qubit_error=float(min(rate * 10, 0.9375)),  # Cap at 15/16
            readout_error=float(min(rate * 2, 0.5)),
        )
        sweep.append((float(rate), noise_model))

    return sweep


def try_fetch_real_noise_model(backend_name: str = "ibm_brisbane") -> NoiseModel | None:
    """Try to fetch noise model from real IBM backend.

    Returns None if IBM token is not available or backend is inaccessible.
    This allows graceful fallback to synthetic noise models.

    Args:
        backend_name: Name of the IBM Quantum backend.

    Returns:
        NoiseModel from the real backend, or None if unavailable.
    """
    try:
        import os

        from qiskit_ibm_runtime import QiskitRuntimeService

        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            return None

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend = service.backend(backend_name)
        noise_model = NoiseModel.from_backend(backend)
        return noise_model
    except Exception:
        return None
