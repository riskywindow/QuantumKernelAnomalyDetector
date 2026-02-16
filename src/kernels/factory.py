"""Kernel factory for creating kernels from configuration dicts.

Decouples the experiment runner from knowing how to construct each
kernel type. Supports all quantum feature maps and classical kernels.
"""

from __future__ import annotations

from src.kernels.base import BaseKernel
from src.kernels.classical import PolynomialKernel, RBFKernel
from src.kernels.feature_maps import (
    CovariantFeatureMap,
    HardwareEfficientFeatureMap,
    IQPFeatureMap,
    ZZFeatureMap,
)
from src.kernels.quantum import QuantumKernel

_FEATURE_MAP_REGISTRY = {
    "zz": ZZFeatureMap,
    "iqp": IQPFeatureMap,
    "covariant": CovariantFeatureMap,
    "hardware_efficient": HardwareEfficientFeatureMap,
}


def create_quantum_kernel(
    config: dict,
    n_qubits: int = 5,
) -> QuantumKernel:
    """Create a QuantumKernel from a config dict.

    Args:
        config: Dict with 'type' (zz/iqp/covariant/hardware_efficient),
            'reps', 'entanglement', and optional 'backend'/'n_shots'.
        n_qubits: Number of qubits (= number of features).

    Returns:
        Configured QuantumKernel instance.

    Raises:
        ValueError: If the feature map type is unknown.
    """
    fm_type = config["type"]
    if fm_type not in _FEATURE_MAP_REGISTRY:
        raise ValueError(
            f"Unknown quantum kernel type: {fm_type!r}. "
            f"Available: {list(_FEATURE_MAP_REGISTRY.keys())}"
        )

    fm_cls = _FEATURE_MAP_REGISTRY[fm_type]
    reps = config.get("reps", 2)

    # Build feature map kwargs
    fm_kwargs: dict = {"n_qubits": n_qubits, "reps": reps}
    if "entanglement" in config and fm_type in ("zz", "iqp"):
        fm_kwargs["entanglement"] = config["entanglement"]

    feature_map = fm_cls(**fm_kwargs)

    # Build quantum kernel
    backend = config.get("backend", "statevector")
    n_shots = config.get("n_shots", 1024)
    return QuantumKernel(feature_map, backend=backend, n_shots=n_shots)


def create_classical_kernel(config: dict) -> BaseKernel:
    """Create a classical kernel from a config dict.

    Args:
        config: Dict with 'type' (rbf/polynomial) and kernel-specific params.

    Returns:
        Configured classical kernel instance.

    Raises:
        ValueError: If the kernel type is unknown.
    """
    kernel_type = config["type"]

    if kernel_type == "rbf":
        gamma = config.get("gamma", "scale")
        return RBFKernel(gamma=gamma)
    elif kernel_type == "polynomial":
        degree = config.get("degree", 3)
        gamma = config.get("gamma", "scale")
        coef0 = config.get("coef0", 1.0)
        return PolynomialKernel(degree=degree, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(
            f"Unknown classical kernel type: {kernel_type!r}. "
            f"Available: ['rbf', 'polynomial']"
        )


def create_all_kernels(
    experiment_config: dict,
    n_qubits: int = 5,
) -> dict[str, BaseKernel]:
    """Create all kernels defined in an experiment config.

    Args:
        experiment_config: Full experiment config dict containing
            'quantum_kernels' and 'classical_kernels' lists.
        n_qubits: Number of qubits for quantum kernels.

    Returns:
        Dict mapping kernel display names to kernel instances.
    """
    kernels: dict[str, BaseKernel] = {}

    for qk_config in experiment_config.get("quantum_kernels", []):
        kernel = create_quantum_kernel(qk_config, n_qubits=n_qubits)
        # Use short display name: "ZZ", "IQP", "Covariant", "HW-Efficient"
        display_name = {
            "zz": "ZZ",
            "iqp": "IQP",
            "covariant": "Covariant",
            "hardware_efficient": "HW-Efficient",
        }[qk_config["type"]]
        kernels[display_name] = kernel

    for ck_config in experiment_config.get("classical_kernels", []):
        kernel = create_classical_kernel(ck_config)
        display_name = {
            "rbf": "RBF",
            "polynomial": "Polynomial",
        }[ck_config["type"]]
        kernels[display_name] = kernel

    return kernels
