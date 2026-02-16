"""Geometric difference metric for quantum kernel advantage.

The geometric difference (Huang et al. 2021, Nature Communications) quantifies
whether a quantum kernel can achieve a prediction advantage over a classical
kernel on a given dataset.

g(K_Q, K_C) >> 1 indicates potential quantum advantage: predictions from K_Q
cannot be well-approximated by K_C.

References:
    - Huang et al. (2021): "Power of data in quantum machine learning"
"""

from __future__ import annotations

import numpy as np


def _matrix_sqrt(K: np.ndarray) -> np.ndarray:
    """Compute the matrix square root of a symmetric PSD matrix.

    Uses eigendecomposition for numerical stability rather than
    scipy.linalg.sqrtm, which can return complex values for
    nearly-PSD matrices.

    Args:
        K: Symmetric PSD matrix of shape (n, n).

    Returns:
        Matrix square root of shape (n, n).
    """
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_eigvals = np.sqrt(eigvals)
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def compute_geometric_difference(
    K_target: np.ndarray,
    K_approx: np.ndarray,
    regularization: float = 1e-5,
) -> float:
    """Compute geometric difference g(K_target, K_approx).

    g(K_target, K_approx) = sqrt(spectral_norm(sqrt(K_approx) @ inv(K_target) @ sqrt(K_approx)))

    If g >> 1: predictions from K_target CANNOT be well-approximated by
    K_approx, indicating an advantage for K_target.

    If g ~ 1: K_approx can approximate predictions from K_target (no advantage).

    Args:
        K_target: Kernel matrix whose predictions we're testing, shape (n, n).
        K_approx: Kernel matrix attempting to approximate, shape (n, n).
        regularization: Tikhonov regularization for numerical stability
            when inverting K_target.

    Returns:
        Geometric difference g >= 1.
    """
    K_target = np.asarray(K_target, dtype=np.float64)
    K_approx = np.asarray(K_approx, dtype=np.float64)
    n = K_target.shape[0]

    if K_target.shape != K_approx.shape:
        raise ValueError(
            f"Kernel matrices must have same shape, got "
            f"{K_target.shape} and {K_approx.shape}"
        )
    if K_target.ndim != 2 or K_target.shape[0] != K_target.shape[1]:
        raise ValueError(f"Kernel matrices must be square, got shape {K_target.shape}")

    # Regularize K_target for inversion stability
    K_target_reg = K_target + regularization * np.eye(n)

    # Compute sqrt(K_approx) via eigendecomposition (numerically stable)
    sqrt_K_approx = _matrix_sqrt(K_approx)

    # Compute inv(K_target)
    K_target_inv = np.linalg.inv(K_target_reg)

    # Compute M = sqrt(K_approx) @ inv(K_target) @ sqrt(K_approx)
    M = sqrt_K_approx @ K_target_inv @ sqrt_K_approx

    # Geometric difference = sqrt(spectral_norm(M))
    # spectral_norm = largest eigenvalue of symmetric matrix M
    eigvals = np.linalg.eigvalsh(M)
    spectral_norm = np.max(np.abs(eigvals))
    g = np.sqrt(spectral_norm)

    # g >= 1 by definition
    return float(max(g, 1.0))


def compute_bidirectional_geometric_difference(
    K_quantum: np.ndarray,
    K_classical: np.ndarray,
    regularization: float = 1e-5,
) -> dict[str, float]:
    """Compute geometric difference in both directions.

    - g_q_over_c = g(K_Q, K_C): Quantum advantage indicator.
      If >> 1, the quantum kernel captures structure the classical cannot.
    - g_c_over_q = g(K_C, K_Q): Classical advantage indicator.
      If >> 1, the classical kernel captures structure the quantum cannot.
    - advantage_ratio = g_q_over_c / g_c_over_q: >1 favors quantum, <1 favors classical.

    Args:
        K_quantum: Quantum kernel matrix (n, n).
        K_classical: Classical kernel matrix (n, n).
        regularization: Tikhonov regularization for numerical stability.

    Returns:
        Dict with g_q_over_c, g_c_over_q, and advantage_ratio.
    """
    g_q_over_c = compute_geometric_difference(
        K_quantum, K_classical, regularization
    )
    g_c_over_q = compute_geometric_difference(
        K_classical, K_quantum, regularization
    )

    advantage_ratio = g_q_over_c / g_c_over_q if g_c_over_q > 1e-15 else float("inf")

    return {
        "g_q_over_c": g_q_over_c,
        "g_c_over_q": g_c_over_q,
        "advantage_ratio": advantage_ratio,
    }


def compute_pairwise_geometric_differences(
    kernel_matrices: dict[str, np.ndarray],
    regularization: float = 1e-5,
) -> dict[tuple[str, str], float]:
    """Compute geometric difference for all pairs of kernels.

    Args:
        kernel_matrices: Dict mapping kernel names to (n, n) kernel matrices.
        regularization: Tikhonov regularization for numerical stability.

    Returns:
        Dict mapping (target_name, approx_name) to geometric difference.
    """
    names = list(kernel_matrices.keys())
    results: dict[tuple[str, str], float] = {}

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                results[(name_i, name_j)] = 1.0
            else:
                results[(name_i, name_j)] = compute_geometric_difference(
                    kernel_matrices[name_i],
                    kernel_matrices[name_j],
                    regularization,
                )

    return results
