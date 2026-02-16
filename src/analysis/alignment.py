"""Kernel target alignment analysis.

Measures how well a kernel matrix matches the ideal kernel for a classification
or anomaly detection task. Higher alignment means the kernel's similarity
structure better matches the label structure.

References:
    - Cristianini et al. (2001): "On Kernel-Target Alignment"
    - Cortes et al. (2012): "Algorithms for Learning Kernels Based on Centered Alignment"
"""

from __future__ import annotations

import numpy as np


def compute_kernel_target_alignment(
    K: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute kernel-target alignment (KTA).

    KTA(K, y) = <K, K_y>_F / (||K||_F * ||K_y||_F)

    where K_y[i,j] = y_i * y_j with y in {-1, +1}.

    Args:
        K: Kernel matrix of shape (n, n).
        y: Binary labels {0, 1}. Will be converted to {-1, +1}.

    Returns:
        KTA score in [-1, 1]. Higher = better alignment.

    Raises:
        ValueError: If K is not square or dimensions don't match y.
    """
    K = np.asarray(K, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    if len(y) != K.shape[0]:
        raise ValueError(
            f"K has {K.shape[0]} samples but y has {len(y)} labels"
        )

    # Convert {0, 1} to {-1, +1}
    y_pm = 2.0 * y - 1.0
    K_y = np.outer(y_pm, y_pm)

    # KTA = <K, K_y>_F / (||K||_F * ||K_y||_F)
    numerator = np.sum(K * K_y)
    denominator = np.linalg.norm(K, "fro") * np.linalg.norm(K_y, "fro")

    if denominator < 1e-15:
        return 0.0

    return float(numerator / denominator)


def _center_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix using the centering matrix H = I - (1/n) * 11^T.

    K_c = H @ K @ H

    This is equivalent to centering the implicit feature space to have
    zero mean.

    Args:
        K: Kernel matrix of shape (n, n).

    Returns:
        Centered kernel matrix of shape (n, n).
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_c = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_c


def compute_centered_kernel_target_alignment(
    K: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute centered kernel-target alignment (centered KTA).

    Centers both the kernel matrix K and the ideal kernel K_y before
    computing alignment. More robust than uncentered KTA, especially
    for imbalanced datasets.

    Args:
        K: Kernel matrix of shape (n, n).
        y: Binary labels {0, 1}. Will be converted to {-1, +1}.

    Returns:
        Centered KTA score in [-1, 1]. Higher = better alignment.

    Raises:
        ValueError: If K is not square or dimensions don't match y.
    """
    K = np.asarray(K, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    if len(y) != K.shape[0]:
        raise ValueError(
            f"K has {K.shape[0]} samples but y has {len(y)} labels"
        )

    # Convert {0, 1} to {-1, +1}
    y_pm = 2.0 * y - 1.0
    K_y = np.outer(y_pm, y_pm)

    # Center both matrices
    K_c = _center_kernel_matrix(K)
    K_y_c = _center_kernel_matrix(K_y)

    # Compute alignment on centered matrices
    numerator = np.sum(K_c * K_y_c)
    denominator = np.linalg.norm(K_c, "fro") * np.linalg.norm(K_y_c, "fro")

    if denominator < 1e-15:
        return 0.0

    return float(numerator / denominator)


def compute_all_kernel_alignments(
    kernel_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    centered: bool = True,
) -> dict[str, float]:
    """Compute kernel-target alignment for multiple kernels.

    Args:
        kernel_matrices: Dict mapping kernel names to (n, n) kernel matrices.
        y: Binary labels {0, 1}.
        centered: If True, use centered KTA (recommended).

    Returns:
        Dict mapping kernel names to KTA scores.
    """
    align_fn = (
        compute_centered_kernel_target_alignment
        if centered
        else compute_kernel_target_alignment
    )

    results: dict[str, float] = {}
    for name, K in kernel_matrices.items():
        results[name] = align_fn(K, y)

    return results
