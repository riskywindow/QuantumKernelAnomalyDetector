"""Effective dimension and eigenspectrum analysis for kernel expressibility.

Measures how much of the feature space a kernel can effectively access.
Based on the eigenspectrum of the kernel matrix: broader spectra indicate
higher expressibility.

References:
    - Abbas et al. (2021): "The power of quantum neural networks" — Nature Computational Science
"""

from __future__ import annotations

import numpy as np


def compute_effective_dimension(
    K: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Compute effective dimension of a kernel matrix via spectral entropy.

    The effective dimension is exp(H) where H is the Shannon entropy of
    the normalized eigenvalue distribution. This measures the effective
    number of independent features captured by the kernel.

    - d_eff = 1: one eigenvalue dominates (low expressibility)
    - d_eff = n: all eigenvalues equal (maximum expressibility)

    Args:
        K: Kernel matrix of shape (n, n). Must be PSD.
        gamma: Scale parameter (unused in spectral entropy approach,
            kept for API compatibility).

    Returns:
        Effective dimension (float between 1 and n).
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")

    eigvals = np.linalg.eigvalsh(K)

    # Keep only positive eigenvalues
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 0.0

    # Normalize to probability distribution
    p = eigvals / eigvals.sum()

    # Shannon entropy of normalized eigenvalues
    entropy = -np.sum(p * np.log(p))

    # Effective dimension = exp(entropy)
    return float(np.exp(entropy))


def compute_participation_ratio(K: np.ndarray) -> float:
    """Compute the participation ratio of a kernel matrix.

    PR = (sum lambda_i)^2 / sum(lambda_i^2)

    The participation ratio equals 1 when one eigenvalue dominates and n
    when all eigenvalues are equal. It's a simpler alternative to spectral
    entropy, commonly used in physics.

    Args:
        K: Kernel matrix of shape (n, n). Must be PSD.

    Returns:
        Participation ratio (float between 1 and n).
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")

    eigvals = np.linalg.eigvalsh(K)

    # Keep only positive eigenvalues
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 0.0

    sum_sq = np.sum(eigvals**2)
    if sum_sq < 1e-15:
        return 0.0

    return float(np.sum(eigvals) ** 2 / sum_sq)


def compute_eigenspectrum(K: np.ndarray) -> dict:
    """Analyze the eigenspectrum of a kernel matrix.

    Provides a comprehensive set of spectral metrics characterizing the
    expressibility and structure of the kernel.

    Args:
        K: Kernel matrix of shape (n, n). Must be PSD.

    Returns:
        Dict with:
        - eigenvalues: Eigenvalues sorted descending.
        - normalized_eigenvalues: Eigenvalues normalized to sum to 1.
        - effective_dimension: exp(spectral entropy).
        - participation_ratio: (sum lambda)^2 / sum(lambda^2).
        - top_k_variance: Cumulative variance explained by top k eigenvalues
            (dict mapping k to fraction of total variance).
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")

    eigvals = np.linalg.eigvalsh(K)

    # Sort descending
    eigvals = np.sort(eigvals)[::-1]

    # Clip negative eigenvalues to zero (numerical noise)
    eigvals_pos = np.maximum(eigvals, 0.0)

    total = eigvals_pos.sum()
    if total < 1e-15:
        return {
            "eigenvalues": eigvals,
            "normalized_eigenvalues": np.zeros_like(eigvals),
            "effective_dimension": 0.0,
            "participation_ratio": 0.0,
            "top_k_variance": {},
        }

    normalized = eigvals_pos / total

    # Effective dimension via spectral entropy
    p = normalized[normalized > 1e-15]
    entropy = -np.sum(p * np.log(p))
    eff_dim = float(np.exp(entropy))

    # Participation ratio
    sum_sq = np.sum(eigvals_pos**2)
    pr = float(total**2 / sum_sq) if sum_sq > 1e-15 else 0.0

    # Cumulative variance explained
    cumulative = np.cumsum(normalized)
    top_k_variance = {}
    for k in [1, 2, 3, 5, 10, 20]:
        if k <= len(cumulative):
            top_k_variance[k] = float(cumulative[k - 1])

    return {
        "eigenvalues": eigvals,
        "normalized_eigenvalues": normalized,
        "effective_dimension": eff_dim,
        "participation_ratio": pr,
        "top_k_variance": top_k_variance,
    }


def compute_all_eigenspectra(
    kernel_matrices: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Compute eigenspectrum analysis for multiple kernels.

    Args:
        kernel_matrices: Dict mapping kernel names to (n, n) kernel matrices.

    Returns:
        Dict mapping kernel names to eigenspectrum analysis dicts.
    """
    results: dict[str, dict] = {}
    for name, K in kernel_matrices.items():
        results[name] = compute_eigenspectrum(K)
    return results
