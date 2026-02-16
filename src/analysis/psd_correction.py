"""Kernel matrix PSD projection for noisy kernels.

Noisy quantum kernel matrices can lose positive semi-definiteness,
which breaks SVM training. This module provides correction methods
to recover valid kernel matrices.
"""

from __future__ import annotations

import numpy as np


def analyze_psd_violation(K: np.ndarray) -> dict:
    """Analyze the degree of PSD violation in a kernel matrix.

    Args:
        K: Kernel matrix to analyze.

    Returns:
        Dictionary with:
        - is_psd: Whether the matrix is PSD (within tolerance).
        - min_eigenvalue: Smallest eigenvalue.
        - n_negative_eigenvalues: Number of negative eigenvalues.
        - negative_eigenvalue_magnitude: Sum of |negative eigenvalues|.
        - correction_needed: 'none', 'minor' (|min_eig| < 0.01), or 'major'.
    """
    eigenvalues = np.linalg.eigvalsh(K)
    min_eig = float(eigenvalues[0])
    negative_mask = eigenvalues < -1e-10
    n_negative = int(np.sum(negative_mask))
    neg_magnitude = float(np.sum(np.abs(eigenvalues[negative_mask])))

    is_psd = min_eig >= -1e-10

    if is_psd:
        correction = "none"
    elif abs(min_eig) < 0.01:
        correction = "minor"
    else:
        correction = "major"

    return {
        "is_psd": is_psd,
        "min_eigenvalue": min_eig,
        "n_negative_eigenvalues": n_negative,
        "negative_eigenvalue_magnitude": neg_magnitude,
        "correction_needed": correction,
    }


def project_to_psd(K: np.ndarray, method: str = "clip") -> np.ndarray:
    """Project a matrix to the nearest PSD matrix.

    Args:
        K: Potentially non-PSD kernel matrix.
        method: Correction method:
            'clip': Eigenvalue clipping — set negative eigenvalues to 0.
            'nearest': Nearest PSD matrix in Frobenius norm (Higham 2002).
            'shift': Add |lambda_min| * I to make all eigenvalues non-negative.

    Returns:
        PSD-corrected kernel matrix.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "clip":
        return _psd_clip(K)
    elif method == "nearest":
        return _psd_nearest(K)
    elif method == "shift":
        return _psd_shift(K)
    else:
        raise ValueError(f"Unknown PSD correction method: {method!r}. Use 'clip', 'nearest', 'shift'.")


def _psd_clip(K: np.ndarray) -> np.ndarray:
    """Eigenvalue clipping: set negative eigenvalues to zero.

    This is the most common approach and tends to change the matrix minimally
    while ensuring PSD-ness.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0)
    K_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Restore symmetry (numerical precision)
    K_psd = (K_psd + K_psd.T) / 2
    return K_psd


def _psd_nearest(K: np.ndarray) -> np.ndarray:
    """Nearest PSD matrix in Frobenius norm (Higham 2002 alternating projections).

    Iteratively projects onto the set of symmetric matrices and the set of
    PSD matrices until convergence.
    """
    n = K.shape[0]
    Y = K.copy()
    delta_S = np.zeros_like(K)

    for _ in range(100):
        R = Y - delta_S
        # Project onto PSD cone
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        eigenvalues = np.maximum(eigenvalues, 0)
        X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        delta_S = X - R
        Y = (X + X.T) / 2

        # Check convergence
        if np.linalg.norm(Y - X, "fro") < 1e-10 * n:
            break

    return Y


def _psd_shift(K: np.ndarray) -> np.ndarray:
    """Shift: add |lambda_min| * I to make all eigenvalues non-negative."""
    lambda_min = float(np.min(np.linalg.eigvalsh(K)))
    if lambda_min < 0:
        n = K.shape[0]
        K_psd = K + (abs(lambda_min) + 1e-10) * np.eye(n)
    else:
        K_psd = K.copy()
    return K_psd
