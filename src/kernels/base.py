"""Abstract kernel interface and kernel matrix validation."""

from abc import ABC, abstractmethod

import numpy as np


class BaseKernel(ABC):
    """Abstract base class for kernel functions.

    All kernel implementations (quantum and classical) must inherit
    from this class and implement compute_entry and compute_matrix.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the kernel."""

    @abstractmethod
    def compute_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute a single kernel entry K(x1, x2).

        Args:
            x1: First data point, shape (n_features,).
            x2: Second data point, shape (n_features,).

        Returns:
            Kernel value between x1 and x2.
        """

    @abstractmethod
    def compute_matrix(
        self, X1: np.ndarray, X2: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute the kernel matrix.

        Args:
            X1: First set of data points, shape (n1, n_features).
            X2: Second set of data points, shape (n2, n_features).
                If None, computes K(X1, X1).

        Returns:
            Kernel matrix of shape (n1, n2).
        """


def validate_kernel_matrix(
    K: np.ndarray,
    tol: float = 1e-8,
) -> dict[str, bool]:
    """Validate properties of a kernel matrix.

    Checks symmetry, positive semi-definiteness, unit diagonal,
    and boundedness (for quantum kernels).

    Args:
        K: Kernel matrix of shape (n, n) or (n, m).
        tol: Numerical tolerance for checks.

    Returns:
        Dictionary mapping property names to pass/fail booleans.
    """
    results: dict[str, bool] = {}

    is_square = K.shape[0] == K.shape[1]

    # Check symmetry (only meaningful for square matrices)
    if is_square:
        results["symmetric"] = bool(np.allclose(K, K.T, atol=tol))
    else:
        results["symmetric"] = False

    # Check PSD: all eigenvalues >= -tol (only for square symmetric matrices)
    if is_square:
        eigenvalues = np.linalg.eigvalsh(K)
        results["positive_semidefinite"] = bool(np.all(eigenvalues >= -tol))
    else:
        results["positive_semidefinite"] = False

    # Check diagonal = 1.0 (only for square matrices)
    if is_square:
        diag = np.diag(K)
        results["unit_diagonal"] = bool(np.allclose(diag, 1.0, atol=tol))
    else:
        results["unit_diagonal"] = False

    # Check bounds [0, 1] (specific to quantum kernels)
    results["bounded_0_1"] = bool(np.all(K >= -tol) and np.all(K <= 1.0 + tol))

    return results
