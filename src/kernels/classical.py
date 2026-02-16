"""Classical kernel baselines: RBF and Polynomial.

Implements classical kernels behind the same BaseKernel interface as quantum
kernels, enabling apples-to-apples comparisons. Uses scikit-learn's efficient
pairwise kernel implementations under the hood.

Note: Classical kernel matrices do NOT necessarily have unit diagonal or
values bounded to [0, 1]. The RBF kernel always has K(x,x) = 1, but the
Polynomial kernel does not unless gamma and coef0 are specifically tuned.
The validate_kernel_matrix function returns per-property results, so these
properties can be checked independently.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

from src.kernels.base import BaseKernel


class RBFKernel(BaseKernel):
    """Radial Basis Function (Gaussian) kernel.

    K(x1, x2) = exp(-gamma * ||x1 - x2||^2)

    Args:
        gamma: Kernel bandwidth parameter. If 'scale', uses
            1 / (n_features * X.var()) following sklearn convention.
    """

    def __init__(self, gamma: float | str = "scale") -> None:
        if isinstance(gamma, str) and gamma != "scale":
            raise ValueError(f"gamma must be a float or 'scale', got {gamma!r}")
        self._gamma_param = gamma
        self._gamma_value: float | None = None

    @property
    def name(self) -> str:
        """Human-readable name of the kernel."""
        if isinstance(self._gamma_param, str):
            return f"RBF(gamma={self._gamma_param})"
        return f"RBF(gamma={self._gamma_param:.4g})"

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Resolve the gamma parameter.

        Args:
            X: Data matrix used to compute 'scale' gamma.

        Returns:
            Resolved gamma value.
        """
        if isinstance(self._gamma_param, (int, float)):
            return float(self._gamma_param)
        # 'scale': 1 / (n_features * X.var())
        n_features = X.shape[1] if X.ndim > 1 else 1
        gamma = 1.0 / (n_features * X.var())
        return float(gamma)

    def compute_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute a single RBF kernel entry.

        Args:
            x1: First data point, shape (n_features,).
            x2: Second data point, shape (n_features,).

        Returns:
            RBF kernel value in (0, 1].
        """
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)

        if self._gamma_value is None:
            # Estimate gamma from the two points (fallback)
            combined = np.stack([x1, x2])
            self._gamma_value = self._resolve_gamma(combined)

        diff = x1 - x2
        return float(np.exp(-self._gamma_value * np.dot(diff, diff)))

    def compute_matrix(
        self, X1: np.ndarray, X2: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute the RBF kernel matrix.

        Args:
            X1: First set of data points, shape (n1, n_features).
            X2: Second set of data points, shape (n2, n_features).
                If None, computes K(X1, X1).

        Returns:
            Kernel matrix of shape (n1, n2).
        """
        X1 = np.asarray(X1, dtype=np.float64)

        gamma = self._resolve_gamma(X1)
        self._gamma_value = gamma

        if X2 is not None:
            X2 = np.asarray(X2, dtype=np.float64)

        return rbf_kernel(X1, X2, gamma=gamma)


class PolynomialKernel(BaseKernel):
    """Polynomial kernel.

    K(x1, x2) = (gamma * x1 . x2 + coef0)^degree

    Args:
        degree: Polynomial degree.
        gamma: Scaling factor. If 'scale', uses
            1 / (n_features * X.var()) following sklearn convention.
        coef0: Independent term (bias).
    """

    def __init__(
        self,
        degree: int = 3,
        gamma: float | str = "scale",
        coef0: float = 1.0,
    ) -> None:
        if isinstance(gamma, str) and gamma != "scale":
            raise ValueError(f"gamma must be a float or 'scale', got {gamma!r}")
        self.degree = degree
        self._gamma_param = gamma
        self._gamma_value: float | None = None
        self.coef0 = coef0

    @property
    def name(self) -> str:
        """Human-readable name of the kernel."""
        gamma_str = (
            self._gamma_param
            if isinstance(self._gamma_param, str)
            else f"{self._gamma_param:.4g}"
        )
        return f"Polynomial(d={self.degree}, gamma={gamma_str})"

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Resolve the gamma parameter.

        Args:
            X: Data matrix used to compute 'scale' gamma.

        Returns:
            Resolved gamma value.
        """
        if isinstance(self._gamma_param, (int, float)):
            return float(self._gamma_param)
        n_features = X.shape[1] if X.ndim > 1 else 1
        gamma = 1.0 / (n_features * X.var())
        return float(gamma)

    def compute_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute a single polynomial kernel entry.

        Args:
            x1: First data point, shape (n_features,).
            x2: Second data point, shape (n_features,).

        Returns:
            Polynomial kernel value.
        """
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)

        if self._gamma_value is None:
            combined = np.stack([x1, x2])
            self._gamma_value = self._resolve_gamma(combined)

        return float((self._gamma_value * np.dot(x1, x2) + self.coef0) ** self.degree)

    def compute_matrix(
        self, X1: np.ndarray, X2: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute the polynomial kernel matrix.

        Args:
            X1: First set of data points, shape (n1, n_features).
            X2: Second set of data points, shape (n2, n_features).
                If None, computes K(X1, X1).

        Returns:
            Kernel matrix of shape (n1, n2).
        """
        X1 = np.asarray(X1, dtype=np.float64)

        gamma = self._resolve_gamma(X1)
        self._gamma_value = gamma

        if X2 is not None:
            X2 = np.asarray(X2, dtype=np.float64)

        return polynomial_kernel(
            X1, X2, degree=self.degree, gamma=gamma, coef0=self.coef0
        )
