"""Kernel matrix estimation with disk caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.kernels.base import BaseKernel


class KernelEstimator:
    """Kernel matrix estimator with disk caching.

    Wraps kernel computation with automatic caching of computed
    kernel matrices as .npy files with JSON metadata sidecars.

    Args:
        cache_dir: Directory for cached kernel matrices.
            Defaults to experiments/results/kernels/.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or "experiments/results/kernels")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def estimate(
        self,
        kernel: BaseKernel,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
        cache_key: str | None = None,
    ) -> np.ndarray:
        """Compute or load a cached kernel matrix.

        Args:
            kernel: Kernel instance to use for computation.
            X1: First set of data points.
            X2: Second set of data points (None for symmetric case).
            cache_key: Optional cache key. If None, one is generated
                from data hashes.

        Returns:
            Kernel matrix as numpy array.
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(kernel, X1, X2)

        npy_path = self.cache_dir / f"{cache_key}.npy"
        meta_path = self.cache_dir / f"{cache_key}.json"

        # Try to load from cache
        if npy_path.exists() and meta_path.exists():
            K = np.load(npy_path)
            return K

        # Compute the kernel matrix
        K = kernel.compute_matrix(X1, X2)

        # Save to cache
        np.save(npy_path, K)
        metadata = {
            "kernel_name": kernel.name,
            "X1_shape": list(X1.shape),
            "X2_shape": list(X2.shape) if X2 is not None else None,
            "symmetric": X2 is None,
            "data_hash_X1": self._hash_array(X1),
            "data_hash_X2": self._hash_array(X2) if X2 is not None else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_key": cache_key,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return K

    @staticmethod
    def _hash_array(arr: np.ndarray) -> str:
        """Compute a short hash of a numpy array for cache identification.

        Args:
            arr: Array to hash.

        Returns:
            Hex digest string (first 12 characters of SHA-256).
        """
        return hashlib.sha256(arr.tobytes()).hexdigest()[:12]

    def _generate_cache_key(
        self,
        kernel: BaseKernel,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> str:
        """Generate a cache key from the kernel config and data.

        Args:
            kernel: Kernel instance.
            X1: First data matrix.
            X2: Second data matrix (or None).

        Returns:
            String cache key.
        """
        parts = [
            kernel.name.replace(" ", "_").replace("(", "").replace(")", ""),
            f"n1_{len(X1)}",
            self._hash_array(X1),
        ]
        if X2 is not None:
            parts.extend([f"n2_{len(X2)}", self._hash_array(X2)])
        else:
            parts.append("symmetric")

        return "_".join(parts)
