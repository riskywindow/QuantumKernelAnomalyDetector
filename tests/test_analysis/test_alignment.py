"""Tests for kernel target alignment."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.alignment import (
    _center_kernel_matrix,
    compute_all_kernel_alignments,
    compute_centered_kernel_target_alignment,
    compute_kernel_target_alignment,
)


class TestKernelTargetAlignment:
    """Tests for uncentered KTA."""

    def test_perfect_kernel_gives_kta_one(self) -> None:
        """KTA should be 1.0 when K exactly matches the label structure."""
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        y_pm = 2 * y - 1
        K = np.outer(y_pm, y_pm)
        kta = compute_kernel_target_alignment(K, y)
        assert kta == pytest.approx(1.0, abs=1e-10)

    def test_kta_bounded_minus_one_to_one(self) -> None:
        """KTA should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        n = 20
        K = rng.random((n, n))
        K = K @ K.T  # Make PSD
        K /= np.max(K)
        y = rng.integers(0, 2, size=n).astype(np.float64)
        kta = compute_kernel_target_alignment(K, y)
        assert -1.0 <= kta <= 1.0

    def test_identity_kernel_kta(self) -> None:
        """Identity kernel only matches same-sample pairs."""
        n = 10
        K = np.eye(n)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        kta = compute_kernel_target_alignment(K, y)
        # I has 1s on diagonal, K_y has 1s on diagonal (all +1 or -1 squared)
        # So <I, K_y> = sum of diagonal of K_y = n (all 1s)
        # ||I||_F = sqrt(n), ||K_y||_F = sqrt(n*n) = n
        # KTA = n / (sqrt(n) * n) = 1/sqrt(n)
        expected = 1.0 / np.sqrt(n)
        assert kta == pytest.approx(expected, abs=1e-10)

    def test_anti_aligned_kernel(self) -> None:
        """Kernel anti-aligned with labels should give negative KTA."""
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        y_pm = 2 * y - 1
        K = -np.outer(y_pm, y_pm)  # Anti-aligned
        # Make it a valid matrix by shifting
        K = K + 2.0 * np.eye(4)  # Add enough to make it PSD
        # The KTA computation doesn't require PSD
        kta_neg = compute_kernel_target_alignment(-np.outer(y_pm, y_pm), y)
        assert kta_neg == pytest.approx(-1.0, abs=1e-10)

    def test_3x3_known_example(self) -> None:
        """Test with a small 3x3 example."""
        K = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        y = np.array([0, 0, 1], dtype=np.float64)
        y_pm = 2 * y - 1  # [-1, -1, 1]
        K_y = np.outer(y_pm, y_pm)
        # K_y = [[1,1,-1],[1,1,-1],[-1,-1,1]]

        expected = np.sum(K * K_y) / (np.linalg.norm(K, "fro") * np.linalg.norm(K_y, "fro"))
        kta = compute_kernel_target_alignment(K, y)
        assert kta == pytest.approx(expected, abs=1e-10)

    def test_shape_mismatch_raises(self) -> None:
        """Should raise if K and y dimensions don't match."""
        K = np.eye(5)
        y = np.array([0, 1, 0], dtype=np.float64)
        with pytest.raises(ValueError, match="labels"):
            compute_kernel_target_alignment(K, y)

    def test_non_square_raises(self) -> None:
        """Should raise if K is not square."""
        K = np.ones((3, 4))
        y = np.array([0, 1, 0], dtype=np.float64)
        with pytest.raises(ValueError, match="square"):
            compute_kernel_target_alignment(K, y)


class TestCenteredKTA:
    """Tests for centered KTA."""

    def test_centered_kta_bounded(self) -> None:
        """Centered KTA should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        n = 20
        K = rng.random((n, n))
        K = K @ K.T
        K /= np.max(K)
        y = rng.integers(0, 2, size=n).astype(np.float64)
        kta = compute_centered_kernel_target_alignment(K, y)
        assert -1.0 - 1e-10 <= kta <= 1.0 + 1e-10

    def test_centered_kta_of_centered_ideal(self) -> None:
        """Centered KTA of the ideal kernel should be 1.0."""
        y = np.array([0, 0, 0, 1, 1], dtype=np.float64)
        y_pm = 2 * y - 1
        K = np.outer(y_pm, y_pm)
        kta = compute_centered_kernel_target_alignment(K, y)
        assert kta == pytest.approx(1.0, abs=1e-10)

    def test_centering_removes_constant_shift(self) -> None:
        """Centering should handle kernels with constant offset."""
        n = 10
        y = np.array([0] * 5 + [1] * 5, dtype=np.float64)
        y_pm = 2 * y - 1
        K_ideal = np.outer(y_pm, y_pm)
        # Add constant offset
        K_shifted = K_ideal + 5.0
        # Centered KTA should still be high
        kta = compute_centered_kernel_target_alignment(K_shifted, y)
        assert kta == pytest.approx(1.0, abs=1e-10)


class TestCenterKernelMatrix:
    """Tests for the centering helper."""

    def test_centered_has_zero_row_means(self) -> None:
        """Centered kernel matrix should have approximately zero row means."""
        rng = np.random.default_rng(42)
        K = rng.random((10, 10))
        K = K @ K.T
        K_c = _center_kernel_matrix(K)
        row_means = K_c.mean(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)

    def test_centering_identity(self) -> None:
        """Centering an already centered kernel should be idempotent."""
        rng = np.random.default_rng(42)
        K = rng.random((10, 10))
        K = K @ K.T
        K_c1 = _center_kernel_matrix(K)
        K_c2 = _center_kernel_matrix(K_c1)
        np.testing.assert_allclose(K_c1, K_c2, atol=1e-10)


class TestComputeAllAlignments:
    """Tests for the batch alignment function."""

    def test_returns_dict_with_correct_keys(self) -> None:
        """Should return a dict with the same keys as input."""
        n = 10
        K1 = np.eye(n)
        K2 = np.ones((n, n))
        y = np.array([0] * 5 + [1] * 5, dtype=np.float64)
        result = compute_all_kernel_alignments({"A": K1, "B": K2}, y)
        assert set(result.keys()) == {"A", "B"}
        assert all(isinstance(v, float) for v in result.values())

    def test_centered_vs_uncentered_flag(self) -> None:
        """centered=True should use centered KTA."""
        n = 10
        K = np.eye(n)
        y = np.array([0] * 5 + [1] * 5, dtype=np.float64)
        centered = compute_all_kernel_alignments({"K": K}, y, centered=True)
        uncentered = compute_all_kernel_alignments({"K": K}, y, centered=False)
        # These will generally differ
        assert isinstance(centered["K"], float)
        assert isinstance(uncentered["K"], float)
