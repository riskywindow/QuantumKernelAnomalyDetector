"""Tests for geometric difference metric."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.geometric import (
    _matrix_sqrt,
    compute_bidirectional_geometric_difference,
    compute_geometric_difference,
    compute_pairwise_geometric_differences,
)


class TestMatrixSqrt:
    """Tests for the matrix square root helper."""

    def test_identity_sqrt_is_identity(self) -> None:
        """sqrt(I) = I."""
        I = np.eye(5)
        result = _matrix_sqrt(I)
        np.testing.assert_allclose(result, I, atol=1e-10)

    def test_sqrt_squared_gives_original(self) -> None:
        """sqrt(K) @ sqrt(K) should give back K."""
        rng = np.random.default_rng(42)
        X = rng.random((5, 3))
        K = X @ X.T  # PSD
        sqrt_K = _matrix_sqrt(K)
        reconstructed = sqrt_K @ sqrt_K
        np.testing.assert_allclose(reconstructed, K, atol=1e-10)

    def test_diagonal_matrix(self) -> None:
        """sqrt of diagonal matrix should have sqrt of diagonal entries."""
        d = np.array([4.0, 9.0, 16.0, 25.0])
        K = np.diag(d)
        result = _matrix_sqrt(K)
        expected = np.diag(np.sqrt(d))
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestGeometricDifference:
    """Tests for the geometric difference metric."""

    def test_same_kernel_gives_one(self) -> None:
        """g(K, K) should equal 1.0."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 3))
        K = X @ X.T + 0.01 * np.eye(10)  # Ensure non-singular
        g = compute_geometric_difference(K, K)
        assert g == pytest.approx(1.0, abs=0.1)

    def test_always_at_least_one(self) -> None:
        """Geometric difference should always be >= 1."""
        rng = np.random.default_rng(42)
        n = 10
        X1 = rng.random((n, 3))
        X2 = rng.random((n, 4))
        K1 = X1 @ X1.T + 0.01 * np.eye(n)
        K2 = X2 @ X2.T + 0.01 * np.eye(n)
        g = compute_geometric_difference(K1, K2)
        assert g >= 1.0

    def test_identity_vs_identity_gives_one(self) -> None:
        """g(I, I) should be 1.0."""
        n = 5
        g = compute_geometric_difference(np.eye(n), np.eye(n))
        assert g == pytest.approx(1.0, abs=0.1)

    def test_high_rank_vs_low_rank(self) -> None:
        """High-rank kernel should show advantage over low-rank."""
        n = 10
        K_high = np.eye(n)
        v = np.ones(n)
        K_low = np.outer(v, v) + 0.01 * np.eye(n)  # Nearly rank-1

        # g(K_high, K_low) should be > 1 since K_low can't approximate K_high
        g = compute_geometric_difference(K_high, K_low)
        assert g > 1.0

    def test_regularization_prevents_nan(self) -> None:
        """Regularization should prevent NaN/inf for singular matrices."""
        n = 10
        # Nearly singular matrix
        K_target = np.ones((n, n)) * 0.01
        K_target += np.eye(n) * 0.001
        K_approx = np.eye(n)
        g = compute_geometric_difference(K_target, K_approx, regularization=1e-3)
        assert np.isfinite(g)
        assert g >= 1.0

    def test_shape_mismatch_raises(self) -> None:
        """Should raise if kernel shapes don't match."""
        K1 = np.eye(5)
        K2 = np.eye(6)
        with pytest.raises(ValueError, match="same shape"):
            compute_geometric_difference(K1, K2)

    def test_non_square_raises(self) -> None:
        """Should raise for non-square kernel."""
        K1 = np.ones((3, 4))
        K2 = np.ones((3, 4))
        with pytest.raises(ValueError, match="square"):
            compute_geometric_difference(K1, K2)

    def test_known_analytical_case(self) -> None:
        """Test with a case where we can compute the answer analytically.

        For K_target = K_approx = aI, g should be 1.
        """
        n = 5
        for a in [0.5, 1.0, 2.0, 10.0]:
            K = a * np.eye(n)
            g = compute_geometric_difference(K, K, regularization=1e-8)
            assert g == pytest.approx(1.0, abs=0.2)


class TestBidirectionalGeometricDifference:
    """Tests for bidirectional geometric difference."""

    def test_returns_all_keys(self) -> None:
        """Should return dict with g_q_over_c, g_c_over_q, advantage_ratio."""
        n = 5
        K1 = np.eye(n)
        K2 = np.eye(n) * 2
        result = compute_bidirectional_geometric_difference(K1, K2)
        assert "g_q_over_c" in result
        assert "g_c_over_q" in result
        assert "advantage_ratio" in result

    def test_same_kernels_symmetric(self) -> None:
        """g_q_over_c should equal g_c_over_q when kernels are the same."""
        rng = np.random.default_rng(42)
        X = rng.random((8, 3))
        K = X @ X.T + 0.01 * np.eye(8)
        result = compute_bidirectional_geometric_difference(K, K)
        assert result["g_q_over_c"] == pytest.approx(result["g_c_over_q"], abs=0.2)
        assert result["advantage_ratio"] == pytest.approx(1.0, abs=0.3)


class TestPairwiseGeometricDifferences:
    """Tests for pairwise computation."""

    def test_diagonal_is_one(self) -> None:
        """g(K, K) should be 1.0 on the diagonal."""
        n = 5
        matrices = {"A": np.eye(n), "B": np.eye(n) * 2}
        result = compute_pairwise_geometric_differences(matrices)
        assert result[("A", "A")] == 1.0
        assert result[("B", "B")] == 1.0

    def test_all_pairs_computed(self) -> None:
        """Should compute all n^2 pairs."""
        n = 5
        matrices = {"A": np.eye(n), "B": np.eye(n) * 2, "C": np.eye(n) * 3}
        result = compute_pairwise_geometric_differences(matrices)
        assert len(result) == 9  # 3x3
