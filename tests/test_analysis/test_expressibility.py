"""Tests for effective dimension and eigenspectrum analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.expressibility import (
    compute_all_eigenspectra,
    compute_effective_dimension,
    compute_eigenspectrum,
    compute_participation_ratio,
)


class TestEffectiveDimension:
    """Tests for spectral entropy effective dimension."""

    def test_identity_matrix_gives_max_dimension(self) -> None:
        """Identity matrix has all equal eigenvalues -> d_eff = n."""
        n = 10
        K = np.eye(n)
        d_eff = compute_effective_dimension(K)
        assert d_eff == pytest.approx(n, abs=0.01)

    def test_rank_one_matrix_gives_dimension_one(self) -> None:
        """Rank-1 matrix has one nonzero eigenvalue -> d_eff = 1."""
        v = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
        K = np.outer(v, v)
        d_eff = compute_effective_dimension(K)
        assert d_eff == pytest.approx(1.0, abs=0.01)

    def test_dimension_between_1_and_n(self) -> None:
        """Effective dimension should be between 1 and n for any PSD matrix."""
        rng = np.random.default_rng(42)
        n = 15
        X = rng.random((n, 5))
        K = X @ X.T
        d_eff = compute_effective_dimension(K)
        assert 1.0 <= d_eff <= n

    def test_two_equal_eigenvalues(self) -> None:
        """Matrix with exactly 2 equal nonzero eigenvalues -> d_eff = 2."""
        # Diagonal matrix with two equal eigenvalues
        K = np.diag([5.0, 5.0, 0.0, 0.0])
        d_eff = compute_effective_dimension(K)
        assert d_eff == pytest.approx(2.0, abs=0.01)

    def test_zero_matrix(self) -> None:
        """Zero matrix should return 0."""
        K = np.zeros((5, 5))
        d_eff = compute_effective_dimension(K)
        assert d_eff == 0.0

    def test_non_square_raises(self) -> None:
        """Should raise for non-square input."""
        K = np.ones((3, 4))
        with pytest.raises(ValueError, match="square"):
            compute_effective_dimension(K)


class TestParticipationRatio:
    """Tests for the participation ratio."""

    def test_identity_gives_n(self) -> None:
        """Identity matrix -> PR = n."""
        n = 10
        K = np.eye(n)
        pr = compute_participation_ratio(K)
        assert pr == pytest.approx(n, abs=0.01)

    def test_rank_one_gives_one(self) -> None:
        """Rank-1 matrix -> PR = 1."""
        v = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
        K = np.outer(v, v)
        pr = compute_participation_ratio(K)
        assert pr == pytest.approx(1.0, abs=0.01)

    def test_pr_between_1_and_n(self) -> None:
        """PR should be between 1 and n."""
        rng = np.random.default_rng(42)
        n = 15
        X = rng.random((n, 5))
        K = X @ X.T
        pr = compute_participation_ratio(K)
        assert 1.0 <= pr <= n

    def test_zero_matrix(self) -> None:
        """Zero matrix should return 0."""
        K = np.zeros((5, 5))
        pr = compute_participation_ratio(K)
        assert pr == 0.0


class TestEigenspectrum:
    """Tests for the full eigenspectrum analysis."""

    def test_eigenvalues_sorted_descending(self) -> None:
        """Eigenvalues should be sorted in descending order."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        K = X @ X.T
        spec = compute_eigenspectrum(K)
        eigvals = spec["eigenvalues"]
        assert all(eigvals[i] >= eigvals[i + 1] for i in range(len(eigvals) - 1))

    def test_normalized_eigenvalues_sum_to_one(self) -> None:
        """Normalized eigenvalues should sum to 1."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        K = X @ X.T
        spec = compute_eigenspectrum(K)
        norm_eigvals = spec["normalized_eigenvalues"]
        assert np.sum(norm_eigvals) == pytest.approx(1.0, abs=1e-10)

    def test_top_k_variance_monotonic(self) -> None:
        """Top-k variance should be monotonically increasing."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 5))
        K = X @ X.T
        spec = compute_eigenspectrum(K)
        top_k = spec["top_k_variance"]
        sorted_keys = sorted(top_k.keys())
        values = [top_k[k] for k in sorted_keys]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_identity_eigenspectrum(self) -> None:
        """Identity matrix eigenspectrum properties."""
        n = 8
        K = np.eye(n)
        spec = compute_eigenspectrum(K)
        assert spec["effective_dimension"] == pytest.approx(n, abs=0.01)
        assert spec["participation_ratio"] == pytest.approx(n, abs=0.01)
        np.testing.assert_allclose(spec["normalized_eigenvalues"], 1.0 / n, atol=1e-10)

    def test_consistent_with_standalone_functions(self) -> None:
        """Eigenspectrum dict should match standalone function results."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        K = X @ X.T
        spec = compute_eigenspectrum(K)
        d_eff = compute_effective_dimension(K)
        pr = compute_participation_ratio(K)
        assert spec["effective_dimension"] == pytest.approx(d_eff, abs=1e-6)
        assert spec["participation_ratio"] == pytest.approx(pr, abs=1e-6)


class TestComputeAllEigenspectra:
    """Tests for the batch eigenspectrum function."""

    def test_returns_dict_with_correct_keys(self) -> None:
        """Should return a dict with the same keys as input."""
        K1 = np.eye(5)
        K2 = np.ones((5, 5))
        result = compute_all_eigenspectra({"A": K1, "B": K2})
        assert set(result.keys()) == {"A", "B"}
        assert "effective_dimension" in result["A"]
        assert "participation_ratio" in result["B"]
