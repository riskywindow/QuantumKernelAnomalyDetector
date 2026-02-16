"""Tests for PSD correction methods."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.psd_correction import analyze_psd_violation, project_to_psd


class TestAnalyzePsdViolation:
    """Tests for analyze_psd_violation."""

    def test_psd_matrix(self) -> None:
        K = np.eye(3)
        result = analyze_psd_violation(K)
        assert result["is_psd"] is True
        assert result["n_negative_eigenvalues"] == 0
        assert result["correction_needed"] == "none"

    def test_non_psd_matrix(self) -> None:
        # Matrix with a negative eigenvalue
        K = np.array([[1.0, 0.5, 0.9], [0.5, 1.0, 0.5], [0.9, 0.5, 1.0]])
        # Perturb to make non-PSD
        K[0, 2] = 1.5
        K[2, 0] = 1.5
        result = analyze_psd_violation(K)
        assert result["is_psd"] is False
        assert result["n_negative_eigenvalues"] > 0
        assert result["min_eigenvalue"] < 0

    def test_minor_violation(self) -> None:
        # Small negative eigenvalue
        eigenvalues = np.array([1.0, 0.5, -0.005])
        V = np.eye(3)  # Orthogonal eigenvectors
        K = V @ np.diag(eigenvalues) @ V.T
        result = analyze_psd_violation(K)
        assert result["is_psd"] is False
        assert result["correction_needed"] == "minor"

    def test_major_violation(self) -> None:
        eigenvalues = np.array([1.0, 0.5, -0.1])
        V = np.eye(3)
        K = V @ np.diag(eigenvalues) @ V.T
        result = analyze_psd_violation(K)
        assert result["is_psd"] is False
        assert result["correction_needed"] == "major"

    def test_negative_eigenvalue_magnitude(self) -> None:
        eigenvalues = np.array([1.0, -0.1, -0.2])
        V = np.eye(3)
        K = V @ np.diag(eigenvalues) @ V.T
        result = analyze_psd_violation(K)
        assert result["negative_eigenvalue_magnitude"] == pytest.approx(0.3, abs=1e-6)


class TestProjectToPsd:
    """Tests for project_to_psd."""

    def _make_non_psd(self) -> np.ndarray:
        """Create a known non-PSD matrix."""
        eigenvalues = np.array([2.0, 0.5, -0.3, -0.1])
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        K = Q @ np.diag(eigenvalues) @ Q.T
        return (K + K.T) / 2

    def test_clip_produces_psd(self) -> None:
        K = self._make_non_psd()
        K_psd = project_to_psd(K, method="clip")
        eigenvalues = np.linalg.eigvalsh(K_psd)
        assert np.all(eigenvalues >= -1e-10)

    def test_nearest_produces_psd(self) -> None:
        K = self._make_non_psd()
        K_psd = project_to_psd(K, method="nearest")
        eigenvalues = np.linalg.eigvalsh(K_psd)
        assert np.all(eigenvalues >= -1e-10)

    def test_shift_produces_psd(self) -> None:
        K = self._make_non_psd()
        K_psd = project_to_psd(K, method="shift")
        eigenvalues = np.linalg.eigvalsh(K_psd)
        assert np.all(eigenvalues >= -1e-10)

    def test_clip_preserves_symmetry(self) -> None:
        K = self._make_non_psd()
        K_psd = project_to_psd(K, method="clip")
        assert np.allclose(K_psd, K_psd.T)

    def test_nearest_preserves_symmetry(self) -> None:
        K = self._make_non_psd()
        K_psd = project_to_psd(K, method="nearest")
        assert np.allclose(K_psd, K_psd.T)

    def test_already_psd_minimal_change_clip(self) -> None:
        K = np.eye(5) * 2 + 0.1 * np.ones((5, 5))
        K_psd = project_to_psd(K, method="clip")
        assert np.allclose(K, K_psd, atol=1e-10)

    def test_already_psd_minimal_change_shift(self) -> None:
        K = np.eye(5) * 2 + 0.1 * np.ones((5, 5))
        K_psd = project_to_psd(K, method="shift")
        assert np.allclose(K, K_psd, atol=1e-10)

    def test_clip_sets_negative_eigenvalues_to_zero(self) -> None:
        eigenvalues = np.array([3.0, 1.0, -0.5])
        V = np.eye(3)
        K = V @ np.diag(eigenvalues) @ V.T
        K_psd = project_to_psd(K, method="clip")
        eigs_corrected = np.linalg.eigvalsh(K_psd)
        assert np.min(eigs_corrected) >= -1e-10

    def test_shift_adds_correct_offset(self) -> None:
        eigenvalues = np.array([3.0, 1.0, -0.5])
        V = np.eye(3)
        K = V @ np.diag(eigenvalues) @ V.T
        K_psd = project_to_psd(K, method="shift")
        # Shift should add 0.5 + eps to diagonal
        expected_shift = 0.5 + 1e-10
        assert np.allclose(np.diag(K_psd) - np.diag(K), expected_shift)

    def test_invalid_method_raises(self) -> None:
        K = np.eye(3)
        with pytest.raises(ValueError, match="Unknown PSD correction"):
            project_to_psd(K, method="invalid")

    def test_realistic_noisy_kernel(self) -> None:
        """Test on a realistic noisy kernel matrix."""
        rng = np.random.default_rng(42)
        # Start with a valid kernel (PSD, diagonal=1)
        n = 10
        X = rng.random((n, 3))
        K_ideal = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K_ideal[i, j] = np.exp(-np.sum((X[i] - X[j]) ** 2))

        # Add noise that may break PSD
        noise = rng.normal(0, 0.1, (n, n))
        noise = (noise + noise.T) / 2
        K_noisy = K_ideal + noise

        for method in ["clip", "nearest", "shift"]:
            K_psd = project_to_psd(K_noisy, method=method)
            eigenvalues = np.linalg.eigvalsh(K_psd)
            assert np.all(eigenvalues >= -1e-10), f"Method {method} failed PSD check"
