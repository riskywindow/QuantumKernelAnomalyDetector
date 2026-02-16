"""Tests for the kernel base interface and validation."""

import numpy as np
import pytest

from src.kernels.base import validate_kernel_matrix


class TestValidateKernelMatrix:
    """Tests for kernel matrix validation."""

    def test_valid_identity_matrix(self) -> None:
        """Identity matrix should pass all checks."""
        K = np.eye(5)
        results = validate_kernel_matrix(K)
        assert results["symmetric"] is True
        assert results["positive_semidefinite"] is True
        assert results["unit_diagonal"] is True
        assert results["bounded_0_1"] is True

    def test_valid_kernel_matrix(self) -> None:
        """A well-formed kernel matrix should pass all checks."""
        K = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        results = validate_kernel_matrix(K)
        assert results["symmetric"] is True
        assert results["positive_semidefinite"] is True
        assert results["unit_diagonal"] is True
        assert results["bounded_0_1"] is True

    def test_asymmetric_matrix(self) -> None:
        """Asymmetric matrix should fail symmetry check."""
        K = np.array([
            [1.0, 0.5],
            [0.3, 1.0],
        ])
        results = validate_kernel_matrix(K)
        assert results["symmetric"] is False

    def test_non_psd_matrix(self) -> None:
        """Matrix with negative eigenvalue should fail PSD check."""
        K = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
        ])
        results = validate_kernel_matrix(K)
        assert results["positive_semidefinite"] is False

    def test_non_unit_diagonal(self) -> None:
        """Matrix with diagonal != 1 should fail unit_diagonal check."""
        K = np.array([
            [0.9, 0.5],
            [0.5, 1.0],
        ])
        results = validate_kernel_matrix(K)
        assert results["unit_diagonal"] is False

    def test_out_of_bounds(self) -> None:
        """Matrix with values > 1 or < 0 should fail bounded check."""
        K = np.array([
            [1.0, 1.5],
            [1.5, 1.0],
        ])
        results = validate_kernel_matrix(K)
        assert results["bounded_0_1"] is False

    def test_non_square_matrix(self) -> None:
        """Non-square matrix should fail symmetry, PSD, diagonal checks."""
        K = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.6, 0.1],
        ])
        results = validate_kernel_matrix(K)
        assert results["symmetric"] is False
        assert results["positive_semidefinite"] is False
        assert results["unit_diagonal"] is False
