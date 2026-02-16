"""Tests for kernel factory."""

import pytest

from src.kernels.base import BaseKernel
from src.kernels.classical import PolynomialKernel, RBFKernel
from src.kernels.factory import (
    create_all_kernels,
    create_classical_kernel,
    create_quantum_kernel,
)
from src.kernels.quantum import QuantumKernel


class TestCreateQuantumKernel:
    """Tests for create_quantum_kernel."""

    @pytest.mark.parametrize(
        "fm_type",
        ["zz", "iqp", "covariant", "hardware_efficient"],
    )
    def test_creates_all_types(self, fm_type):
        """Should create a QuantumKernel for each supported feature map type."""
        config = {"type": fm_type, "reps": 2}
        kernel = create_quantum_kernel(config, n_qubits=3)
        assert isinstance(kernel, QuantumKernel)

    def test_unknown_type_raises(self):
        """Unknown feature map type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown quantum kernel type"):
            create_quantum_kernel({"type": "unknown"}, n_qubits=3)

    def test_entanglement_passed(self):
        """Entanglement parameter should be passed to ZZ and IQP."""
        config = {"type": "zz", "reps": 2, "entanglement": "full"}
        kernel = create_quantum_kernel(config, n_qubits=3)
        assert kernel.feature_map.entanglement == "full"

    def test_default_backend(self):
        """Default backend should be statevector."""
        config = {"type": "zz", "reps": 1}
        kernel = create_quantum_kernel(config, n_qubits=2)
        assert kernel.backend == "statevector"

    def test_custom_backend(self):
        """Should accept custom backend."""
        config = {"type": "zz", "reps": 1, "backend": "sampler", "n_shots": 512}
        kernel = create_quantum_kernel(config, n_qubits=2)
        assert kernel.backend == "sampler"
        assert kernel.n_shots == 512


class TestCreateClassicalKernel:
    """Tests for create_classical_kernel."""

    def test_creates_rbf(self):
        """Should create an RBF kernel."""
        config = {"type": "rbf", "gamma": "scale"}
        kernel = create_classical_kernel(config)
        assert isinstance(kernel, RBFKernel)

    def test_creates_polynomial(self):
        """Should create a Polynomial kernel."""
        config = {"type": "polynomial", "degree": 3, "gamma": "scale"}
        kernel = create_classical_kernel(config)
        assert isinstance(kernel, PolynomialKernel)
        assert kernel.degree == 3

    def test_unknown_type_raises(self):
        """Unknown kernel type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown classical kernel type"):
            create_classical_kernel({"type": "unknown"})


class TestCreateAllKernels:
    """Tests for create_all_kernels."""

    def test_creates_correct_count(self):
        """Should create the right number of kernels."""
        config = {
            "quantum_kernels": [
                {"type": "zz", "reps": 1},
                {"type": "iqp", "reps": 1},
            ],
            "classical_kernels": [
                {"type": "rbf", "gamma": "scale"},
            ],
        }
        kernels = create_all_kernels(config, n_qubits=3)
        assert len(kernels) == 3

    def test_display_names(self):
        """Should use short display names."""
        config = {
            "quantum_kernels": [
                {"type": "zz", "reps": 1},
                {"type": "hardware_efficient", "reps": 1},
            ],
            "classical_kernels": [
                {"type": "rbf", "gamma": "scale"},
                {"type": "polynomial", "degree": 2},
            ],
        }
        kernels = create_all_kernels(config, n_qubits=3)
        assert "ZZ" in kernels
        assert "HW-Efficient" in kernels
        assert "RBF" in kernels
        assert "Polynomial" in kernels

    def test_all_are_base_kernel(self):
        """All returned kernels should be BaseKernel instances."""
        config = {
            "quantum_kernels": [{"type": "zz", "reps": 1}],
            "classical_kernels": [{"type": "rbf"}],
        }
        kernels = create_all_kernels(config, n_qubits=3)
        for kernel in kernels.values():
            assert isinstance(kernel, BaseKernel)

    def test_empty_config(self):
        """Empty config should return empty dict."""
        kernels = create_all_kernels({}, n_qubits=3)
        assert len(kernels) == 0
