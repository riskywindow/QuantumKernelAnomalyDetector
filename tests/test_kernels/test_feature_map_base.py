"""Tests for the BaseFeatureMap interface and all feature map implementations."""

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from src.kernels.feature_maps.base import BaseFeatureMap
from src.kernels.feature_maps.covariant import CovariantFeatureMap
from src.kernels.feature_maps.hardware_efficient import HardwareEfficientFeatureMap
from src.kernels.feature_maps.iqp import IQPFeatureMap
from src.kernels.feature_maps.zz import ZZFeatureMap


ALL_FEATURE_MAPS = [
    lambda: ZZFeatureMap(n_qubits=3, reps=1, entanglement="linear"),
    lambda: IQPFeatureMap(n_qubits=3, reps=1, entanglement="linear"),
    lambda: CovariantFeatureMap(n_qubits=3, reps=1),
    lambda: HardwareEfficientFeatureMap(n_qubits=3, reps=1),
]


class TestBaseFeatureMapInterface:
    """Tests that all feature maps properly implement BaseFeatureMap."""

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_inherits_from_base(self, fm_factory) -> None:
        """All feature maps should inherit from BaseFeatureMap."""
        fm = fm_factory()
        assert isinstance(fm, BaseFeatureMap)

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_has_n_qubits_property(self, fm_factory) -> None:
        """All feature maps should have n_qubits property."""
        fm = fm_factory()
        assert fm.n_qubits == 3

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_has_name_property(self, fm_factory) -> None:
        """All feature maps should have a non-empty name property."""
        fm = fm_factory()
        assert isinstance(fm.name, str)
        assert len(fm.name) > 0

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_has_reps_property(self, fm_factory) -> None:
        """All feature maps should have reps property."""
        fm = fm_factory()
        assert fm.reps == 1

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_has_total_gate_count(self, fm_factory) -> None:
        """All feature maps should have total_gate_count > 0."""
        fm = fm_factory()
        assert fm.total_gate_count > 0

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_build_circuit_returns_quantum_circuit(self, fm_factory) -> None:
        """build_circuit should return a QuantumCircuit."""
        fm = fm_factory()
        x = np.array([1.0, 2.0, 3.0])
        qc = fm.build_circuit(x)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_wrong_dimension_raises(self, fm_factory) -> None:
        """Passing wrong input dimension should raise ValueError."""
        fm = fm_factory()
        x_wrong = np.array([1.0, 2.0])  # 2 features for 3-qubit map
        with pytest.raises(ValueError, match="Expected data point"):
            fm.build_circuit(x_wrong)

    @pytest.mark.parametrize("fm_factory", ALL_FEATURE_MAPS)
    def test_invalid_n_qubits_raises(self, fm_factory) -> None:
        """n_qubits=0 should raise ValueError for all feature maps."""
        # Extract class from factory
        fm = fm_factory()
        cls = type(fm)
        with pytest.raises(ValueError):
            if cls in (ZZFeatureMap, IQPFeatureMap):
                cls(n_qubits=0, reps=1, entanglement="linear")
            else:
                cls(n_qubits=0, reps=1)
