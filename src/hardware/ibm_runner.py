"""IBM Quantum job submission and local noise simulation.

Provides IBMQuantumRunner for real hardware execution and LocalNoiseRunner
for realistic noise simulation using Aer, enabling development and testing
without consuming hardware queue time.
"""

from __future__ import annotations

import os

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


class IBMQuantumRunner:
    """Manages connections to IBM Quantum hardware.

    Handles authentication, backend selection, transpilation,
    and job submission for kernel circuit execution.

    Args:
        backend_name: Specific backend to use. If None, selects least busy.
        channel: IBM Quantum channel ('ibm_quantum' or 'ibm_cloud').
        min_qubits: Minimum number of qubits required when auto-selecting backend.
    """

    def __init__(
        self,
        backend_name: str | None = None,
        channel: str = "ibm_quantum",
        min_qubits: int = 5,
    ) -> None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            raise ImportError(
                "qiskit-ibm-runtime is required for IBM Quantum hardware. "
                "Install with: uv add qiskit-ibm-runtime"
            )

        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise RuntimeError(
                "IBM_QUANTUM_TOKEN environment variable is not set. "
                "Set it with: export IBM_QUANTUM_TOKEN='your_token_here'\n"
                "Get your token at https://quantum.ibm.com/"
            )

        self.service = QiskitRuntimeService(channel=channel, token=token)

        if backend_name:
            self.backend = self.service.backend(backend_name)
        else:
            self.backend = self.service.least_busy(
                min_num_qubits=min_qubits, operational=True, simulator=False
            )

        self.backend_name = self.backend.name
        self._transpilation_stats: dict | None = None

    def get_backend_info(self) -> dict:
        """Return backend information for reporting.

        Returns:
            Dictionary with backend name, qubit count, basis gates,
            and error rate information.
        """
        config = self.backend.configuration()
        properties = self.backend.properties()

        info = {
            "name": self.backend_name,
            "n_qubits": config.n_qubits,
            "basis_gates": config.basis_gates,
        }

        if properties is not None:
            # Median CX error
            cx_errors = []
            for gate in properties.gates:
                if gate.gate == "cx":
                    for param in gate.parameters:
                        if param.name == "gate_error":
                            cx_errors.append(param.value)
            if cx_errors:
                info["median_cx_error"] = float(np.median(cx_errors))

            # Median readout error
            ro_errors = []
            for qubit_props in properties.qubits:
                for param in qubit_props:
                    if param.name == "readout_error":
                        ro_errors.append(param.value)
            if ro_errors:
                info["median_readout_error"] = float(np.median(ro_errors))

        return info

    def transpile_circuit(
        self, circuit: QuantumCircuit, optimization_level: int = 1
    ) -> QuantumCircuit:
        """Transpile a circuit for the selected backend.

        Args:
            circuit: Circuit to transpile.
            optimization_level: Transpiler optimization level (0-3).

        Returns:
            Transpiled circuit.
        """
        original_ops = dict(circuit.count_ops())
        original_depth = circuit.depth()

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=self.backend
        )
        transpiled = pm.run(circuit)

        transpiled_ops = dict(transpiled.count_ops())
        transpiled_depth = transpiled.depth()

        self._transpilation_stats = {
            "original_ops": original_ops,
            "original_depth": original_depth,
            "transpiled_ops": transpiled_ops,
            "transpiled_depth": transpiled_depth,
            "original_gate_count": sum(original_ops.values()),
            "transpiled_gate_count": sum(transpiled_ops.values()),
        }

        return transpiled

    def run_circuits(
        self, circuits: list[QuantumCircuit], shots: int = 1024
    ) -> list[dict[str, int]]:
        """Submit a batch of circuits and return measurement counts.

        Args:
            circuits: List of circuits to execute.
            shots: Number of measurement shots per circuit.

        Returns:
            List of measurement count dictionaries, one per circuit.
        """
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(backend=self.backend)
        job = sampler.run(circuits, shots=shots)
        print(f"Job ID: {job.job_id()}")
        print("Waiting for results...")
        result = job.result()

        all_counts = []
        for pub_result in result:
            counts = pub_result.data.meas.get_counts()
            all_counts.append(counts)

        return all_counts

    def estimate_kernel_entry(
        self, circuit: QuantumCircuit, shots: int = 1024
    ) -> float:
        """Run a single kernel circuit and extract the kernel value.

        Args:
            circuit: Kernel circuit with measurements.
            shots: Number of measurement shots.

        Returns:
            Kernel value (probability of all-zeros outcome).
        """
        counts_list = self.run_circuits([circuit], shots=shots)
        counts = counts_list[0]
        n_qubits = circuit.num_qubits
        zero_key = "0" * n_qubits
        zero_count = counts.get(zero_key, 0)
        return float(zero_count / shots)

    @property
    def transpilation_stats(self) -> dict | None:
        """Stats from the most recent transpilation."""
        return self._transpilation_stats


class LocalNoiseRunner:
    """Simulates IBM hardware noise locally using Aer.

    Uses noise models from real IBM backends or synthetic depolarizing
    models for realistic simulation without consuming hardware queue time.

    Args:
        noise_model: Noise model to use. If None, uses noiseless simulation.
        backend_name: If provided, tries to fetch noise model from real backend.
    """

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        backend_name: str | None = None,
    ) -> None:
        if noise_model is not None:
            self.noise_model = noise_model
        elif backend_name is not None:
            from src.hardware.noise_models import try_fetch_real_noise_model

            fetched = try_fetch_real_noise_model(backend_name)
            if fetched is not None:
                self.noise_model = fetched
            else:
                self.noise_model = None
        else:
            self.noise_model = None

        if self.noise_model is not None:
            self.simulator = AerSimulator(noise_model=self.noise_model)
        else:
            self.simulator = AerSimulator()

    def run_circuits(
        self, circuits: list[QuantumCircuit], shots: int = 1024
    ) -> list[dict[str, int]]:
        """Run circuits on the local noisy simulator.

        Args:
            circuits: List of circuits with measurement gates.
            shots: Number of measurement shots per circuit.

        Returns:
            List of measurement count dictionaries.
        """
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.simulator)
        transpiled = pm.run(circuits)

        all_counts = []
        for circ in transpiled:
            result = self.simulator.run(circ, shots=shots).result()
            counts = result.get_counts()
            all_counts.append(counts)

        return all_counts

    def estimate_kernel_entry(
        self, circuit: QuantumCircuit, shots: int = 1024
    ) -> float:
        """Run a single kernel circuit and extract the kernel value.

        Args:
            circuit: Kernel circuit with measurements.
            shots: Number of measurement shots.

        Returns:
            Kernel value (probability of all-zeros outcome).
        """
        counts_list = self.run_circuits([circuit], shots=shots)
        counts = counts_list[0]
        n_qubits = circuit.num_qubits
        zero_key = "0" * n_qubits
        zero_count = counts.get(zero_key, 0)
        return float(zero_count / shots)
