"""Phase 5 verification script.

Quick verification that all Phase 5 components work end-to-end.
Target: under 120 seconds.
"""

from __future__ import annotations

import time

import numpy as np

from src.analysis.noise import compute_kernel_fidelity, compute_noisy_kernel_matrix
from src.analysis.psd_correction import analyze_psd_violation, project_to_psd
from src.analysis.zne import zero_noise_extrapolation
from src.hardware.ibm_runner import LocalNoiseRunner
from src.hardware.noise_models import build_depolarizing_noise_model
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


def main() -> None:
    start = time.time()
    all_passed = True

    print("=" * 60)
    print("PHASE 5 VERIFICATION")
    print("=" * 60)

    # Setup
    rng = np.random.default_rng(42)
    n_qubits = 3
    n_samples = 10
    fm = ZZFeatureMap(n_qubits=n_qubits, reps=1)
    X = rng.random((n_samples, n_qubits)) * 2 * np.pi

    # 1. Compute ideal kernel matrix
    print("\n1. Computing ideal kernel matrix (statevector)...")
    qk = QuantumKernel(fm, backend="statevector")
    K_ideal = qk.compute_matrix(X, show_progress=False)
    print(f"   Shape: {K_ideal.shape}")
    print(f"   Diagonal: all 1.0? {np.allclose(np.diag(K_ideal), 1.0)}")
    print(f"   Symmetric? {np.allclose(K_ideal, K_ideal.T)}")

    # 2. Compute noisy kernel matrix
    print("\n2. Computing noisy kernel matrix...")
    noise_model = build_depolarizing_noise_model(
        single_qubit_error=0.005, two_qubit_error=0.02, readout_error=0.01
    )
    K_noisy = compute_noisy_kernel_matrix(fm, X, noise_model, shots=2048)
    print(f"   Shape: {K_noisy.shape}")

    # 3. Compare ideal vs noisy
    print("\n3. Kernel fidelity metrics:")
    fidelity = compute_kernel_fidelity(K_noisy, K_ideal)
    print(f"   Frobenius error: {fidelity['frobenius_error']:.4f}")
    print(f"   Mean abs error:  {fidelity['mean_abs_error']:.4f}")
    print(f"   Max abs error:   {fidelity['max_abs_error']:.4f}")
    print(f"   Correlation:     {fidelity['correlation']:.4f}")
    print(f"   PSD?             {fidelity['is_psd']}")

    if fidelity["frobenius_error"] > 2.0:
        print("   WARN: Frobenius error very high")

    # 4. PSD analysis and correction
    print("\n4. PSD analysis and correction:")
    psd_info = analyze_psd_violation(K_noisy)
    print(f"   Is PSD: {psd_info['is_psd']}")
    print(f"   Min eigenvalue: {psd_info['min_eigenvalue']:.6f}")
    print(f"   Negative eigenvalues: {psd_info['n_negative_eigenvalues']}")
    print(f"   Correction needed: {psd_info['correction_needed']}")

    # Test all correction methods
    for method in ["clip", "nearest", "shift"]:
        K_corrected = project_to_psd(K_noisy, method=method)
        corrected_info = analyze_psd_violation(K_corrected)
        status = "PASS" if corrected_info["is_psd"] else "FAIL"
        if not corrected_info["is_psd"]:
            all_passed = False
        print(f"   {method}: PSD after correction? {status}")

    # 5. ZNE on a single kernel entry
    print("\n5. Zero-noise extrapolation (single entry):")
    noise_params = {
        "single_qubit_error": 0.005,
        "two_qubit_error": 0.02,
        "readout_error": 0.01,
    }

    # Pick two data points
    x1, x2 = X[0], X[1]
    ideal_val = float(K_ideal[0, 1])

    zne_result = zero_noise_extrapolation(
        fm, x1, x2, noise_params,
        scale_factors=[1.0, 1.5, 2.0],
        shots=2048,
        extrapolation="linear",
    )

    zne_val = zne_result["zne_estimate"]
    raw_val = zne_result["raw_estimates"][0]

    print(f"   Ideal value:   {ideal_val:.4f}")
    print(f"   Raw (noisy):   {raw_val:.4f}")
    print(f"   ZNE estimate:  {zne_val:.4f}")
    print(f"   Raw error:     {abs(raw_val - ideal_val):.4f}")
    print(f"   ZNE error:     {abs(zne_val - ideal_val):.4f}")
    print(f"   Fit R²:        {zne_result['fit_info']['r_squared']:.4f}")

    # 6. LocalNoiseRunner verification
    print("\n6. LocalNoiseRunner verification:")
    runner = LocalNoiseRunner(noise_model=noise_model)
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    k_val = runner.estimate_kernel_entry(qc, shots=2048)
    print(f"   Identity circuit kernel value: {k_val:.4f} (should be ~1.0)")
    if k_val < 0.85:
        print("   WARN: Identity circuit value too low")
        all_passed = False

    # 7. QuantumKernel with noise model
    print("\n7. QuantumKernel with noise_model:")
    qk_noisy = QuantumKernel(fm, backend="sampler", n_shots=2048, noise_model=noise_model)
    print(f"   Name: {qk_noisy.name}")
    K_qk_noisy = qk_noisy.compute_matrix(X[:3], show_progress=False)
    print(f"   3x3 noisy matrix computed, shape: {K_qk_noisy.shape}")
    print(f"   Symmetric? {np.allclose(K_qk_noisy, K_qk_noisy.T)}")

    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Elapsed time: {elapsed:.1f}s")
    if all_passed:
        print("PHASE 5 VERIFICATION PASSED")
    else:
        print("PHASE 5 VERIFICATION FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
