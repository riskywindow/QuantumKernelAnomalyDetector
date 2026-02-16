"""Phase 4 verification script.

Quick verification of all Phase 4 components:
- Kernel target alignment
- Effective dimension / eigenspectrum
- Geometric difference
- Synthetic advantage dataset

Target: under 90 seconds.

Usage:
    python -m scripts.verify_phase4
"""

from __future__ import annotations

import time

import numpy as np

from src.analysis.alignment import (
    compute_centered_kernel_target_alignment,
    compute_kernel_target_alignment,
)
from src.analysis.expressibility import compute_eigenspectrum
from src.analysis.geometric import compute_bidirectional_geometric_difference
from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.synthetic import generate_quantum_advantage_dataset
from src.data.transforms import QuantumPreprocessor
from src.kernels.classical import RBFKernel
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


def main() -> None:
    """Run Phase 4 verification."""
    total_start = time.time()

    print("=" * 60)
    print("PHASE 4 VERIFICATION")
    print("=" * 60)

    n_qubits = 4
    n_samples = 30

    # ================================================================
    # Step 1: Load data and compute kernel matrices
    # ================================================================
    print("\n[1] Loading data and computing kernel matrices...")
    t0 = time.time()
    data_dir = __import__("pathlib").Path("data/raw")
    X_full, y_full = load_credit_card_fraud(data_dir)
    _, X_test, y_test = prepare_anomaly_split(
        X_full, y_full, normal_train_size=200, test_size=n_samples, seed=42
    )

    preprocessor = QuantumPreprocessor(n_features=n_qubits)
    X_train_dummy = prepare_anomaly_split(
        X_full, y_full, normal_train_size=200, test_size=n_samples, seed=42
    )[0]
    preprocessor.fit_transform(X_train_dummy)
    X_q = preprocessor.transform(X_test)

    zz_fm = ZZFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear")
    qk = QuantumKernel(zz_fm, backend="statevector")
    K_zz = qk.compute_matrix(X_q, show_progress=False)

    rbf = RBFKernel(gamma="scale")
    K_rbf = rbf.compute_matrix(X_q)
    print(f"  Done ({time.time()-t0:.1f}s)")
    print(f"  K_zz: {K_zz.shape}, K_rbf: {K_rbf.shape}")

    # ================================================================
    # Step 2: Kernel Target Alignment
    # ================================================================
    print("\n[2] Kernel Target Alignment...")
    kta_zz = compute_kernel_target_alignment(K_zz, y_test)
    kta_rbf = compute_kernel_target_alignment(K_rbf, y_test)
    ckta_zz = compute_centered_kernel_target_alignment(K_zz, y_test)
    ckta_rbf = compute_centered_kernel_target_alignment(K_rbf, y_test)

    print(f"  ZZ:  KTA={kta_zz:.4f}, centered KTA={ckta_zz:.4f}")
    print(f"  RBF: KTA={kta_rbf:.4f}, centered KTA={ckta_rbf:.4f}")

    assert np.isfinite(kta_zz), "ZZ KTA is not finite!"
    assert np.isfinite(kta_rbf), "RBF KTA is not finite!"
    assert -1.0 <= ckta_zz <= 1.0, f"ZZ centered KTA out of bounds: {ckta_zz}"
    assert -1.0 <= ckta_rbf <= 1.0, f"RBF centered KTA out of bounds: {ckta_rbf}"
    print("  PASS: All KTA values finite and bounded")

    # ================================================================
    # Step 3: Eigenspectrum Analysis
    # ================================================================
    print("\n[3] Eigenspectrum Analysis...")
    spec_zz = compute_eigenspectrum(K_zz)
    spec_rbf = compute_eigenspectrum(K_rbf)

    print(f"  ZZ:  eff_dim={spec_zz['effective_dimension']:.2f}, "
          f"PR={spec_zz['participation_ratio']:.2f}")
    print(f"  RBF: eff_dim={spec_rbf['effective_dimension']:.2f}, "
          f"PR={spec_rbf['participation_ratio']:.2f}")

    assert spec_zz["effective_dimension"] >= 1.0
    assert spec_rbf["effective_dimension"] >= 1.0
    assert spec_zz["participation_ratio"] >= 1.0
    assert spec_rbf["participation_ratio"] >= 1.0
    norm_eigvals = spec_zz["normalized_eigenvalues"]
    assert abs(norm_eigvals.sum() - 1.0) < 1e-8, "Normalized eigenvalues don't sum to 1"
    print("  PASS: Eigenspectrum analysis valid")

    # ================================================================
    # Step 4: Geometric Difference
    # ================================================================
    print("\n[4] Geometric Difference...")
    geo = compute_bidirectional_geometric_difference(K_zz, K_rbf)
    print(f"  g(K_ZZ, K_RBF) = {geo['g_q_over_c']:.4f}")
    print(f"  g(K_RBF, K_ZZ) = {geo['g_c_over_q']:.4f}")
    print(f"  Advantage ratio = {geo['advantage_ratio']:.4f}")

    assert geo["g_q_over_c"] >= 1.0, "g_q_over_c < 1.0"
    assert geo["g_c_over_q"] >= 1.0, "g_c_over_q < 1.0"
    assert np.isfinite(geo["g_q_over_c"]), "g_q_over_c is not finite"
    assert np.isfinite(geo["g_c_over_q"]), "g_c_over_q is not finite"
    print("  PASS: Geometric difference valid")

    # ================================================================
    # Step 5: Synthetic Advantage Dataset
    # ================================================================
    print("\n[5] Synthetic Advantage Dataset...")
    t0 = time.time()
    zz_3 = ZZFeatureMap(n_qubits=3, reps=1, entanglement="linear")
    X_syn, y_syn = generate_quantum_advantage_dataset(
        zz_3, n_samples=100, n_features=3, noise_rate=0.05, seed=42
    )
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  Shape: X={X_syn.shape}, y={y_syn.shape}")
    print(f"  Label balance: {y_syn.mean():.2f} (target ~0.5)")

    assert X_syn.shape == (100, 3)
    assert y_syn.shape == (100,)
    assert set(np.unique(y_syn)).issubset({0.0, 1.0})
    assert 0.2 <= y_syn.mean() <= 0.8, f"Labels too imbalanced: {y_syn.mean():.2f}"

    # Verify that ZZ kernel has higher alignment with labels than RBF
    qk_syn = QuantumKernel(zz_3, backend="statevector")
    K_syn_zz = qk_syn.compute_matrix(X_syn, show_progress=False)
    rbf_syn = RBFKernel(gamma="scale")
    K_syn_rbf = rbf_syn.compute_matrix(X_syn)

    kta_syn_zz = compute_centered_kernel_target_alignment(K_syn_zz, y_syn)
    kta_syn_rbf = compute_centered_kernel_target_alignment(K_syn_rbf, y_syn)
    print(f"  ZZ  centered KTA on synthetic: {kta_syn_zz:.4f}")
    print(f"  RBF centered KTA on synthetic: {kta_syn_rbf:.4f}")
    print(f"  ZZ > RBF: {kta_syn_zz > kta_syn_rbf}")
    print("  PASS: Synthetic dataset valid")

    # ================================================================
    # Summary
    # ================================================================
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"ALL CHECKS PASSED ({total_elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
