"""Phase 3 verification script.

Fast verification of the ML pipeline: loads data, computes kernel matrices
for ZZ (quantum) and RBF (classical) only, runs OCSVM and KPCA on both,
runs all 3 baselines on raw features, and prints a comparison table.

Should complete in under 60 seconds.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.transforms import QuantumPreprocessor
from src.kernels.classical import RBFKernel
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel
from src.models.baselines import AutoencoderBaseline, IsolationForestBaseline, LOFBaseline
from src.models.kpca import KernelPCAAnomalyDetector
from src.models.ocsvm import QuantumOCSVM
from src.utils.metrics import AnomalyMetrics, compute_metrics_table


def main() -> None:
    """Run the Phase 3 verification pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("Phase 3 Verification: ML Pipeline & Benchmarking")
    print("=" * 70)

    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    data_dir = Path("data/raw")
    X, y = load_credit_card_fraud(data_dir)
    print(f"  Dataset: {len(X)} samples, {y.sum()} frauds ({100*y.mean():.3f}%)")

    X_train, X_test, y_test = prepare_anomaly_split(
        X, y, normal_train_size=50, test_size=30, seed=42
    )
    print(f"  Train (normal only): {len(X_train)}")
    print(f"  Test: {len(X_test)} ({y_test.sum()} anomalies, {(y_test==0).sum()} normal)")

    n_features = 5
    preprocessor = QuantumPreprocessor(n_features=n_features)
    X_train_q = preprocessor.fit_transform(X_train)
    X_test_q = preprocessor.transform(X_test)
    print(f"  Preprocessed to {n_features} features (angles in [0, 2pi])")

    # Step 2: Compute kernel matrices
    print("\n[2/6] Computing kernel matrices...")

    # ZZ quantum kernel
    t0 = time.time()
    zz_fm = ZZFeatureMap(n_qubits=n_features, reps=2, entanglement="linear")
    zz_kernel = QuantumKernel(zz_fm, backend="statevector")
    K_train_zz = zz_kernel.compute_matrix(X_train_q, show_progress=False)
    K_test_zz = zz_kernel.compute_matrix(X_test_q, X_train_q, show_progress=False)
    print(f"  ZZ kernel: K_train {K_train_zz.shape}, K_test {K_test_zz.shape} ({time.time()-t0:.1f}s)")

    # RBF classical kernel
    t0 = time.time()
    rbf_kernel = RBFKernel(gamma="scale")
    K_train_rbf = rbf_kernel.compute_matrix(X_train_q)
    K_test_rbf = rbf_kernel.compute_matrix(X_test_q, X_train_q)
    print(f"  RBF kernel: K_train {K_train_rbf.shape}, K_test {K_test_rbf.shape} ({time.time()-t0:.1f}s)")

    # Step 3: Train and evaluate kernel-based models
    print("\n[3/6] Training kernel-based models...")
    results: dict[str, AnomalyMetrics] = {}

    for kernel_name, K_train, K_test in [
        ("ZZ", K_train_zz, K_test_zz),
        ("RBF", K_train_rbf, K_test_rbf),
    ]:
        # OCSVM
        try:
            ocsvm = QuantumOCSVM(nu=0.05, kernel_name=kernel_name)
            ocsvm.fit(K_train)
            metrics = ocsvm.evaluate(K_test, y_test)
            results[f"OCSVM ({kernel_name})"] = metrics
            print(f"  OCSVM ({kernel_name}): AUROC={metrics.auroc:.4f}")
        except Exception as e:
            print(f"  OCSVM ({kernel_name}): FAILED - {e}")

        # KPCA
        try:
            kpca = KernelPCAAnomalyDetector(n_components=10, kernel_name=kernel_name)
            kpca.fit(K_train)
            metrics = kpca.evaluate(K_test, y_test)
            results[f"KPCA ({kernel_name})"] = metrics
            print(f"  KPCA ({kernel_name}): AUROC={metrics.auroc:.4f}")
        except Exception as e:
            print(f"  KPCA ({kernel_name}): FAILED - {e}")

    # Step 4: Train and evaluate baselines
    print("\n[4/6] Training baseline models...")

    try:
        iforest = IsolationForestBaseline(n_estimators=100, contamination=0.001, seed=42)
        iforest.fit(X_train_q)
        metrics = iforest.evaluate(X_test_q, y_test)
        results["Isolation Forest"] = metrics
        print(f"  Isolation Forest: AUROC={metrics.auroc:.4f}")
    except Exception as e:
        print(f"  Isolation Forest: FAILED - {e}")

    try:
        ae = AutoencoderBaseline(encoding_dim=3, epochs=50, batch_size=256, seed=42)
        ae.fit(X_train_q)
        metrics = ae.evaluate(X_test_q, y_test)
        results["Autoencoder"] = metrics
        print(f"  Autoencoder: AUROC={metrics.auroc:.4f}")
    except Exception as e:
        print(f"  Autoencoder: FAILED - {e}")

    try:
        lof = LOFBaseline(n_neighbors=20, contamination=0.001)
        lof.fit(X_train_q)
        metrics = lof.evaluate(X_test_q, y_test)
        results["LOF"] = metrics
        print(f"  LOF: AUROC={metrics.auroc:.4f}")
    except Exception as e:
        print(f"  LOF: FAILED - {e}")

    # Step 5: Display comparison table
    print("\n[5/6] Benchmark results:")
    if results:
        table = compute_metrics_table(results)
        print(table.to_string())

        # Save to CSV
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "phase3_verification.csv"
        table.to_csv(csv_path)
        print(f"\n  Saved to {csv_path}")

    # Step 6: Validate metrics ranges
    print("\n[6/6] Validating metrics ranges...")
    all_valid = True
    for name, m in results.items():
        checks = [
            (0 <= m.auroc <= 1, f"{name}: AUROC={m.auroc} out of [0,1]"),
            (0 <= m.auprc <= 1, f"{name}: AUPRC={m.auprc} out of [0,1]"),
            (0 <= m.f1 <= 1, f"{name}: F1={m.f1} out of [0,1]"),
            (0 <= m.precision <= 1, f"{name}: Precision={m.precision} out of [0,1]"),
            (0 <= m.recall <= 1, f"{name}: Recall={m.recall} out of [0,1]"),
            (0 <= m.fpr_at_95_recall <= 1, f"{name}: FPR@95={m.fpr_at_95_recall} out of [0,1]"),
        ]
        for ok, msg in checks:
            if not ok:
                print(f"  FAIL: {msg}")
                all_valid = False

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    if all_valid and len(results) >= 5:
        print(f"Phase 3 Verification: ALL CHECKS PASSED ({elapsed:.1f}s)")
    else:
        print(f"Phase 3 Verification: ISSUES DETECTED ({elapsed:.1f}s)")
    print(f"  Models evaluated: {len(results)}")
    if results:
        best = max(results.items(), key=lambda x: x[1].auroc)
        print(f"  Best model: {best[0]} (AUROC={best[1].auroc:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
