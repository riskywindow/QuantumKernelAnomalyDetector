"""Phase 1 end-to-end verification script.

Loads the credit card fraud dataset, preprocesses to 5 features,
computes a 20x20 kernel matrix using the ZZ feature map with
statevector simulation, validates all kernel properties, and
saves a heatmap visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.transforms import QuantumPreprocessor
from src.kernels.base import validate_kernel_matrix
from src.kernels.estimation import KernelEstimator
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


def main() -> None:
    """Run the Phase 1 verification pipeline."""
    print("=" * 60)
    print("Phase 1 Verification: Quantum Kernel Anomaly Detection")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading credit card fraud dataset...")
    data_dir = Path("data/raw")
    X, y = load_credit_card_fraud(data_dir)
    print(f"  Dataset: {X.shape[0]} transactions, {X.shape[1]} features")
    print(f"  Fraud: {(y == 1).sum()} ({(y == 1).mean() * 100:.3f}%)")
    print(f"  Normal: {(y == 0).sum()} ({(y == 0).mean() * 100:.3f}%)")

    # Step 2: Prepare anomaly split
    print("\n[2/5] Preparing train/test split...")
    X_train, X_test, y_test = prepare_anomaly_split(
        X, y, normal_train_size=100, test_size=50, seed=42
    )
    print(f"  Train (normal only): {X_train.shape}")
    print(f"  Test (mixed): {X_test.shape}, anomalies: {(y_test == 1).sum()}")

    # Step 3: Preprocess to 5 features
    print("\n[3/5] Preprocessing to 5 features (StandardScale -> PCA -> [0, 2π])...")
    preprocessor = QuantumPreprocessor(n_features=5)
    X_train_q = preprocessor.fit_transform(X_train)
    print(f"  Preprocessed shape: {X_train_q.shape}")
    print(f"  Value range: [{X_train_q.min():.4f}, {X_train_q.max():.4f}]")

    # Take 20 samples for kernel computation
    X_20 = X_train_q[:20]
    print(f"  Using first 20 samples for kernel matrix")

    # Step 4: Compute kernel matrix
    print("\n[4/5] Computing 20x20 kernel matrix (ZZ feature map, 5 qubits, statevector)...")
    feature_map = ZZFeatureMap(n_qubits=5, reps=2, entanglement="linear")
    kernel = QuantumKernel(feature_map, backend="statevector")

    # Use the estimator with caching
    estimator = KernelEstimator()
    K = estimator.estimate(kernel, X_20, cache_key="phase1_verification")
    print(f"  Kernel matrix shape: {K.shape}")

    # Step 5: Validate kernel matrix
    print("\n[5/5] Validating kernel matrix properties...")
    results = validate_kernel_matrix(K)
    for prop, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {prop}")

    # Print some example values
    print("\nExample kernel values:")
    print(f"  K[0,0] = {K[0, 0]:.10f}  (should be 1.0)")
    print(f"  K[0,1] = {K[0, 1]:.10f}")
    print(f"  K[0,2] = {K[0, 2]:.10f}")
    print(f"  K[1,2] = {K[1, 2]:.10f}")

    eigvals = np.linalg.eigvalsh(K)
    print(f"\n  Min eigenvalue: {eigvals.min():.10f}")
    print(f"  Max eigenvalue: {eigvals.max():.10f}")
    print(f"  Mean off-diagonal: {(K.sum() - np.trace(K)) / (20 * 19):.6f}")

    # Save heatmap
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase1_kernel_heatmap.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        K,
        vmin=0,
        vmax=1,
        cmap="viridis",
        square=True,
        annot=False,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Quantum Kernel Matrix (ZZ Feature Map, 5 qubits, 2 reps)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")

    # Final verdict
    all_pass = all(results.values())
    print("\n" + "=" * 60)
    if all_pass:
        print("Phase 1 Verification: ALL CHECKS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Phase 1 Verification: FAILED checks: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
