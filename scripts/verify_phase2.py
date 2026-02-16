"""Phase 2 end-to-end verification script.

Loads credit card fraud data, preprocesses to 5 features, computes 20x20
kernel matrices for all six kernels (ZZ, IQP, Covariant, HW-Efficient,
RBF, Polynomial), compares statevector vs sampler, and produces
comparison heatmaps and shot noise analysis plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.transforms import QuantumPreprocessor
from src.kernels.base import validate_kernel_matrix
from src.kernels.classical import PolynomialKernel, RBFKernel
from src.kernels.feature_maps.covariant import CovariantFeatureMap
from src.kernels.feature_maps.hardware_efficient import HardwareEfficientFeatureMap
from src.kernels.feature_maps.iqp import IQPFeatureMap
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


def main() -> None:
    """Run the Phase 2 verification pipeline."""
    print("=" * 70)
    print("Phase 2 Verification: Kernel Engine Expansion")
    print("=" * 70)

    # Step 1: Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    data_dir = Path("data/raw")
    X, y = load_credit_card_fraud(data_dir)
    X_train, X_test, y_test = prepare_anomaly_split(
        X, y, normal_train_size=100, test_size=50, seed=42
    )
    preprocessor = QuantumPreprocessor(n_features=5)
    X_train_q = preprocessor.fit_transform(X_train)
    X_20 = X_train_q[:20]
    print(f"  Using {len(X_20)} samples, {X_20.shape[1]} features")

    # Step 2: Define all kernels
    print("\n[2/5] Computing kernel matrices for all 6 kernels...")

    n_qubits = 5
    quantum_feature_maps = [
        ("ZZ", ZZFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear")),
        ("IQP", IQPFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear")),
        ("Covariant", CovariantFeatureMap(n_qubits=n_qubits, reps=2)),
        ("HW-Efficient", HardwareEfficientFeatureMap(n_qubits=n_qubits, reps=2)),
    ]

    quantum_kernels = {}
    for name, fm in quantum_feature_maps:
        print(f"  Computing {name} kernel matrix...", end=" ", flush=True)
        qk = QuantumKernel(fm, backend="statevector")
        K = qk.compute_matrix(X_20, show_progress=False)
        quantum_kernels[name] = K
        print("done.")

    classical_kernels = {}
    print("  Computing RBF kernel matrix...", end=" ", flush=True)
    rbf = RBFKernel(gamma="scale")
    classical_kernels["RBF"] = rbf.compute_matrix(X_20)
    print("done.")

    print("  Computing Polynomial kernel matrix...", end=" ", flush=True)
    poly = PolynomialKernel(degree=3, gamma="scale")
    classical_kernels["Polynomial"] = poly.compute_matrix(X_20)
    print("done.")

    all_kernels = {**quantum_kernels, **classical_kernels}

    # Step 3: Validate all kernel matrices
    print("\n[3/5] Validating kernel matrices...")
    print(f"  {'Kernel':<15} {'Symmetric':<12} {'PSD':<8} {'Diag=1':<10} {'[0,1]':<8} "
          f"{'Mean Off-Diag':<15} {'Min':<8} {'Max':<8}")
    print("  " + "-" * 90)

    for name, K in all_kernels.items():
        results = validate_kernel_matrix(K)
        off_diag_mask = ~np.eye(K.shape[0], dtype=bool)
        mean_off = K[off_diag_mask].mean()
        min_val = K.min()
        max_val = K.max()

        sym = "PASS" if results["symmetric"] else "FAIL"
        psd = "PASS" if results["positive_semidefinite"] else "FAIL"
        diag = "PASS" if results["unit_diagonal"] else "FAIL"
        bnd = "PASS" if results["bounded_0_1"] else "FAIL"

        print(f"  {name:<15} {sym:<12} {psd:<8} {diag:<10} {bnd:<8} "
              f"{mean_off:<15.6f} {min_val:<8.4f} {max_val:<8.4f}")

    # Step 4: Shot noise analysis
    print("\n[4/5] Shot noise analysis (ZZ: statevector vs sampler)...")
    X_10 = X_20[:10]
    K_exact = quantum_kernels["ZZ"][:10, :10]

    fm_zz = ZZFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear")
    qk_sampler = QuantumKernel(fm_zz, backend="sampler", n_shots=1024)
    print("  Computing 10x10 sampler kernel matrix (1024 shots)...", end=" ", flush=True)
    K_sampler = qk_sampler.compute_matrix(X_10, show_progress=False)
    print("done.")

    # Compare statevector vs sampler
    off_diag = ~np.eye(10, dtype=bool)
    exact_vals = K_exact[off_diag]
    sampler_vals = K_sampler[off_diag]
    abs_diff = np.abs(exact_vals - sampler_vals)

    print(f"  Statevector vs Sampler (1024 shots) - off-diagonal entries:")
    print(f"    Mean absolute difference: {abs_diff.mean():.4f}")
    print(f"    Max absolute difference:  {abs_diff.max():.4f}")
    print(f"    Correlation:              {np.corrcoef(exact_vals, sampler_vals)[0, 1]:.4f}")

    # Step 5: Save plots
    print("\n[5/5] Saving plots...")
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2x3 grid of kernel matrix heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    kernel_names = list(all_kernels.keys())

    for idx, (name, K) in enumerate(all_kernels.items()):
        ax = axes[idx // 3, idx % 3]

        # For quantum kernels, use [0, 1] range; for classical, auto-scale
        if name in quantum_kernels:
            sns.heatmap(K, vmin=0, vmax=1, cmap="viridis", square=True,
                        annot=False, ax=ax)
        else:
            sns.heatmap(K, cmap="viridis", square=True, annot=False, ax=ax)

        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Sample")

    fig.suptitle("Kernel Matrix Comparison (20x20, 5 qubits/features)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    comparison_path = output_dir / "phase2_kernel_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kernel comparison heatmaps: {comparison_path}")

    # Shot noise analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: statevector vs sampler
    ax = axes[0]
    ax.scatter(exact_vals, sampler_vals, alpha=0.6, s=20)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Statevector (exact)")
    ax.set_ylabel("Sampler (1024 shots)")
    ax.set_title("Kernel Values: Statevector vs Sampler")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # Histogram of absolute differences
    ax = axes[1]
    ax.hist(abs_diff, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(abs_diff.mean(), color="red", linestyle="--",
               label=f"Mean = {abs_diff.mean():.4f}")
    ax.set_xlabel("Absolute Difference")
    ax.set_ylabel("Count")
    ax.set_title("Shot Noise: |Statevector - Sampler|")
    ax.legend()

    fig.suptitle("Shot Noise Analysis (ZZ, 5 qubits, 1024 shots)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    noise_path = output_dir / "phase2_shot_noise.png"
    fig.savefig(noise_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Shot noise analysis plot: {noise_path}")

    # Final verdict
    all_quantum_pass = True
    for name, K in quantum_kernels.items():
        results = validate_kernel_matrix(K)
        if not all(results.values()):
            all_quantum_pass = False

    print("\n" + "=" * 70)
    if all_quantum_pass:
        print("Phase 2 Verification: ALL QUANTUM KERNEL CHECKS PASSED")
    else:
        print("Phase 2 Verification: SOME CHECKS FAILED")
    print(f"  Quantum kernels validated: {len(quantum_kernels)}")
    print(f"  Classical kernels computed: {len(classical_kernels)}")
    print(f"  Shot noise mean |diff|: {abs_diff.mean():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
