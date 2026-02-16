"""Synthetic advantage benchmark: demonstrates quantum kernel advantage.

Generates a dataset where labels are determined by the quantum kernel's
structure, then benchmarks quantum vs classical kernels on classification.
The quantum kernel matching the feature map used to generate labels should
significantly outperform classical kernels.

Usage:
    python -m experiments.run_synthetic_benchmark
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

from src.analysis.alignment import compute_centered_kernel_target_alignment
from src.analysis.geometric import compute_geometric_difference
from src.data.synthetic import generate_quantum_advantage_split
from src.kernels.classical import RBFKernel, PolynomialKernel
from src.kernels.feature_maps.covariant import CovariantFeatureMap
from src.kernels.feature_maps.hardware_efficient import HardwareEfficientFeatureMap
from src.kernels.feature_maps.iqp import IQPFeatureMap
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


def main() -> None:
    """Run the synthetic advantage benchmark."""
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_qubits = 4
    n_train = 150
    n_test = 100
    noise_rate = 0.05
    seed = 42

    print("=" * 70)
    print("SYNTHETIC ADVANTAGE BENCHMARK")
    print("=" * 70)
    print(f"  n_qubits={n_qubits}, n_train={n_train}, n_test={n_test}")
    print(f"  noise_rate={noise_rate}, seed={seed}")

    total_start = time.time()

    # Generate dataset using ZZ feature map
    print("\n[Data] Generating synthetic advantage dataset (ZZ feature map)...")
    t0 = time.time()
    zz_fm = ZZFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear")
    X_train, X_test, y_train, y_test = generate_quantum_advantage_split(
        feature_map=zz_fm,
        n_train=n_train,
        n_test=n_test,
        n_features=n_qubits,
        noise_rate=noise_rate,
        seed=seed,
    )
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  Train: {len(X_train)} ({int(y_train.sum())} positive)")
    print(f"  Test:  {len(X_test)} ({int(y_test.sum())} positive)")

    # Build all feature maps
    feature_maps = {
        "ZZ": ZZFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear"),
        "IQP": IQPFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear"),
        "Covariant": CovariantFeatureMap(n_qubits=n_qubits, reps=2),
        "HW-Efficient": HardwareEfficientFeatureMap(n_qubits=n_qubits, reps=2),
    }

    # Compute quantum kernel matrices
    print("\n[Kernels] Computing kernel matrices...")
    kernel_matrices_train: dict[str, np.ndarray] = {}
    kernel_matrices_test: dict[str, np.ndarray] = {}

    for name, fm in feature_maps.items():
        t0 = time.time()
        print(f"  {name}...", end=" ", flush=True)
        qk = QuantumKernel(fm, backend="statevector")
        K_train = qk.compute_matrix(X_train, show_progress=False)
        K_test = qk.compute_matrix(X_test, X_train, show_progress=False)
        kernel_matrices_train[name] = K_train
        kernel_matrices_test[name] = K_test
        print(f"done ({time.time()-t0:.1f}s)")

    # Compute classical kernel matrices
    for name, kernel in [("RBF", RBFKernel(gamma="scale")), ("Polynomial", PolynomialKernel(degree=3, gamma="scale"))]:
        t0 = time.time()
        print(f"  {name}...", end=" ", flush=True)
        K_train = kernel.compute_matrix(X_train)
        K_test = kernel.compute_matrix(X_test, X_train)
        kernel_matrices_train[name] = K_train
        kernel_matrices_test[name] = K_test
        print(f"done ({time.time()-t0:.1f}s)")

    # Train SVC with each kernel
    print("\n[Classification] Training SVC models...")
    results_list = []
    for name in kernel_matrices_train:
        K_train = kernel_matrices_train[name]
        K_test = kernel_matrices_test[name]

        # SVC with precomputed kernel
        svc = SVC(kernel="precomputed", random_state=seed)
        svc.fit(K_train, y_train)
        y_pred = svc.predict(K_test)
        y_scores = svc.decision_function(K_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auroc = roc_auc_score(y_test, y_scores)

        results_list.append({
            "Kernel": name,
            "Accuracy": acc,
            "F1": f1,
            "AUROC": auroc,
        })
        print(f"  {name}: Accuracy={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")

    results_df = pd.DataFrame(results_list).sort_values("Accuracy", ascending=False)

    # Compute geometric differences vs RBF
    print("\n[Analysis] Computing geometric differences vs RBF...")
    K_rbf = kernel_matrices_train["RBF"]
    geo_diffs = []
    for name in ["ZZ", "IQP", "Covariant", "HW-Efficient"]:
        K_q = kernel_matrices_train[name]
        g_q_over_c = compute_geometric_difference(K_q, K_rbf)
        g_c_over_q = compute_geometric_difference(K_rbf, K_q)
        geo_diffs.append({
            "Kernel": name,
            "g(K_Q, K_RBF)": g_q_over_c,
            "g(K_RBF, K_Q)": g_c_over_q,
        })
        print(f"  {name}: g(K_Q, K_RBF)={g_q_over_c:.4f}, g(K_RBF, K_Q)={g_c_over_q:.4f}")

    geo_df = pd.DataFrame(geo_diffs)

    # Compute KTA for each kernel
    print("\n[Analysis] Computing kernel target alignment...")
    kta_scores = {}
    for name in kernel_matrices_train:
        kta = compute_centered_kernel_target_alignment(
            kernel_matrices_train[name], y_train
        )
        kta_scores[name] = kta
        print(f"  {name}: centered KTA = {kta:.4f}")

    results_df["KTA"] = results_df["Kernel"].map(kta_scores)

    # Save results
    csv_path = output_dir / "phase4_synthetic_benchmark.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved results to {csv_path}")

    # Generate plots
    print("\n[Plots] Generating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: Accuracy by kernel type
    ax = axes[0]
    colors = ["#2196F3" if k in ["ZZ", "IQP", "Covariant", "HW-Efficient"]
              else "#FF9800" for k in results_df["Kernel"]]
    bars = ax.barh(results_df["Kernel"], results_df["Accuracy"], color=colors)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy by Kernel\n(Synthetic Advantage Dataset)", fontsize=13)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(fontsize=10)
    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.1)

    # Scatter: KTA vs Accuracy
    ax = axes[1]
    for _, row in results_df.iterrows():
        is_quantum = row["Kernel"] in ["ZZ", "IQP", "Covariant", "HW-Efficient"]
        color = "#2196F3" if is_quantum else "#FF9800"
        marker = "o" if is_quantum else "s"
        ax.scatter(row["KTA"], row["Accuracy"], c=color, marker=marker, s=100, zorder=5)
        ax.annotate(row["Kernel"], (row["KTA"], row["Accuracy"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Centered KTA", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Kernel-Target Alignment vs Accuracy", fontsize=13)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_path = output_dir / "phase4_synthetic_results.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nGeometric Differences:")
    print(geo_df.to_string(index=False))
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
