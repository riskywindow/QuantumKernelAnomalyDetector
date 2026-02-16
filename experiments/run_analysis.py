"""Combined expressibility analysis on real credit card fraud data.

Runs the full analysis from Phase 4: kernel target alignment, effective
dimension / eigenspectrum, and geometric difference for all kernels.
Produces tables and plots for the writeup.

Usage:
    python -m experiments.run_analysis
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.alignment import compute_centered_kernel_target_alignment
from src.analysis.expressibility import compute_eigenspectrum
from src.analysis.geometric import (
    compute_geometric_difference,
    compute_pairwise_geometric_differences,
)
from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.transforms import QuantumPreprocessor
from src.kernels.classical import RBFKernel
from src.kernels.feature_maps.covariant import CovariantFeatureMap
from src.kernels.feature_maps.hardware_efficient import HardwareEfficientFeatureMap
from src.kernels.feature_maps.iqp import IQPFeatureMap
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel


# AUROC values from Phase 3 benchmark (best OCSVM result per kernel)
PHASE3_AUROC = {
    "ZZ": 0.8744,
    "IQP": 0.8178,
    "Covariant": 0.8756,
    "HW-Efficient": 0.8978,
    "RBF": 0.9078,
}


def load_or_compute_kernels(
    X: np.ndarray,
    n_qubits: int,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    """Load cached kernel matrices or compute fresh ones.

    Tries to load from cache first. If not found, computes and saves.

    Args:
        X: Data matrix of shape (n, n_features).
        n_qubits: Number of qubits / features.
        cache_dir: Directory for cached kernel matrices.

    Returns:
        Dict mapping kernel name to (n, n) kernel matrix.
    """
    kernel_matrices: dict[str, np.ndarray] = {}

    feature_maps = {
        "ZZ": ZZFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear"),
        "IQP": IQPFeatureMap(n_qubits=n_qubits, reps=2, entanglement="linear"),
        "Covariant": CovariantFeatureMap(n_qubits=n_qubits, reps=2),
        "HW-Efficient": HardwareEfficientFeatureMap(n_qubits=n_qubits, reps=2),
    }

    for name, fm in feature_maps.items():
        cache_path = cache_dir / f"analysis_{name}_{len(X)}.npy"
        if cache_path.exists():
            print(f"  {name}: loaded from cache")
            kernel_matrices[name] = np.load(cache_path)
        else:
            t0 = time.time()
            print(f"  {name}: computing...", end=" ", flush=True)
            qk = QuantumKernel(fm, backend="statevector")
            K = qk.compute_matrix(X, show_progress=False)
            np.save(cache_path, K)
            kernel_matrices[name] = K
            print(f"done ({time.time()-t0:.1f}s)")

    # RBF kernel
    cache_path = cache_dir / f"analysis_RBF_{len(X)}.npy"
    if cache_path.exists():
        print(f"  RBF: loaded from cache")
        kernel_matrices["RBF"] = np.load(cache_path)
    else:
        t0 = time.time()
        print(f"  RBF: computing...", end=" ", flush=True)
        rbf = RBFKernel(gamma="scale")
        K = rbf.compute_matrix(X)
        np.save(cache_path, K)
        kernel_matrices["RBF"] = K
        print(f"done ({time.time()-t0:.1f}s)")

    return kernel_matrices


def main() -> None:
    """Run the combined expressibility analysis."""
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = 5
    n_samples = 100
    seed = 42

    print("=" * 70)
    print("EXPRESSIBILITY ANALYSIS — CREDIT CARD FRAUD DATA")
    print("=" * 70)

    total_start = time.time()

    # Load and preprocess data
    print("\n[Data] Loading credit card fraud dataset...")
    data_dir = Path("data/raw")
    X_full, y_full = load_credit_card_fraud(data_dir)

    # Use test split (has both normal + anomaly for KTA)
    _, X_test, y_test = prepare_anomaly_split(
        X_full, y_full, normal_train_size=200, test_size=n_samples, seed=seed
    )
    preprocessor = QuantumPreprocessor(n_features=n_features)
    preprocessor.fit_transform(
        prepare_anomaly_split(X_full, y_full, normal_train_size=200, test_size=n_samples, seed=seed)[0]
    )
    X_test_q = preprocessor.transform(X_test)
    print(f"  Test set: {len(X_test_q)} samples ({int(y_test.sum())} anomalies)")

    # Compute kernel matrices
    print("\n[Kernels] Computing/loading kernel matrices...")
    kernel_matrices = load_or_compute_kernels(X_test_q, n_features, output_dir)

    kernel_names = ["ZZ", "IQP", "Covariant", "HW-Efficient", "RBF"]

    # ======================================================================
    # TABLE 1: Expressibility Metrics
    # ======================================================================
    print("\n[Analysis] Computing expressibility metrics...")
    expressibility_rows = []
    eigenspectra = {}
    for name in kernel_names:
        K = kernel_matrices[name]
        spec = compute_eigenspectrum(K)
        eigenspectra[name] = spec

        kta = compute_centered_kernel_target_alignment(K, y_test)

        expressibility_rows.append({
            "Kernel": name,
            "Eff. Dimension": spec["effective_dimension"],
            "Participation Ratio": spec["participation_ratio"],
            "KTA (centered)": kta,
        })
        print(f"  {name}: eff_dim={spec['effective_dimension']:.2f}, "
              f"PR={spec['participation_ratio']:.2f}, KTA={kta:.4f}")

    table1 = pd.DataFrame(expressibility_rows)
    print("\nTable 1: Expressibility Metrics")
    print(table1.to_string(index=False))

    # ======================================================================
    # TABLE 2: Geometric Differences vs RBF
    # ======================================================================
    print("\n[Analysis] Computing geometric differences vs RBF...")
    K_rbf = kernel_matrices["RBF"]
    geo_rows = []
    for name in ["ZZ", "IQP", "Covariant", "HW-Efficient"]:
        K_q = kernel_matrices[name]
        g_q_over_c = compute_geometric_difference(K_q, K_rbf)
        g_c_over_q = compute_geometric_difference(K_rbf, K_q)
        ratio = g_q_over_c / g_c_over_q if g_c_over_q > 1e-15 else float("inf")

        geo_rows.append({
            "Kernel": name,
            "g(K_Q, K_RBF)": g_q_over_c,
            "g(K_RBF, K_Q)": g_c_over_q,
            "Advantage Ratio": ratio,
        })
        print(f"  {name}: g(K_Q,K_RBF)={g_q_over_c:.4f}, "
              f"g(K_RBF,K_Q)={g_c_over_q:.4f}, ratio={ratio:.4f}")

    table2 = pd.DataFrame(geo_rows)
    print("\nTable 2: Geometric Differences vs RBF")
    print(table2.to_string(index=False))

    # ======================================================================
    # TABLE 3: Correlation Analysis
    # ======================================================================
    print("\n[Analysis] Computing correlation with Phase 3 AUROC...")
    # Build aligned arrays for correlation
    names_with_auroc = [n for n in kernel_names if n in PHASE3_AUROC]
    auroc_values = np.array([PHASE3_AUROC[n] for n in names_with_auroc])
    kta_values = np.array([
        next(r["KTA (centered)"] for r in expressibility_rows if r["Kernel"] == n)
        for n in names_with_auroc
    ])
    effdim_values = np.array([
        eigenspectra[n]["effective_dimension"] for n in names_with_auroc
    ])
    geodiff_values = np.array([
        next((r["g(K_Q, K_RBF)"] for r in geo_rows if r["Kernel"] == n), 1.0)
        for n in names_with_auroc
    ])

    corr_kta, pval_kta = stats.pearsonr(kta_values, auroc_values)
    corr_effdim, pval_effdim = stats.pearsonr(effdim_values, auroc_values)

    # Geometric diff only for quantum kernels
    q_names = [n for n in names_with_auroc if n != "RBF"]
    q_auroc = np.array([PHASE3_AUROC[n] for n in q_names])
    q_geodiff = np.array([
        next(r["g(K_Q, K_RBF)"] for r in geo_rows if r["Kernel"] == n)
        for n in q_names
    ])
    corr_geo, pval_geo = stats.pearsonr(q_geodiff, q_auroc) if len(q_names) > 2 else (float("nan"), float("nan"))

    corr_rows = [
        {"Metric Pair": "KTA vs AUROC", "Pearson r": corr_kta, "p-value": pval_kta},
        {"Metric Pair": "Eff. Dimension vs AUROC", "Pearson r": corr_effdim, "p-value": pval_effdim},
        {"Metric Pair": "Geometric Diff vs AUROC (quantum only)", "Pearson r": corr_geo, "p-value": pval_geo},
    ]
    table3 = pd.DataFrame(corr_rows)
    print("\nTable 3: Correlation Analysis")
    print(table3.to_string(index=False))

    # Save all tables
    all_tables = pd.concat([
        table1.assign(Table="Expressibility"),
        table2.assign(Table="Geometric Difference"),
    ], ignore_index=True)
    csv_path = output_dir / "phase4_analysis.csv"
    all_tables.to_csv(csv_path, index=False)

    corr_path = output_dir / "phase4_correlations.csv"
    table3.to_csv(corr_path, index=False)
    print(f"\n  Saved tables to {csv_path}")
    print(f"  Saved correlations to {corr_path}")

    # ======================================================================
    # PLOTS
    # ======================================================================
    print("\n[Plots] Generating visualizations...")

    # Plot 1: Eigenspectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"ZZ": "#1f77b4", "IQP": "#ff7f0e", "Covariant": "#2ca02c",
              "HW-Efficient": "#d62728", "RBF": "#9467bd"}
    for name in kernel_names:
        spec = eigenspectra[name]
        norm_eigvals = spec["normalized_eigenvalues"]
        # Only plot positive values on log scale
        mask = norm_eigvals > 1e-15
        ax.semilogy(range(1, mask.sum() + 1), norm_eigvals[mask],
                     "o-", label=f"{name} (d_eff={spec['effective_dimension']:.1f})",
                     color=colors.get(name, "gray"), markersize=4, alpha=0.8)

    ax.set_xlabel("Eigenvalue Index", fontsize=12)
    ax.set_ylabel("Normalized Eigenvalue (log scale)", fontsize=12)
    ax.set_title("Kernel Eigenspectra — Expressibility Analysis", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_eigenspectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eigenspectra plot")

    # Plot 2: KTA vs AUROC scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names_with_auroc):
        kta = kta_values[i]
        auroc = auroc_values[i]
        is_quantum = name != "RBF"
        color = "#2196F3" if is_quantum else "#FF9800"
        marker = "o" if is_quantum else "s"
        ax.scatter(kta, auroc, c=color, marker=marker, s=120, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(name, (kta, auroc), textcoords="offset points",
                    xytext=(8, 6), fontsize=11)

    # Regression line
    if len(kta_values) > 2:
        slope, intercept = np.polyfit(kta_values, auroc_values, 1)
        x_fit = np.linspace(kta_values.min() - 0.01, kta_values.max() + 0.01, 50)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.4,
                label=f"r={corr_kta:.3f}")

    ax.set_xlabel("Centered KTA", fontsize=12)
    ax.set_ylabel("AUROC (Phase 3)", fontsize=12)
    ax.set_title("Kernel-Target Alignment vs AUROC", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_kta_vs_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved KTA vs AUROC scatter")

    # Plot 3: Geometric difference heatmap
    print("  Computing pairwise geometric differences...")
    pairwise = compute_pairwise_geometric_differences(kernel_matrices)
    names = kernel_names
    n = len(names)
    geo_matrix = np.ones((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            geo_matrix[i, j] = pairwise[(ni, nj)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(geo_matrix, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=11, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11)
    for i in range(n):
        for j in range(n):
            val = geo_matrix[i, j]
            color = "white" if val > np.median(geo_matrix) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="g(K_row, K_col)")
    ax.set_title("Pairwise Geometric Difference\ng(K_target, K_approx)", fontsize=14)
    ax.set_xlabel("K_approx (approximating kernel)", fontsize=11)
    ax.set_ylabel("K_target (target kernel)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_geometric_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved geometric difference heatmap")

    total_elapsed = time.time() - total_start
    print(f"\nTotal analysis time: {total_elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
