"""Full experiment runner for quantum kernel anomaly detection benchmark.

Orchestrates end-to-end: data loading, kernel computation, model training,
evaluation, and visualization. Configurable via YAML files.

Usage:
    python -m experiments.run_experiment --config configs/experiments/kernel_comparison.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.transforms import QuantumPreprocessor
from src.kernels.estimation import KernelEstimator
from src.kernels.factory import create_all_kernels
from src.models.baselines import AutoencoderBaseline, IsolationForestBaseline, LOFBaseline
from src.models.kpca import KernelPCAAnomalyDetector
from src.models.ocsvm import QuantumOCSVM
from src.utils.metrics import AnomalyMetrics, compute_metrics_table


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Experiment configuration dict.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["experiment"]


def load_and_preprocess(
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load credit card fraud data and preprocess.

    Args:
        config: Experiment config dict.

    Returns:
        Tuple of (X_train_q, X_test_q, y_test)
        where X_*_q are the angle-encoded feature arrays.
    """
    print("\n[Data] Loading credit card fraud dataset...")
    data_dir = Path("data/raw")
    X, y = load_credit_card_fraud(data_dir)
    print(f"  Dataset: {len(X)} samples, {int(y.sum())} frauds ({100*y.mean():.3f}%)")

    data_config = config["data"]
    n_train = data_config["n_train_samples"]
    n_test = data_config["n_test_samples"]
    seed = config["seed"]

    X_train, X_test, y_test = prepare_anomaly_split(
        X, y, normal_train_size=n_train, test_size=n_test, seed=seed
    )
    print(f"  Train (normal only): {len(X_train)}")
    print(f"  Test: {len(X_test)} ({int(y_test.sum())} anomalies, {int((y_test==0).sum())} normal)")

    n_features = data_config["n_features"]
    preprocessor = QuantumPreprocessor(n_features=n_features)
    X_train_q = preprocessor.fit_transform(X_train)
    X_test_q = preprocessor.transform(X_test)
    print(f"  Preprocessed to {n_features} features (angles in [0, 2pi])")

    return X_train_q, X_test_q, y_test


def compute_kernel_matrices(
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute train and test kernel matrices for all configured kernels.

    Args:
        config: Experiment config dict.
        X_train: Training features (angle-encoded).
        X_test: Test features (angle-encoded).

    Returns:
        Dict mapping kernel name to (K_train, K_test) tuples.
    """
    n_features = config["data"]["n_features"]
    kernels = create_all_kernels(config, n_qubits=n_features)
    estimator = KernelEstimator()

    kernel_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print(f"\n[Kernels] Computing kernel matrices for {len(kernels)} kernels...")
    for name, kernel in kernels.items():
        try:
            t0 = time.time()
            print(f"  {name}: computing K_train ({len(X_train)}x{len(X_train)})...", end=" ", flush=True)
            cache_key_train = f"{name}_train_{len(X_train)}"
            K_train = estimator.estimate(
                kernel, X_train, cache_key=cache_key_train
            )
            print(f"done ({time.time()-t0:.1f}s)")

            t0 = time.time()
            print(f"  {name}: computing K_test ({len(X_test)}x{len(X_train)})...", end=" ", flush=True)
            cache_key_test = f"{name}_test_{len(X_test)}x{len(X_train)}"
            K_test = estimator.estimate(
                kernel, X_test, X_train, cache_key=cache_key_test
            )
            print(f"done ({time.time()-t0:.1f}s)")

            kernel_matrices[name] = (K_train, K_test)
        except Exception as e:
            print(f"FAILED - {e}")

    return kernel_matrices


def train_and_evaluate(
    config: dict,
    kernel_matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, AnomalyMetrics]:
    """Train all models and evaluate them.

    Args:
        config: Experiment config dict.
        kernel_matrices: Dict of kernel name -> (K_train, K_test).
        X_train: Raw training features (for baselines).
        X_test: Raw test features (for baselines).
        y_test: Test labels.

    Returns:
        Dict mapping model name to AnomalyMetrics.
    """
    results: dict[str, AnomalyMetrics] = {}
    model_config = config.get("models", {})
    nu = model_config.get("ocsvm", {}).get("nu", 0.05)
    n_components = model_config.get("kpca", {}).get("n_components", 10)

    # Kernel-based models
    print("\n[Models] Training kernel-based models...")
    for kernel_name, (K_train, K_test) in kernel_matrices.items():
        # OCSVM
        try:
            t0 = time.time()
            ocsvm = QuantumOCSVM(nu=nu, kernel_name=kernel_name)
            ocsvm.fit(K_train)
            metrics = ocsvm.evaluate(K_test, y_test)
            results[f"OCSVM ({kernel_name})"] = metrics
            print(f"  OCSVM ({kernel_name}): AUROC={metrics.auroc:.4f} ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"  OCSVM ({kernel_name}): FAILED - {e}")

        # KPCA
        try:
            t0 = time.time()
            kpca = KernelPCAAnomalyDetector(n_components=n_components, kernel_name=kernel_name)
            kpca.fit(K_train)
            metrics = kpca.evaluate(K_test, y_test)
            results[f"KPCA ({kernel_name})"] = metrics
            print(f"  KPCA ({kernel_name}): AUROC={metrics.auroc:.4f} ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"  KPCA ({kernel_name}): FAILED - {e}")

    # Baselines
    print("\n[Baselines] Training baseline models...")
    baseline_config = config.get("baselines", {})
    seed = config["seed"]

    try:
        t0 = time.time()
        if_config = baseline_config.get("isolation_forest", {})
        iforest = IsolationForestBaseline(
            n_estimators=if_config.get("n_estimators", 100),
            contamination=if_config.get("contamination", 0.001),
            seed=seed,
        )
        iforest.fit(X_train)
        metrics = iforest.evaluate(X_test, y_test)
        results["Isolation Forest"] = metrics
        print(f"  Isolation Forest: AUROC={metrics.auroc:.4f} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  Isolation Forest: FAILED - {e}")

    try:
        t0 = time.time()
        ae_config = baseline_config.get("autoencoder", {})
        ae = AutoencoderBaseline(
            encoding_dim=ae_config.get("encoding_dim", 3),
            epochs=ae_config.get("epochs", 50),
            batch_size=ae_config.get("batch_size", 256),
            seed=seed,
        )
        ae.fit(X_train)
        metrics = ae.evaluate(X_test, y_test)
        results["Autoencoder"] = metrics
        print(f"  Autoencoder: AUROC={metrics.auroc:.4f} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  Autoencoder: FAILED - {e}")

    try:
        t0 = time.time()
        lof_config = baseline_config.get("lof", {})
        lof = LOFBaseline(
            n_neighbors=lof_config.get("n_neighbors", 20),
            contamination=lof_config.get("contamination", 0.001),
        )
        lof.fit(X_train)
        metrics = lof.evaluate(X_test, y_test)
        results["LOF"] = metrics
        print(f"  LOF: AUROC={metrics.auroc:.4f} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  LOF: FAILED - {e}")

    return results


def generate_plots(
    results: dict[str, AnomalyMetrics],
    output_dir: Path,
) -> None:
    """Generate results heatmap.

    Args:
        results: Dict mapping model names to AnomalyMetrics.
        output_dir: Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("  No results to plot.")
        return

    # Results heatmap
    table = compute_metrics_table(results)
    fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.6)))
    metric_cols = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR@95%Recall"]
    data = table[metric_cols].values.astype(float)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index, fontsize=10)

    # Annotate cells
    for i in range(len(table)):
        for j in range(len(metric_cols)):
            val = data[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Anomaly Detection Benchmark Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "phase3_results_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Results heatmap: {path}")


def main() -> None:
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description="Quantum kernel anomaly detection benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/kernel_comparison.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    experiment_name = config.get("name", "experiment")

    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print("=" * 70)

    total_start = time.time()

    # Step 1: Load and preprocess data
    X_train_q, X_test_q, y_test = load_and_preprocess(config)

    # Step 2: Compute kernel matrices
    kernel_matrices = compute_kernel_matrices(config, X_train_q, X_test_q)

    # Step 3: Train and evaluate all models
    results = train_and_evaluate(
        config, kernel_matrices, X_train_q, X_test_q, y_test
    )

    # Step 4: Display results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    if results:
        table = compute_metrics_table(results)
        print(table.to_string())

        # Save CSV
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "phase3_benchmark.csv"
        table.to_csv(csv_path)
        print(f"\nSaved to {csv_path}")

        # Generate plots
        print("\n[Plots] Generating visualizations...")
        generate_plots(results, output_dir)

        # Top 3 models
        sorted_results = sorted(results.items(), key=lambda x: x[1].auroc, reverse=True)
        print("\nTop 3 models by AUROC:")
        for i, (name, m) in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {name}: AUROC={m.auroc:.4f}, F1={m.f1:.4f}")

    total_elapsed = time.time() - total_start
    print(f"\nTotal experiment time: {total_elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
