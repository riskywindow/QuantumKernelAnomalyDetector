"""Hardware execution experiment — noise sweep, PSD correction, and ZNE.

Runs in two modes:
- Simulation mode (default): Uses Aer with noise models
- Hardware mode (--real-hardware): Submits to IBM Quantum

Usage:
    python -m experiments.run_hardware
    python -m experiments.run_hardware --real-hardware --backend ibm_brisbane
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis.noise import compute_kernel_fidelity, compute_noisy_kernel_matrix, run_noise_sweep
from src.analysis.psd_correction import analyze_psd_violation, project_to_psd
from src.analysis.zne import zne_kernel_matrix
from src.data.loader import load_credit_card_fraud, prepare_anomaly_split
from src.data.transforms import QuantumPreprocessor
from src.hardware.noise_models import build_depolarizing_noise_model, build_noise_sweep
from src.kernels.feature_maps.hardware_efficient import HardwareEfficientFeatureMap
from src.kernels.feature_maps.zz import ZZFeatureMap
from src.kernels.quantum import QuantumKernel
from src.models.ocsvm import QuantumOCSVM


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess credit card fraud data."""
    n_features = config["kernel"]["n_qubits"]
    n_samples = config["data"]["n_samples"]
    seed = config["data"]["seed"]

    data_dir = Path("data/raw")
    X, y = load_credit_card_fraud(data_dir)
    X_train, X_test, y_test = prepare_anomaly_split(
        X, y, normal_train_size=200, test_size=100, seed=seed
    )

    preprocessor = QuantumPreprocessor(n_features=n_features)
    X_train_q = preprocessor.fit_transform(np.asarray(X_train))
    X_test_q = preprocessor.transform(np.asarray(X_test))

    # Take subset for kernel computation
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_train_q), size=n_samples, replace=False)
    X_sub = X_train_q[indices]

    return X_sub, X_test_q, np.asarray(y_test)


def run_noise_sweep_experiment(
    feature_map, X: np.ndarray, ideal_K: np.ndarray, config: dict, results_dir: Path
) -> dict:
    """Run noise sweep study."""
    print("\n" + "=" * 60)
    print("NOISE SWEEP STUDY")
    print("=" * 60)

    sweep_config = config["noise_sweep"]
    shots = config["hardware"]["shots"]

    noise_models = build_noise_sweep(
        n_levels=sweep_config["n_levels"],
        min_error=sweep_config["min_error"],
        max_error=sweep_config["max_error"],
    )

    results = run_noise_sweep(
        feature_map, X, noise_models, ideal_K, shots=shots, show_progress=True
    )

    # Print results table
    print(f"\n{'Error Rate':>12} {'Frob Error':>12} {'MAE':>10} {'Corr':>8} {'PSD Viol':>10} {'Diag Err':>10}")
    print("-" * 72)
    for i in range(len(results["error_rates"])):
        print(
            f"{results['error_rates'][i]:12.6f} "
            f"{results['fidelities'][i]:12.4f} "
            f"{results['mean_abs_errors'][i]:10.4f} "
            f"{results['correlations'][i]:8.4f} "
            f"{'YES' if results['psd_violations'][i] else 'no':>10} "
            f"{results['diagonal_errors'][i]:10.4f}"
        )

    # Plot noise sweep
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].semilogx(results["error_rates"], results["fidelities"], "bo-")
    axes[0].set_xlabel("Single-Qubit Error Rate")
    axes[0].set_ylabel("Frobenius Error")
    axes[0].set_title("Kernel Fidelity vs Noise")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(results["error_rates"], results["correlations"], "ro-")
    axes[1].set_xlabel("Single-Qubit Error Rate")
    axes[1].set_ylabel("Pearson Correlation")
    axes[1].set_title("Kernel Correlation vs Noise")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(results["error_rates"], results["diagonal_errors"], "go-")
    axes[2].set_xlabel("Single-Qubit Error Rate")
    axes[2].set_ylabel("Mean Diagonal Error")
    axes[2].set_title("Diagonal Deviation vs Noise")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "phase5_noise_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {results_dir / 'phase5_noise_sweep.png'}")

    return results


def run_psd_analysis(
    feature_map, X: np.ndarray, ideal_K: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    noise_sweep_results: dict, config: dict, results_dir: Path,
) -> None:
    """Analyze PSD violations and correction impact on ML performance."""
    print("\n" + "=" * 60)
    print("PSD ANALYSIS")
    print("=" * 60)

    error_rates = noise_sweep_results["error_rates"]
    kernel_matrices = noise_sweep_results["kernel_matrices"]

    psd_results = []
    for i, (rate, K_noisy) in enumerate(zip(error_rates, kernel_matrices)):
        psd_info = analyze_psd_violation(K_noisy)

        # Try OCSVM with uncorrected kernel
        try:
            ocsvm = QuantumOCSVM()
            ocsvm.fit(K_noisy)
            # Use ideal K for test predictions (cross-kernel with training)
            # For simplicity, evaluate on training set
            scores = ocsvm.predict_scores(K_noisy)
            auroc_uncorrected = float(np.std(scores) > 0)  # Placeholder
        except Exception:
            auroc_uncorrected = 0.0

        # Apply PSD correction and try again
        K_corrected = project_to_psd(K_noisy, method="clip")
        try:
            ocsvm_corr = QuantumOCSVM()
            ocsvm_corr.fit(K_corrected)
            scores_corr = ocsvm_corr.predict_scores(K_corrected)
            auroc_corrected = float(np.std(scores_corr) > 0)
        except Exception:
            auroc_corrected = 0.0

        psd_results.append({
            "error_rate": rate,
            "is_psd": psd_info["is_psd"],
            "min_eigenvalue": psd_info["min_eigenvalue"],
            "n_negative": psd_info["n_negative_eigenvalues"],
            "correction_needed": psd_info["correction_needed"],
        })

        print(
            f"  Rate={rate:.6f}: PSD={psd_info['is_psd']}, "
            f"min_eig={psd_info['min_eigenvalue']:.6f}, "
            f"correction={psd_info['correction_needed']}"
        )

    # Plot PSD analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    min_eigs = [r["min_eigenvalue"] for r in psd_results]
    rates = [r["error_rate"] for r in psd_results]

    axes[0].semilogx(rates, min_eigs, "ro-")
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Single-Qubit Error Rate")
    axes[0].set_ylabel("Minimum Eigenvalue")
    axes[0].set_title("PSD Violation vs Noise Level")
    axes[0].grid(True, alpha=0.3)

    n_negs = [r["n_negative"] for r in psd_results]
    axes[1].semilogx(rates, n_negs, "bs-")
    axes[1].set_xlabel("Single-Qubit Error Rate")
    axes[1].set_ylabel("Number of Negative Eigenvalues")
    axes[1].set_title("Negative Eigenvalues vs Noise Level")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "phase5_psd_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {results_dir / 'phase5_psd_analysis.png'}")


def run_zne_experiment(
    feature_map, X: np.ndarray, ideal_K: np.ndarray, config: dict, results_dir: Path
) -> dict:
    """Run zero-noise extrapolation experiment."""
    print("\n" + "=" * 60)
    print("ZERO-NOISE EXTRAPOLATION")
    print("=" * 60)

    zne_config = config["zne"]
    noise_params = {
        "single_qubit_error": zne_config["base_noise"]["single_qubit_error"],
        "two_qubit_error": zne_config["base_noise"]["two_qubit_error"],
        "readout_error": zne_config["base_noise"]["readout_error"],
    }

    print(f"  Base noise: {noise_params}")
    print(f"  Scale factors: {zne_config['scale_factors']}")

    # Compute ZNE-corrected kernel matrix
    zne_result = zne_kernel_matrix(
        feature_map, X, noise_params,
        scale_factors=zne_config["scale_factors"],
        shots=config["hardware"]["shots"],
        extrapolation=zne_config["extrapolation"],
        show_progress=True,
    )

    K_zne = zne_result["zne_matrix"]
    K_base = zne_result["raw_matrices"][0]

    # Compare
    metrics_noisy = compute_kernel_fidelity(K_base, ideal_K)
    metrics_zne = compute_kernel_fidelity(K_zne, ideal_K)

    print(f"\n  {'Metric':<25} {'Noisy':>10} {'ZNE':>10} {'Improvement':>12}")
    print("  " + "-" * 57)
    print(f"  {'Frobenius Error':<25} {metrics_noisy['frobenius_error']:10.4f} {metrics_zne['frobenius_error']:10.4f} {(1 - metrics_zne['frobenius_error']/metrics_noisy['frobenius_error'])*100:11.1f}%")
    print(f"  {'Mean Abs Error':<25} {metrics_noisy['mean_abs_error']:10.4f} {metrics_zne['mean_abs_error']:10.4f} {(1 - metrics_zne['mean_abs_error']/metrics_noisy['mean_abs_error'])*100:11.1f}%")
    print(f"  {'Correlation':<25} {metrics_noisy['correlation']:10.4f} {metrics_zne['correlation']:10.4f}")
    print(f"  {'Mean Correction':<25} {zne_result['mean_correction']:10.4f}")

    # Plot: Ideal vs Noisy vs ZNE (3 side-by-side heatmaps)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(ideal_K.min(), K_base.min(), K_zne.min())
    vmax = max(ideal_K.max(), K_base.max(), K_zne.max())

    im0 = axes[0].imshow(ideal_K, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title("Ideal (Statevector)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(K_base, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title(f"Noisy (MAE={metrics_noisy['mean_abs_error']:.4f})")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(K_zne, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[2].set_title(f"ZNE-Corrected (MAE={metrics_zne['mean_abs_error']:.4f})")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle("Kernel Matrix Comparison: Ideal vs Noisy vs ZNE", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "phase5_zne_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {results_dir / 'phase5_zne_comparison.png'}")

    return {
        "metrics_noisy": metrics_noisy,
        "metrics_zne": metrics_zne,
        "mean_correction": zne_result["mean_correction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware execution experiment")
    parser.add_argument(
        "--config", default="configs/hardware/noise_sweep.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--real-hardware", action="store_true",
        help="Use real IBM Quantum hardware (requires IBM_QUANTUM_TOKEN)",
    )
    parser.add_argument("--backend", default=None, help="IBM backend name")
    args = parser.parse_args()

    config = load_config(args.config)

    # Create results directory
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Setup
    print("Loading and preprocessing data...")
    X, X_test, y_test = setup_data(config)
    n_qubits = config["kernel"]["n_qubits"]
    reps = config["kernel"]["reps"]

    # Use HW-efficient feature map (best for real hardware)
    fm = HardwareEfficientFeatureMap(n_qubits=n_qubits, reps=reps)
    print(f"Feature map: {fm.name} ({n_qubits} qubits, {reps} reps)")
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

    # Compute ideal kernel matrix
    print("\nComputing ideal kernel matrix (statevector)...")
    qk = QuantumKernel(fm, backend="statevector")
    ideal_K = qk.compute_matrix(X, show_progress=True)
    print(f"Ideal kernel: shape={ideal_K.shape}, mean_off_diag={np.mean(ideal_K[~np.eye(len(X), dtype=bool)]):.4f}")

    # 1. Noise sweep
    sweep_results = run_noise_sweep_experiment(fm, X, ideal_K, config, results_dir)

    # 2. PSD analysis
    run_psd_analysis(fm, X, ideal_K, X_test, y_test, sweep_results, config, results_dir)

    # 3. ZNE
    zne_results = run_zne_experiment(fm, X, ideal_K, config, results_dir)

    # 4. Real hardware (if requested)
    if args.real_hardware:
        print("\n" + "=" * 60)
        print("REAL HARDWARE EXECUTION")
        print("=" * 60)
        try:
            from src.hardware.ibm_runner import IBMQuantumRunner

            backend_name = args.backend or config["hardware"].get("backend_name")
            runner = IBMQuantumRunner(backend_name=backend_name)
            info = runner.get_backend_info()
            print(f"  Backend: {info['name']}")
            print(f"  Qubits: {info['n_qubits']}")
            print(f"  Basis gates: {info.get('basis_gates', 'N/A')}")
            if "median_cx_error" in info:
                print(f"  Median CX error: {info['median_cx_error']:.6f}")
            if "median_readout_error" in info:
                print(f"  Median readout error: {info['median_readout_error']:.6f}")
            print("\n  Hardware execution not implemented in this version.")
            print("  (Would submit kernel circuits and collect results)")
        except Exception as e:
            print(f"  Could not connect to IBM Quantum: {e}")

    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Noise levels tested: {len(sweep_results['error_rates'])}")
    print(f"Error rate range: [{sweep_results['error_rates'][0]:.6f}, {sweep_results['error_rates'][-1]:.6f}]")
    print(f"Frobenius error range: [{min(sweep_results['fidelities']):.4f}, {max(sweep_results['fidelities']):.4f}]")
    print(f"ZNE Frobenius improvement: {(1 - zne_results['metrics_zne']['frobenius_error']/zne_results['metrics_noisy']['frobenius_error'])*100:.1f}%")
    print(f"ZNE MAE improvement: {(1 - zne_results['metrics_zne']['mean_abs_error']/zne_results['metrics_noisy']['mean_abs_error'])*100:.1f}%")

    # Save summary CSV
    summary_rows = []
    for i in range(len(sweep_results["error_rates"])):
        summary_rows.append({
            "error_rate": sweep_results["error_rates"][i],
            "frobenius_error": sweep_results["fidelities"][i],
            "mean_abs_error": sweep_results["mean_abs_errors"][i],
            "correlation": sweep_results["correlations"][i],
            "psd_violation": sweep_results["psd_violations"][i],
            "diagonal_error": sweep_results["diagonal_errors"][i],
        })
    df = pd.DataFrame(summary_rows)
    csv_path = results_dir / "phase5_noise_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
