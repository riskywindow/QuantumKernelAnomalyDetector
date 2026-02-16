"""Export all experiment results to JSON for the React dashboard.

Reads cached CSV/NPY results from experiments/results/ and writes:
  - dashboard/public/data/benchmark_results.json
  - dashboard/public/data/kernel_matrices.json
  - dashboard/public/data/expressibility.json
  - dashboard/public/data/synthetic_benchmark.json
  - dashboard/public/data/noise_study.json
  - dashboard/data_bundle.js  (all data embedded as JS variables)

Usage:
    python -m dashboard.export_data [--recompute]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "public" / "data"
BUNDLE_PATH = Path(__file__).resolve().parent / "data_bundle.js"

KERNEL_NAMES = ["ZZ", "IQP", "Covariant", "HW-Efficient", "RBF", "Polynomial"]
QUANTUM_KERNELS = ["ZZ", "IQP", "Covariant", "HW-Efficient"]
HEATMAP_SIZE = 20  # Use 20x20 subset for dashboard heatmaps


def export_benchmark_results() -> dict | None:
    """Export Phase 3 benchmark results."""
    csv_path = RESULTS_DIR / "phase3_benchmark.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping benchmark results")
        return None

    df = pd.read_csv(csv_path)
    models = []
    for _, row in df.iterrows():
        name = row["Model"]
        # Determine kernel type
        if any(q in name for q in QUANTUM_KERNELS):
            if "KPCA" in name:
                kernel_type = "quantum_kpca"
            else:
                kernel_type = "quantum"
        elif any(c in name for c in ["RBF", "Polynomial"]):
            if "KPCA" in name:
                kernel_type = "classical_kpca"
            else:
                kernel_type = "classical"
        else:
            kernel_type = "baseline"

        models.append({
            "name": name,
            "kernel_type": kernel_type,
            "auroc": round(float(row["AUROC"]), 4),
            "auprc": round(float(row["AUPRC"]), 4),
            "f1": round(float(row["F1"]), 4),
            "precision": round(float(row["Precision"]), 4),
            "recall": round(float(row["Recall"]), 4),
            "fpr_at_95_recall": round(float(row["FPR@95%Recall"]), 4),
        })

    data = {"models": models}
    _write_json(OUTPUT_DIR / "benchmark_results.json", data)
    print(f"  Exported {len(models)} model results")
    return data


def export_kernel_matrices() -> dict | None:
    """Export kernel matrices (20x20 subsets) for heatmap visualization."""
    kernels: dict[str, dict] = {}
    found = 0

    for name in KERNEL_NAMES:
        npy_path = RESULTS_DIR / f"analysis_{name}_100.npy"
        if not npy_path.exists():
            print(f"  WARNING: {npy_path} not found, skipping {name}")
            continue

        K_full = np.load(npy_path)
        # Take 20x20 subset for visualization
        K = K_full[:HEATMAP_SIZE, :HEATMAP_SIZE]

        # Compute off-diagonal statistics
        mask = ~np.eye(K.shape[0], dtype=bool)
        off_diag = K[mask]

        kernels[name] = {
            "matrix": np.round(K, 6).tolist(),
            "stats": {
                "mean_off_diag": round(float(np.mean(off_diag)), 4),
                "std_off_diag": round(float(np.std(off_diag)), 4),
                "min_off_diag": round(float(np.min(off_diag)), 4),
                "max_off_diag": round(float(np.max(off_diag)), 4),
            },
        }
        found += 1

    if found == 0:
        print("  WARNING: No kernel matrices found")
        return None

    data = {
        "kernels": kernels,
        "sample_count": HEATMAP_SIZE,
        "n_qubits": 5,
    }
    _write_json(OUTPUT_DIR / "kernel_matrices.json", data)
    print(f"  Exported {found} kernel matrices ({HEATMAP_SIZE}x{HEATMAP_SIZE})")
    return data


def export_expressibility() -> dict | None:
    """Export Phase 4 expressibility analysis results."""
    csv_path = RESULTS_DIR / "phase4_analysis.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping expressibility")
        return None

    df = pd.read_csv(csv_path)

    # Parse expressibility table
    expr_rows = df[df["Table"] == "Expressibility"]
    metrics: dict[str, dict] = {}
    for _, row in expr_rows.iterrows():
        name = row["Kernel"]
        metrics[name] = {
            "effective_dimension": round(float(row["Eff. Dimension"]), 2),
            "participation_ratio": round(float(row["Participation Ratio"]), 2),
            "kta_centered": round(float(row["KTA (centered)"]), 4),
        }

    # Parse geometric differences
    geo_rows = df[df["Table"] == "Geometric Difference"]
    geometric_differences: dict[str, dict] = {}
    for _, row in geo_rows.iterrows():
        name = row["Kernel"]
        geometric_differences[f"{name}_vs_RBF"] = {
            "g_q_over_c": round(float(row["g(K_Q, K_RBF)"]), 2),
            "g_c_over_q": round(float(row["g(K_RBF, K_Q)"]), 2),
        }

    # Load eigenspectra from kernel matrices
    eigenspectra: dict[str, dict] = {}
    for name in KERNEL_NAMES:
        if name == "Polynomial":
            continue  # Skip polynomial (non-normalized)
        npy_path = RESULTS_DIR / f"analysis_{name}_100.npy"
        if not npy_path.exists():
            continue
        K = np.load(npy_path)
        eigenvalues = np.sort(np.linalg.eigvalsh(K))[::-1]
        # Normalize
        eigenvalues_pos = np.maximum(eigenvalues, 0)
        total = eigenvalues_pos.sum()
        if total > 0:
            normalized = eigenvalues_pos / total
        else:
            normalized = eigenvalues_pos
        cumulative = np.cumsum(normalized)

        eigenspectra[name] = {
            "eigenvalues": np.round(eigenvalues[:20], 6).tolist(),
            "normalized": np.round(normalized[:20], 6).tolist(),
            "cumulative_variance": np.round(cumulative[:20], 6).tolist(),
        }

    data = {
        "eigenspectra": eigenspectra,
        "metrics": metrics,
        "geometric_differences": geometric_differences,
    }
    _write_json(OUTPUT_DIR / "expressibility.json", data)
    print(f"  Exported expressibility for {len(metrics)} kernels")
    return data


def export_synthetic_benchmark() -> dict | None:
    """Export Phase 4 synthetic advantage results."""
    csv_path = RESULTS_DIR / "phase4_synthetic_benchmark.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping synthetic benchmark")
        return None

    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        results.append({
            "kernel": row["Kernel"],
            "accuracy": round(float(row["Accuracy"]), 3),
            "f1": round(float(row["F1"]), 3),
            "auroc": round(float(row["AUROC"]), 3),
            "kta": round(float(row["KTA"]), 4),
        })

    data = {"results": results}
    _write_json(OUTPUT_DIR / "synthetic_benchmark.json", data)
    print(f"  Exported {len(results)} synthetic benchmark results")
    return data


def export_noise_study() -> dict | None:
    """Export Phase 5 noise sweep and ZNE results."""
    csv_path = RESULTS_DIR / "phase5_noise_sweep.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping noise study")
        return None

    df = pd.read_csv(csv_path)
    noise_sweep = []
    for _, row in df.iterrows():
        noise_sweep.append({
            "single_qubit_error": round(float(row["error_rate"]), 6),
            "frobenius_error": round(float(row["frobenius_error"]), 4),
            "mae": round(float(row["mean_abs_error"]), 4),
            "correlation": round(float(row["correlation"]), 4),
            "diagonal_error": round(float(row["diagonal_error"]), 4),
        })

    # ZNE results (hardcoded from Phase 5 verified results)
    zne = {
        "noisy": {
            "frobenius_error": 0.3919,
            "mae": 0.1162,
            "correlation": 0.9988,
        },
        "corrected": {
            "frobenius_error": 0.2418,
            "mae": 0.0710,
            "correlation": 0.9989,
        },
        "improvement_pct": 38.3,
    }

    data = {
        "noise_sweep": noise_sweep,
        "zne": zne,
        "noise_tolerance_threshold": 0.006,
    }
    _write_json(OUTPUT_DIR / "noise_study.json", data)
    print(f"  Exported {len(noise_sweep)} noise sweep points + ZNE results")
    return data


def generate_data_bundle(all_data: dict[str, dict | None]) -> None:
    """Generate dashboard/data_bundle.js with all data as JS variables."""
    lines = [
        "// Auto-generated data bundle for the Quantum Kernel Dashboard",
        "// Do not edit manually — regenerate with: python -m dashboard.export_data",
        "",
    ]

    var_names = {
        "benchmark": "BENCHMARK_DATA",
        "kernels": "KERNEL_MATRICES",
        "expressibility": "EXPRESSIBILITY_DATA",
        "synthetic": "SYNTHETIC_DATA",
        "noise": "NOISE_DATA",
    }

    for key, var_name in var_names.items():
        data = all_data.get(key)
        if data is not None:
            json_str = json.dumps(data, indent=2)
            lines.append(f"const {var_name} = {json_str};")
        else:
            lines.append(f"const {var_name} = null;")
        lines.append("")

    BUNDLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    size_kb = BUNDLE_PATH.stat().st_size / 1024
    print(f"  Generated data_bundle.js ({size_kb:.1f} KB)")


def _write_json(path: Path, data: dict) -> None:
    """Write data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    """Export all experiment results for the dashboard."""
    parser = argparse.ArgumentParser(description="Export dashboard data")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute results instead of loading cached files",
    )
    args = parser.parse_args()

    if args.recompute:
        print("WARNING: --recompute not implemented. Using cached results.")

    print("=== Exporting Dashboard Data ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict | None] = {}

    print("\n1. Benchmark results (Phase 3)...")
    all_data["benchmark"] = export_benchmark_results()

    print("\n2. Kernel matrices (Phase 2/4)...")
    all_data["kernels"] = export_kernel_matrices()

    print("\n3. Expressibility analysis (Phase 4)...")
    all_data["expressibility"] = export_expressibility()

    print("\n4. Synthetic benchmark (Phase 4)...")
    all_data["synthetic"] = export_synthetic_benchmark()

    print("\n5. Noise study (Phase 5)...")
    all_data["noise"] = export_noise_study()

    print("\n6. Generating data bundle...")
    generate_data_bundle(all_data)

    # Summary
    exported = sum(1 for v in all_data.values() if v is not None)
    print(f"\n=== Export Complete: {exported}/5 datasets exported ===")

    if exported < 5:
        print("WARNING: Some datasets were missing. Run experiments first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
