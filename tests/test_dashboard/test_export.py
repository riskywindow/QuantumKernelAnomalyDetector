"""Tests for dashboard data export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.export_data import (
    BUNDLE_PATH,
    OUTPUT_DIR,
    export_benchmark_results,
    export_expressibility,
    export_kernel_matrices,
    export_noise_study,
    export_synthetic_benchmark,
    generate_data_bundle,
)


class TestExportBenchmarkResults:
    """Tests for Phase 3 benchmark results export."""

    def test_returns_dict(self) -> None:
        result = export_benchmark_results()
        assert result is not None
        assert "models" in result

    def test_model_count(self) -> None:
        result = export_benchmark_results()
        assert len(result["models"]) == 15

    def test_model_fields(self) -> None:
        result = export_benchmark_results()
        required_fields = {
            "name", "kernel_type", "auroc", "auprc",
            "f1", "precision", "recall", "fpr_at_95_recall",
        }
        for model in result["models"]:
            assert required_fields.issubset(model.keys()), (
                f"Missing fields in {model['name']}: "
                f"{required_fields - model.keys()}"
            )

    def test_kernel_type_classification(self) -> None:
        result = export_benchmark_results()
        types = {m["name"]: m["kernel_type"] for m in result["models"]}
        assert types["OCSVM (HW-Efficient)"] == "quantum"
        assert types["OCSVM (RBF)"] == "classical"
        assert types["Isolation Forest"] == "baseline"
        assert types["KPCA (ZZ)"] == "quantum_kpca"

    def test_hw_efficient_best_f1(self) -> None:
        result = export_benchmark_results()
        hw = next(m for m in result["models"] if m["name"] == "OCSVM (HW-Efficient)")
        assert hw["f1"] == pytest.approx(0.7619, abs=0.001)

    def test_json_output_valid(self) -> None:
        export_benchmark_results()
        path = OUTPUT_DIR / "benchmark_results.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "models" in data


class TestExportKernelMatrices:
    """Tests for kernel matrix export."""

    def test_returns_dict(self) -> None:
        result = export_kernel_matrices()
        assert result is not None
        assert "kernels" in result

    def test_matrix_size(self) -> None:
        result = export_kernel_matrices()
        for name, k in result["kernels"].items():
            matrix = k["matrix"]
            assert len(matrix) == 20, f"{name} rows != 20"
            assert len(matrix[0]) == 20, f"{name} cols != 20"

    def test_diagonal_near_one(self) -> None:
        result = export_kernel_matrices()
        for name, k in result["kernels"].items():
            if name == "Polynomial":
                continue
            for i in range(len(k["matrix"])):
                assert abs(k["matrix"][i][i] - 1.0) < 0.01, (
                    f"{name} diagonal [{i}] = {k['matrix'][i][i]}"
                )

    def test_stats_present(self) -> None:
        result = export_kernel_matrices()
        for name, k in result["kernels"].items():
            assert "mean_off_diag" in k["stats"]


class TestExportExpressibility:
    """Tests for expressibility analysis export."""

    def test_returns_dict(self) -> None:
        result = export_expressibility()
        assert result is not None

    def test_sections(self) -> None:
        result = export_expressibility()
        assert "eigenspectra" in result
        assert "metrics" in result
        assert "geometric_differences" in result

    def test_metrics_values(self) -> None:
        result = export_expressibility()
        hw = result["metrics"]["HW-Efficient"]
        assert hw["effective_dimension"] == pytest.approx(12.17, abs=0.1)
        assert hw["kta_centered"] == pytest.approx(0.1099, abs=0.001)

    def test_eigenspectra_length(self) -> None:
        result = export_expressibility()
        for name, spec in result["eigenspectra"].items():
            assert len(spec["eigenvalues"]) == 20
            assert len(spec["normalized"]) == 20


class TestExportSyntheticBenchmark:
    """Tests for synthetic benchmark export."""

    def test_returns_dict(self) -> None:
        result = export_synthetic_benchmark()
        assert result is not None
        assert "results" in result

    def test_result_count(self) -> None:
        result = export_synthetic_benchmark()
        assert len(result["results"]) == 6

    def test_zz_wins(self) -> None:
        result = export_synthetic_benchmark()
        zz = next(r for r in result["results"] if r["kernel"] == "ZZ")
        rbf = next(r for r in result["results"] if r["kernel"] == "RBF")
        assert zz["accuracy"] > rbf["accuracy"]
        assert zz["accuracy"] == pytest.approx(0.69, abs=0.01)


class TestExportNoiseStudy:
    """Tests for noise study export."""

    def test_returns_dict(self) -> None:
        result = export_noise_study()
        assert result is not None

    def test_noise_sweep_count(self) -> None:
        result = export_noise_study()
        assert len(result["noise_sweep"]) == 10

    def test_zne_improvement(self) -> None:
        result = export_noise_study()
        assert result["zne"]["improvement_pct"] == pytest.approx(38.3, abs=0.5)

    def test_noise_tolerance_threshold(self) -> None:
        result = export_noise_study()
        assert result["noise_tolerance_threshold"] == pytest.approx(0.006, abs=0.001)


class TestDataBundle:
    """Tests for JS data bundle generation."""

    def test_bundle_generation(self) -> None:
        all_data = {
            "benchmark": export_benchmark_results(),
            "kernels": export_kernel_matrices(),
            "expressibility": export_expressibility(),
            "synthetic": export_synthetic_benchmark(),
            "noise": export_noise_study(),
        }
        generate_data_bundle(all_data)
        assert BUNDLE_PATH.exists()

    def test_bundle_contains_variables(self) -> None:
        content = BUNDLE_PATH.read_text()
        assert "BENCHMARK_DATA" in content
        assert "KERNEL_MATRICES" in content
        assert "EXPRESSIBILITY_DATA" in content
        assert "SYNTHETIC_DATA" in content
        assert "NOISE_DATA" in content

    def test_bundle_size_reasonable(self) -> None:
        size_kb = BUNDLE_PATH.stat().st_size / 1024
        assert size_kb < 200, f"Bundle too large: {size_kb:.1f} KB"
        assert size_kb > 10, f"Bundle too small: {size_kb:.1f} KB"
