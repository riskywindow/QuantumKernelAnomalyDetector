"""Phase 6 verification: Dashboard, README, and project polish.

Verifies all Phase 6 deliverables exist and are correct.
Run: uv run python -m scripts.verify_phase6
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_file_exists(path: Path, min_size_bytes: int = 0) -> bool:
    """Check that a file exists and meets minimum size."""
    if not path.exists():
        print(f"  FAIL: {path.relative_to(PROJECT_ROOT)} does not exist")
        return False
    size = path.stat().st_size
    if size < min_size_bytes:
        print(
            f"  FAIL: {path.relative_to(PROJECT_ROOT)} too small "
            f"({size} bytes, need {min_size_bytes})"
        )
        return False
    print(f"  PASS: {path.relative_to(PROJECT_ROOT)} ({size:,} bytes)")
    return True


def check_file_contains(path: Path, keywords: list[str]) -> bool:
    """Check that a file contains all required keywords."""
    if not path.exists():
        return False
    content = path.read_text()
    missing = [k for k in keywords if k not in content]
    if missing:
        print(f"  FAIL: {path.relative_to(PROJECT_ROOT)} missing: {missing}")
        return False
    return True


def count_source_lines(directory: Path) -> int:
    """Count lines of Python source code."""
    total = 0
    for py_file in directory.rglob("*.py"):
        total += len(py_file.read_text().splitlines())
    return total


def main() -> None:
    """Run all Phase 6 verification checks."""
    print("=== Phase 6 Verification ===\n")
    all_pass = True

    # 1. Dashboard files
    print("1. Dashboard files:")
    all_pass &= check_file_exists(PROJECT_ROOT / "dashboard" / "index.html", 10_000)
    all_pass &= check_file_exists(PROJECT_ROOT / "dashboard" / "data_bundle.js", 10_000)
    all_pass &= check_file_exists(PROJECT_ROOT / "dashboard" / "export_data.py", 1_000)

    # Check data bundle contains all variables
    bundle_ok = check_file_contains(
        PROJECT_ROOT / "dashboard" / "data_bundle.js",
        ["BENCHMARK_DATA", "KERNEL_MATRICES", "EXPRESSIBILITY_DATA",
         "SYNTHETIC_DATA", "NOISE_DATA"],
    )
    if bundle_ok:
        print("  PASS: data_bundle.js contains all 5 data objects")
    all_pass &= bundle_ok

    # Check dashboard HTML contains key elements
    html_ok = check_file_contains(
        PROJECT_ROOT / "dashboard" / "index.html",
        ["react.production.min.js", "plotly-", "Quantum Kernel",
         "tab-btn", "Overview"],
    )
    if html_ok:
        print("  PASS: index.html contains React, Plotly, and tab components")
    all_pass &= html_ok

    # 2. JSON data files
    print("\n2. Dashboard data files:")
    data_dir = PROJECT_ROOT / "dashboard" / "public" / "data"
    for name in [
        "benchmark_results.json",
        "kernel_matrices.json",
        "expressibility.json",
        "synthetic_benchmark.json",
        "noise_study.json",
    ]:
        all_pass &= check_file_exists(data_dir / name, 100)

    # 3. README
    print("\n3. README.md:")
    all_pass &= check_file_exists(PROJECT_ROOT / "README.md", 3_000)
    readme_ok = check_file_contains(
        PROJECT_ROOT / "README.md",
        ["Quick Start", "Results", "Architecture", "Methodology",
         "Tech Stack", "References", "License"],
    )
    if readme_ok:
        print("  PASS: README contains all required sections")
    all_pass &= readme_ok

    # 4. LICENSE
    print("\n4. LICENSE:")
    all_pass &= check_file_exists(PROJECT_ROOT / "LICENSE", 500)
    lic_ok = check_file_contains(PROJECT_ROOT / "LICENSE", ["MIT License"])
    if lic_ok:
        print("  PASS: MIT License present")
    all_pass &= lic_ok

    # 5. run_all.sh
    print("\n5. scripts/run_all.sh:")
    all_pass &= check_file_exists(PROJECT_ROOT / "scripts" / "run_all.sh", 200)

    # 6. Run tests
    print("\n6. Test suite:")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    # Parse test count from output
    for line in result.stdout.splitlines():
        if "passed" in line:
            print(f"  {line.strip()}")
            break
    if result.returncode != 0:
        print(f"  FAIL: Tests failed (return code {result.returncode})")
        if result.stderr:
            # Print last few lines of stderr for debugging
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        all_pass = False
    else:
        print("  PASS: All tests passing")

    # 7. Project summary
    print("\n7. Project Summary:")
    src_lines = count_source_lines(PROJECT_ROOT / "src")
    test_lines = count_source_lines(PROJECT_ROOT / "tests")
    dashboard_size = (PROJECT_ROOT / "dashboard" / "index.html").stat().st_size / 1024
    bundle_size = (PROJECT_ROOT / "dashboard" / "data_bundle.js").stat().st_size / 1024

    py_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    test_files = list((PROJECT_ROOT / "tests").rglob("*.py"))

    print(f"  Source code: {src_lines:,} lines across {len(py_files)} files")
    print(f"  Test code: {test_lines:,} lines across {len(test_files)} files")
    print(f"  Dashboard: {dashboard_size:.1f} KB HTML + {bundle_size:.1f} KB data")

    experiment_scripts = sorted(
        p.relative_to(PROJECT_ROOT)
        for p in (PROJECT_ROOT / "experiments").glob("*.py")
    )
    print(f"  Experiment scripts: {len(experiment_scripts)}")
    for script in experiment_scripts:
        print(f"    - {script}")

    # Final verdict
    print("\n" + "=" * 50)
    if all_pass:
        print("PHASE 6 VERIFICATION: ALL CHECKS PASSED")
    else:
        print("PHASE 6 VERIFICATION: SOME CHECKS FAILED")
    print("=" * 50)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
