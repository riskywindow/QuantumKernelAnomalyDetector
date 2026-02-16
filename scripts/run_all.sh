#!/bin/bash
set -e

echo "=== Quantum Kernel Anomaly Detection — Full Pipeline ==="
echo ""

echo "Step 1: Running tests..."
python -m pytest tests/ -q
echo ""

echo "Step 2: Running benchmark (Phase 3)..."
python -m experiments.run_experiment --config configs/experiments/kernel_comparison.yaml
echo ""

echo "Step 3: Running expressibility analysis (Phase 4)..."
python -m experiments.run_analysis
echo ""

echo "Step 4: Running synthetic benchmark (Phase 4)..."
python -m experiments.run_synthetic_benchmark
echo ""

echo "Step 5: Running noise study (Phase 5)..."
python -m experiments.run_hardware
echo ""

echo "Step 6: Exporting dashboard data..."
python -m dashboard.export_data
echo ""

echo "=== All Complete ==="
echo "Open dashboard/index.html in your browser to view results."
