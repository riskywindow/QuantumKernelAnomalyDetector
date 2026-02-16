# Quantum Kernel Anomaly Detection

> Hybrid quantum-classical anomaly detection using parameterized quantum circuits as kernel functions for financial fraud detection, with rigorous expressibility analysis and IBM Quantum hardware noise characterization.

Quantum kernel methods have the strongest theoretical evidence for quantum advantage in machine learning (Huang et al. 2021, Liu et al. 2021). This project provides empirical validation on real-world credit card fraud data (284,807 transactions) and a synthetic benchmark dataset, testing 4 quantum feature maps against classical baselines across 15 model configurations. It goes beyond the standard "quantum vs classical" comparison by analyzing *why* quantum kernels perform the way they do through expressibility metrics, geometric difference analysis, and hardware noise tolerance characterization.

## Key Results

- **Hardware-Efficient quantum kernel OCSVM achieves best F1 (0.76) across all 15 models** tested on real credit card fraud data
- Quantum kernel OCSVM achieves **0.90 AUROC**, competitive with classical methods (RBF OCSVM: 0.91)
- Synthetic benchmark demonstrates **17pp quantum advantage** when label structure matches quantum kernel geometry
- Higher quantum expressibility **negatively correlates** with AUROC (r = -0.835) — overly expressive kernels overfit
- Zero-noise extrapolation **recovers 38% of kernel fidelity** lost to hardware noise
- Quantum kernels remain useful (correlation > 0.99) up to **0.6% single-qubit error rate** (typical IBM hardware: 0.1-0.5%)

## Architecture

```
Credit Card Data (284K txns)
        |
        v
  StandardScale + PCA (5 dims)
        |
        v
  Quantum Feature Map  ──────────────  Classical Kernel
  (ZZ / IQP / Covariant /             (RBF / Polynomial)
   Hardware-Efficient)                        |
        |                                     |
        v                                     v
  |psi(x)> = U(x)|0>                   K(x1,x2) = exp(-gamma * ||x1-x2||^2)
        |
        v
  Kernel Matrix K[i,j] = |<psi(xi)|psi(xj)>|^2
        |
        v
  One-Class SVM / Kernel PCA / Isolation Forest / Autoencoder / LOF
        |
        v
  Anomaly Scores -> AUROC / AUPRC / F1
```

## Project Structure

```
quantum-kernel-anomaly/
├── src/
│   ├── data/           # Data loading (OpenML), preprocessing, synthetic generation
│   ├── kernels/        # Quantum & classical kernels, 4 feature maps, estimation
│   ├── models/         # OCSVM, Kernel PCA, baselines (IF, AE, LOF)
│   ├── analysis/       # Expressibility, geometric difference, noise study, ZNE
│   ├── hardware/       # IBM Quantum runner, noise models
│   └── utils/          # Metrics, caching, plotting
├── tests/              # 317 tests across all modules
├── experiments/        # Experiment runners and configs
├── configs/            # YAML experiment and hardware configs
├── dashboard/          # Interactive React + Plotly.js dashboard
└── scripts/            # Verification and runner scripts
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/quantum-kernel-anomaly.git
cd quantum-kernel-anomaly
uv sync                    # or: pip install -e .

# Run tests (317 tests, ~30s)
uv run python -m pytest tests/ -q

# Run the full benchmark (Phase 3, ~3 min)
uv run python -m experiments.run_experiment --config configs/experiments/kernel_comparison.yaml

# Run expressibility analysis (Phase 4, ~20s)
uv run python -m experiments.run_analysis

# Run noise study (Phase 5, ~40s)
uv run python -m experiments.run_hardware

# Export data and open the dashboard
uv run python -m dashboard.export_data
open dashboard/index.html
```

## Results

### Anomaly Detection Benchmark

200 normal training samples, 100 test samples (10 fraud), 5 qubits, statevector simulation.

| Model | AUROC | AUPRC | F1 | Type |
|-------|-------|-------|-----|------|
| Isolation Forest | 0.9144 | 0.6988 | 0.6250 | Baseline |
| OCSVM (RBF) | 0.9078 | 0.6707 | 0.6667 | Classical |
| **OCSVM (HW-Efficient)** | **0.8978** | **0.6687** | **0.7619** | **Quantum** |
| LOF | 0.8967 | 0.7208 | 0.7059 | Baseline |
| OCSVM (Covariant) | 0.8756 | 0.6560 | 0.6957 | Quantum |
| OCSVM (ZZ) | 0.8744 | 0.4219 | 0.4848 | Quantum |
| KPCA (IQP) | 0.8733 | 0.5129 | 0.5714 | Quantum |
| OCSVM (IQP) | 0.8178 | 0.5196 | 0.5714 | Quantum |

Full results for all 15 models available in the [interactive dashboard](dashboard/index.html).

### Expressibility Analysis

| Kernel | Eff. Dimension | KTA (centered) | AUROC |
|--------|---------------|----------------|-------|
| HW-Efficient | 12.17 | 0.110 | 0.898 |
| Covariant | 15.35 | 0.104 | 0.876 |
| ZZ | 55.67 | 0.091 | 0.874 |
| IQP | 62.53 | 0.091 | 0.818 |
| RBF | 2.57 | 0.096 | 0.908 |

**Finding:** Higher effective dimension correlates *negatively* with AUROC (r = -0.835, p = 0.078). The most expressive quantum kernels (ZZ, IQP) are not the best performers — they overfit to noise in the training data. The Hardware-Efficient kernel, with its constrained IBM-native gate set, achieves the best quantum AUROC.

### Hardware Noise Tolerance

| Metric | Noisy (1q=0.5%) | ZNE-Corrected | Improvement |
|--------|-----------------|---------------|-------------|
| Frobenius Error | 0.392 | 0.242 | 38.3% |
| Mean Abs Error | 0.116 | 0.071 | 38.8% |
| Correlation | 0.999 | 0.999 | &mdash; |

**Finding:** Quantum kernels remain useful (Pearson r > 0.99) up to ~0.6% single-qubit error rate, which covers typical IBM quantum hardware (0.1-0.5%). Beyond ~1.3% error, kernel structure degrades significantly. ZNE recovers 38% of lost fidelity.

## Methodology

### Quantum Feature Maps

Four parameterized quantum circuits encode classical data into quantum states:

- **ZZ Feature Map** (Havlicek et al. 2019): H &rarr; P(2x) &rarr; CX-P(2(pi-xi)(pi-xj))-CX. The canonical choice with pairwise data encoding.
- **IQP Feature Map**: H &rarr; RZ(x) &rarr; CX-RZ(xi*xj)-CX. Instantaneous Quantum Polynomial circuits with different entangling structure.
- **Covariant Feature Map**: RY(x) &rarr; ring-CX &rarr; RZ(x). Group-theoretic construction with ring entanglement and two rotation axes.
- **Hardware-Efficient Feature Map**: RZ-SX-RZ &rarr; linear-CX. Uses *only* IBM native gates (rz, sx, cx) for zero transpilation overhead on real hardware.

### Kernel Computation

The quantum kernel is computed as K(x1, x2) = |&langle;0|U&dagger;(x2)U(x1)|0&rangle;|&sup2;, where U(x) is the feature map circuit. The circuit U(x1) followed by U&dagger;(x2) is simulated, and the probability of the all-zeros state gives the kernel value. For the statevector backend, this is exact; for shot-based estimation, we sample 4096 shots per circuit.

### Error Mitigation

**Zero-Noise Extrapolation (ZNE):** Run kernel circuits at noise scale factors [1.0, 1.5, 2.0, 2.5, 3.0], then linearly extrapolate each kernel entry to the zero-noise limit. Combined with **eigenvalue clipping** for PSD correction (noisy kernel matrices always have negative eigenvalues from shot noise).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Quantum simulation | Qiskit 2.3.0, Qiskit Aer |
| Hardware backend | IBM Quantum (qiskit-ibm-runtime) |
| ML models | scikit-learn, PyTorch |
| Data | OpenML Credit Card Fraud (284,807 txns) |
| Visualization | Plotly.js, matplotlib, seaborn |
| Dashboard | React 18, Plotly.js (single-file, no build step) |
| Testing | pytest (317 tests) |
| Package management | uv |

## References

1. Havlicek, V. et al. "Supervised learning with quantum-enhanced feature spaces." *Nature* 567, 209-212 (2019).
2. Huang, H.-Y. et al. "Power of data in quantum machine learning." *Nature Communications* 12, 2631 (2021).
3. Liu, Y. et al. "A rigorous and robust quantum speed-up in supervised machine learning." *Nature Physics* 17, 1013-1017 (2021).
4. Abbas, A. et al. "The power of quantum neural networks." *Nature Computational Science* 1, 403-409 (2021).

## License

MIT
