# PROGRESS.md — Quantum Kernel Anomaly Detection

> This file is updated at the end of every Claude Code session. Read this FIRST to understand current project state.

---

## Project Status: ✅ COMPLETE — All 6 Phases Done

### Completed Phases
- **Phase 1: Foundation** — Completed 2026-02-15
- **Phase 2: Kernel Engine Expansion** — Completed 2026-02-15
- **Phase 3: ML Pipeline & Benchmarking** — Completed 2026-02-15
- **Phase 4: Expressibility Analysis** — Completed 2026-02-15
- **Phase 5: Hardware Execution & Noise Analysis** — Completed 2026-02-15
- **Phase 6: Dashboard, README & Polish** — Completed 2026-02-15

### Phase 6 Task Checklist
- [x] Task 6.1: Dashboard Data Export (`dashboard/export_data.py`)
- [x] Task 6.2: React Dashboard (`dashboard/index.html`)
- [x] Task 6.3: README (`README.md`)
- [x] Task 6.4: Project Polish (LICENSE, .gitignore, run_all.sh, warnings fix)
- [x] Task 6.5: Tests (`tests/test_dashboard/test_export.py` — 24 new tests)
- [x] Task 6.6: Verification (`scripts/verify_phase6.py`)

### Phase 6 Exit Criteria
- [x] Dashboard data export generates all 5 JSON data files + JS bundle (54.4 KB)
- [x] React dashboard opens in browser and displays all 6 tabs with correct data
- [x] Dashboard is a single HTML file (32 KB) — no build step required
- [x] README is comprehensive, well-formatted, and includes Quick Start that works
- [x] LICENSE file exists (MIT)
- [x] scripts/run_all.sh executes end-to-end without errors
- [x] All new tests pass, no regressions on existing 317 tests — 341/341 total
- [x] Verification script confirms all deliverables exist
- [x] PROGRESS.md updated — FINAL UPDATE marking project as complete

### Phase 5 Task Checklist
- [x] Task 5.1: IBM Quantum Backend Integration (`src/hardware/ibm_runner.py`)
- [x] Task 5.2: Noise Models (`src/hardware/noise_models.py`)
- [x] Task 5.3: Noise Impact Study (`src/analysis/noise.py`)
- [x] Task 5.4: PSD Correction (`src/analysis/psd_correction.py`)
- [x] Task 5.5: Zero-Noise Extrapolation (`src/analysis/zne.py`)
- [x] Task 5.6: Hardware Execution Script (`experiments/run_hardware.py`)
- [x] Task 5.7: QuantumKernel Noise Support (updated `src/kernels/quantum.py`)
- [x] Task 5.8: Tests (73 new tests, 317 total, all passing)
- [x] Task 5.9: Verification Script (`scripts/verify_phase5.py`)

### Phase 5 Exit Criteria
- [x] IBMQuantumRunner authenticates and can query backend info (or raises clear error if no token)
- [x] LocalNoiseRunner computes kernel matrices with realistic noise
- [x] Depolarizing noise models build correctly with configurable error rates
- [x] Noise sweep produces degradation curves showing kernel quality decreasing with noise
- [x] PSD correction recovers valid kernel matrices from noisy ones (all three methods)
- [x] ZNE produces kernel estimates closer to ideal than raw noisy estimates
- [x] QuantumKernel supports noise_model parameter for noisy simulation
- [x] run_hardware.py completes in simulation mode end-to-end (38.9s)
- [x] All new tests pass, no regressions on existing 244 tests — 317/317 total
- [x] Verification script completes in under 120 seconds (0.5s)
- [x] PROGRESS.md updated with noise analysis results

### Phase 4 Task Checklist
- [x] Task 4.1: Kernel Target Alignment (`src/analysis/alignment.py`)
- [x] Task 4.2: Effective Dimension (`src/analysis/expressibility.py`)
- [x] Task 4.3: Geometric Difference (`src/analysis/geometric.py`)
- [x] Task 4.4: Synthetic Advantage Dataset (`src/data/synthetic.py`)
- [x] Task 4.5: Synthetic Advantage Benchmark (`experiments/run_synthetic_benchmark.py`)
- [x] Task 4.6: Combined Analysis Report (`experiments/run_analysis.py`)
- [x] Task 4.7: Tests (56 new tests, 244 total, all passing)
- [x] Task 4.8: Verification Script (`scripts/verify_phase4.py`)

### Phase 4 Exit Criteria
- [x] Kernel target alignment (centered and uncentered) computed correctly
- [x] Effective dimension and eigenspectrum analysis work for all kernel matrices
- [x] Geometric difference metric computes without numerical errors (with regularization)
- [x] Synthetic advantage dataset generates balanced, labeled data using quantum kernel structure
- [x] Synthetic benchmark shows quantum kernel outperforming classical on the synthetic data
- [x] Combined analysis produces all three tables and three plots on real fraud data
- [x] Correlation between KTA and Phase 3 AUROC is documented
- [x] All new tests pass, no regressions on existing 188 tests — 244/244 total
- [x] Verification script completes in under 90 seconds (2.1s)
- [x] PROGRESS.md updated with analysis results

### Phase 3 Task Checklist
- [x] Task 3.1: Metrics module (AnomalyMetrics dataclass, compute_anomaly_metrics, compute_metrics_table)
- [x] Task 3.2: One-Class SVM with precomputed kernels (QuantumOCSVM)
- [x] Task 3.3: Kernel PCA anomaly detector (KernelPCAAnomalyDetector)
- [x] Task 3.4: Classical baselines (IsolationForest, Autoencoder, LOF)
- [x] Task 3.5: Experiment runner (experiments/run_experiment.py)
- [x] Task 3.6: Updated experiment config (kernel_comparison.yaml)
- [x] Task 3.7: Kernel factory (src/kernels/factory.py)
- [x] Task 3.8: Tests (65 new tests, 188 total, all passing)
- [x] Task 3.9: Verification script (scripts/verify_phase3.py)

### Phase 3 Exit Criteria
- [x] AnomalyMetrics dataclass and compute_anomaly_metrics work correctly
- [x] QuantumOCSVM trains on precomputed kernel and produces valid anomaly scores
- [x] KernelPCAAnomalyDetector trains and scores with precomputed kernels
- [x] All three baselines (IsolationForest, Autoencoder, LOF) train and evaluate
- [x] All baselines follow the same interface: fit(), predict_scores(), evaluate()
- [x] Kernel factory creates all kernel types from config dicts
- [x] Experiment runner executes end-to-end with the YAML config
- [x] Benchmark produces CSV results table + results heatmap plot
- [x] All new tests pass, no regressions on existing 123 tests — 188/188 total
- [x] Verification script completes in under 60 seconds (3.6s)
- [x] PROGRESS.md updated with session details and benchmark results

### Phase 2 Task Checklist
- [x] Task 2.1: BaseFeatureMap ABC + refactor ZZFeatureMap and QuantumKernel
- [x] Task 2.2: IQP Feature Map
- [x] Task 2.3: Covariant Feature Map
- [x] Task 2.4: Hardware-Efficient Feature Map
- [x] Task 2.5: Shot-based kernel estimation (sampler backend)
- [x] Task 2.6: Classical kernel baselines (RBF, Polynomial)
- [x] Task 2.7: Experiment config (kernel_comparison.yaml)
- [x] Task 2.8: Tests (85 new tests, 123 total, all passing)
- [x] Task 2.9: Verification script + plots

### Phase 1 Task Checklist
- [x] Task 1.1: Project setup (uv, directory structure, pyproject.toml, .gitignore, git init)
- [x] Task 1.2: Data loading & preprocessing (loader.py, transforms.py)
- [x] Task 1.3: Abstract kernel interface (base.py with validation)
- [x] Task 1.4: ZZ Feature Map implementation (zz.py)
- [x] Task 1.5: Quantum kernel + statevector simulation (quantum.py, estimation.py)
- [x] Task 1.6: Tests + verification script (38 tests pass, heatmap generated)

---

## Session Log

### Session 6 — 2026-02-15
**Phase:** 6 — Dashboard, README & Polish
**Status:** COMPLETE (PROJECT COMPLETE)

**Accomplished:**
- Created `dashboard/export_data.py`: Exports all Phase 3-5 results from cached CSV/NPY files to 5 JSON files + a JavaScript data bundle (54.4 KB) for the React dashboard. Handles missing files gracefully.
- Created `dashboard/index.html`: Single-file React + Plotly.js dashboard (32 KB) with 6 tabs: Overview (key metrics cards + architecture diagram), Kernel Comparison (2×3 heatmap grid + stats bar chart), Benchmark Results (grouped bar chart + sortable table), Expressibility (eigenspectrum + scatter + geometric difference), Quantum Advantage (accuracy bar chart + explanation + results table), Hardware & Noise (fidelity/correlation curves + ZNE summary). Dark theme (#0d1117), CDN-loaded React 18 + Plotly 2.35.0 + Babel standalone. No build step — opens by double-clicking.
- Created `README.md`: Comprehensive project README with Key Results, ASCII Architecture diagram, Project Structure, Quick Start (copy-pasteable), Results tables (top 8 models), Expressibility/Noise findings, Methodology (feature maps, kernel computation, error mitigation), Tech Stack table, References.
- Created `LICENSE`: MIT License with Rishi Vinod Kumar, 2026.
- Updated `.gitignore`: Added granular experiment result patterns (*.npy, *.csv, *.png) instead of blanket `experiments/results/`. Added `node_modules/` for dashboard.
- Created `scripts/run_all.sh`: End-to-end pipeline script (tests → benchmark → analysis → synthetic → noise → dashboard export).
- Fixed RuntimeWarning in `src/utils/metrics.py`: Replaced np.where with zero-safe denominator to prevent divide-by-zero warning when P+R=0.
- Created `tests/test_dashboard/test_export.py`: 24 tests covering all 5 export functions + data bundle generation. Tests verify data structure, known values (HW-Efficient F1=0.7619, ZNE improvement=38.3%), and bundle size.
- Created `scripts/verify_phase6.py`: Comprehensive verification checking all deliverables exist, contain required content, and all 341 tests pass.

**Project Final Stats:**
- 341 tests, all passing (317 from Phases 1-5 + 24 new)
- 3,928 lines of source code across 35 files
- 3,586 lines of test code across 35 files
- 4 experiment runner scripts
- Dashboard: 32 KB HTML + 54.4 KB data bundle
- README: 8 KB, 7 sections

**Issues Encountered:**
- None — clean implementation.

### Session 5 — 2026-02-15
**Phase:** 5 — Hardware Execution & Noise Analysis
**Status:** COMPLETE

**Accomplished:**
- Implemented `src/hardware/noise_models.py`: Configurable depolarizing noise models with `build_depolarizing_noise_model` (single/two-qubit/readout errors on all gate types), `build_noise_sweep` (log-spaced error rates, 2q=10x 1q, caps at physical limits 3/4 and 15/16), `try_fetch_real_noise_model` (graceful fallback if no IBM token).
- Implemented `src/hardware/ibm_runner.py`: `IBMQuantumRunner` for real IBM Quantum hardware (token from env var, backend selection, transpilation stats, SamplerV2 execution) and `LocalNoiseRunner` for local noise simulation (AerSimulator with noise model, same interface).
- Implemented `src/analysis/noise.py`: `compute_noisy_kernel_matrix` builds all U†(x2)U(x1) circuits with measurements and runs through noisy AerSimulator. `compute_kernel_fidelity` computes Frobenius error, MAE, max error, Pearson correlation, PSD check. `run_noise_sweep` runs full sweep and collects all metrics.
- Implemented `src/analysis/psd_correction.py`: Three PSD correction methods — eigenvalue clipping (set negative eigenvalues to 0), Higham nearest PSD (alternating projections), and diagonal shift (add |λ_min|·I). All produce valid PSD matrices. `analyze_psd_violation` reports severity.
- Implemented `src/analysis/zne.py`: Zero-noise extrapolation with `scale_noise_model` (multiply error rates by scale factor), linear and exponential curve fitting via scipy, entry-level and matrix-level ZNE. Exponential falls back to linear if fit fails. Results clamped to [0, 1].
- Updated `src/kernels/quantum.py`: Added `noise_model` parameter to QuantumKernel. When provided with sampler backend, uses AerSimulator(noise_model=noise_model). Statevector backend ignores noise model (exact simulation).
- Created `configs/hardware/noise_sweep.yaml`: Full configuration for noise sweep, ZNE, and hardware execution.
- Created `experiments/run_hardware.py`: End-to-end experiment script with simulation mode (default) and hardware mode (--real-hardware). Runs noise sweep, PSD analysis, ZNE comparison, produces 3 plots and CSV.
- Created `scripts/verify_phase5.py`: Quick verification (0.5s) of all Phase 5 components.
- Wrote 73 new tests across 6 test files (test_noise_models, test_ibm_runner, test_psd_correction, test_zne, test_noise_study, test_quantum_noisy). 317 total tests, all passing.
- Added `qiskit-ibm-runtime` (v0.45.1) dependency.

**Noise Sweep Results (HW-Efficient, 30 samples, 5 qubits, 4096 shots):**
| Single-Qubit Error | Frobenius Error | MAE | Correlation | PSD Violation | Diagonal Error |
|-------------------|----------------|------|-------------|---------------|----------------|
| 0.000100 | 0.0219 | 0.0067 | 0.9998 | Yes | 0.0187 |
| 0.000199 | 0.0368 | 0.0113 | 0.9998 | Yes | 0.0361 |
| 0.000398 | 0.0700 | 0.0207 | 0.9997 | Yes | 0.0727 |
| 0.000794 | 0.1309 | 0.0387 | 0.9996 | Yes | 0.1377 |
| 0.001583 | 0.2430 | 0.0721 | 0.9994 | Yes | 0.2540 |
| 0.003158 | 0.4191 | 0.1246 | 0.9986 | Yes | 0.4386 |
| 0.006300 | 0.6483 | 0.1932 | 0.9948 | Yes | 0.6762 |
| 0.012566 | 0.8450 | 0.2530 | 0.9715 | Yes | 0.8729 |
| 0.025066 | 0.9317 | 0.2811 | 0.6766 | Yes | 0.9547 |
| 0.050000 | 0.9470 | 0.2868 | 0.1108 | Yes | 0.9686 |

**ZNE Results (base noise: 1q=0.005, 2q=0.02, ro=0.01, scale factors [1.0, 1.5, 2.0, 2.5, 3.0]):**
| Metric | Noisy | ZNE-Corrected | Improvement |
|--------|-------|---------------|-------------|
| Frobenius Error | 0.3919 | 0.2418 | 38.3% |
| Mean Abs Error | 0.1162 | 0.0710 | 38.8% |
| Correlation | 0.9988 | 0.9989 | — |

**Key Findings:**
1. **Kernel quality degrades monotonically with noise:** Frobenius error increases from 0.022 at 10⁻⁴ single-qubit error to 0.947 at 5×10⁻². The degradation is roughly linear in log-error-rate up to ~1% error, then saturates as the kernel matrix approaches uniform noise.
2. **Correlation is remarkably robust:** Pearson correlation between noisy and ideal off-diagonal elements stays above 0.99 up to 0.63% single-qubit error (typical IBM hardware is ~0.1-0.5%). Only at 2.5% error does correlation drop significantly (0.68), and at 5% it becomes near-random (0.11).
3. **All noisy kernel matrices are non-PSD:** Even at the lowest noise level (10⁻⁴), shot noise creates small negative eigenvalues. PSD correction is essential for any shot-based kernel estimation, not just high-noise scenarios.
4. **ZNE recovers ~38% of lost kernel fidelity:** At realistic IBM hardware noise levels (1q=0.005, 2q=0.02), ZNE reduces Frobenius error from 0.39 to 0.24 and MAE from 0.116 to 0.071. This is a meaningful but not transformative improvement — consistent with ZNE's known limitations for deep circuits.
5. **Diagonal error dominates:** The mean diagonal error (|K[i,i] - 1.0|) closely tracks the overall Frobenius error, showing that noise most visibly corrupts the K(x,x)=1 identity. At 5% error, diagonal values average 0.03 instead of 1.0.
6. **Noise tolerance threshold:** Quantum kernels remain useful (correlation > 0.99) up to ~0.6% single-qubit error rate. Beyond ~1.3% error, the kernel structure is significantly degraded (correlation 0.97, Frobenius error 0.85).

**Decisions Made:**
- Used HW-Efficient feature map for noise study — native IBM gate set means zero transpilation overhead
- Depolarizing noise model adds errors to all gate types: rz, sx, h, p, ry, cx
- Two-qubit error = 10× single-qubit error (standard IBM ratio)
- Readout error = 2× single-qubit error
- Error rates capped at 3/4 (single-qubit) and 15/16 (two-qubit) — physical limits of depolarizing channel
- ZNE uses linear extrapolation by default (exponential available as fallback)
- PSD correction: eigenvalue clipping as primary method (minimal matrix perturbation)
- IBM_QUANTUM_TOKEN loaded from environment variable only — never hardcoded

**Issues Encountered:**
- `qiskit-ibm-runtime` was not installed — added via `uv add qiskit-ibm-runtime` (v0.45.1, installed cleanly with 17 transitive dependencies)
- `QuantumPreprocessor` constructor requires `n_features` as first arg (not keyword-only) — fixed in experiment runner
- `prepare_anomaly_split` returns numpy arrays not DataFrames — removed `.values` calls in experiment runner
- No other issues — clean implementation

**Next Steps:**
- Phase 6: Dashboard & Polish — React + Plotly.js dashboard, LaTeX writeup, README, demo video

### Session 4 — 2026-02-15
**Phase:** 4 — Expressibility Analysis
**Status:** COMPLETE

**Accomplished:**
- Implemented `src/analysis/alignment.py`: Kernel target alignment (KTA) and centered KTA (Cortes et al. 2012). Converts labels from {0,1} to {-1,+1}, computes Frobenius inner product with ideal kernel K_y = outer(y_pm, y_pm). Centering via H@K@H removes constant shift bias.
- Implemented `src/analysis/expressibility.py`: Effective dimension via spectral entropy (exp(-sum(p*log(p)))), participation ratio ((sum λ)²/sum(λ²)), full eigenspectrum analysis with cumulative variance explained. Uses `eigvalsh` for numerical stability on symmetric matrices.
- Implemented `src/analysis/geometric.py`: Geometric difference metric from Huang et al. (2021). Uses eigendecomposition-based matrix sqrt (not `scipy.linalg.sqrtm`) for numerical stability. Regularizes K_target for inversion (eps=1e-5). Bidirectional and pairwise computation supported.
- Implemented `src/data/synthetic.py`: Quantum advantage dataset generation. Labels determined by fidelity |<psi_ref|psi(x_i)>|² with a random reference state. Median threshold for balanced labels. Configurable noise rate.
- Created `experiments/run_synthetic_benchmark.py`: Full classification benchmark (SVC with precomputed kernels) on synthetic data. Evaluates accuracy, F1, AUROC for 4 quantum + 2 classical kernels.
- Created `experiments/run_analysis.py`: Combined expressibility analysis on real fraud data. Three tables (expressibility, geometric difference, correlation), three plots (eigenspectra, KTA vs AUROC, geometric heatmap). Loads Phase 3 AUROC for correlation analysis.
- Created `scripts/verify_phase4.py`: Quick verification (2.1s) of all components.
- Wrote 56 new tests across 4 test files (test_alignment, test_expressibility, test_geometric, test_synthetic).

**Synthetic Benchmark Results (ZZ-generated labels, 150 train, 100 test, 4 qubits):**
| Kernel | Accuracy | F1 | AUROC | Centered KTA |
|--------|----------|-----|-------|-------------|
| ZZ (generating kernel) | 0.690 | 0.699 | 0.773 | 0.116 |
| Covariant | 0.520 | 0.564 | 0.568 | 0.067 |
| RBF | 0.520 | 0.529 | 0.461 | 0.052 |
| Polynomial | 0.480 | 0.490 | 0.442 | 0.039 |
| HW-Efficient | 0.460 | 0.481 | 0.415 | 0.060 |
| IQP | 0.430 | 0.457 | 0.481 | 0.062 |

**Expressibility Analysis (Credit Card Fraud, 100 test samples, 5 qubits):**
| Kernel | Eff. Dimension | Participation Ratio | KTA (centered) | Phase 3 AUROC |
|--------|---------------|--------------------|--------------------|------------|
| ZZ | 55.67 | 33.62 | 0.0906 | 0.8744 |
| IQP | 62.53 | 42.34 | 0.0907 | 0.8178 |
| Covariant | 15.35 | 7.16 | 0.1035 | 0.8756 |
| HW-Efficient | 12.17 | 5.66 | 0.1099 | 0.8978 |
| RBF | 2.57 | 1.62 | 0.0959 | 0.9078 |

**Geometric Differences vs RBF (Credit Card Fraud):**
| Kernel | g(K_Q, K_RBF) | g(K_RBF, K_Q) | Advantage Ratio |
|--------|--------------|---------------|----------------|
| ZZ | 3.63 | 339.85 | 0.011 |
| IQP | 3.75 | 392.59 | 0.010 |
| Covariant | 2.48 | 95.34 | 0.026 |
| HW-Efficient | 3.29 | 72.74 | 0.045 |

**Correlation Analysis:**
| Metric Pair | Pearson r | p-value |
|-------------|-----------|---------|
| KTA vs AUROC | 0.534 | 0.354 |
| Eff. Dimension vs AUROC | -0.835 | 0.078 |
| Geometric Diff vs AUROC (quantum only) | -0.460 | 0.540 |

**Key Findings:**
1. **Synthetic benchmark confirms quantum advantage by construction:** ZZ kernel (the generating kernel) achieves 69% accuracy vs 52% for RBF — a clear 17pp gap demonstrating that the quantum kernel captures structure classical kernels miss when labels are quantum-determined.
2. **On real fraud data, no quantum advantage:** g(K_RBF, K_Q) >> g(K_Q, K_RBF) for ALL quantum kernels, meaning the classical RBF kernel captures significantly more predictive structure than quantum kernels on this dataset. This is consistent with Phase 3's finding that RBF OCSVM slightly outperforms quantum OCSVM.
3. **Higher expressibility ≠ higher performance:** ZZ/IQP have the highest effective dimensions (56-63) but NOT the best AUROC. HW-Efficient has lower dimensionality (12) but better AUROC (0.90). Negative correlation r=-0.835 (p=0.078) suggests that overly expressive kernels overfit on this dataset.
4. **KTA positively but weakly correlated with AUROC:** r=0.534 but not significant (p=0.354) with only 5 data points. HW-Efficient has highest KTA (0.110) matching its best quantum AUROC.
5. **Quantum kernels and classical kernels operate in fundamentally different regimes:** Quantum kernel matrices are very sparse (mean off-diagonal ~0.05-0.18) while RBF is denser (mean ~0.41). This explains the large geometric differences in both directions.

**Decisions Made:**
- Used eigendecomposition for matrix sqrt (not `scipy.linalg.sqrtm`) to avoid complex-valued results from nearly-PSD matrices
- Regularization eps=1e-5 for geometric difference — prevents NaN/inf from singular quantum kernel matrices
- Median fidelity threshold for synthetic dataset — ensures approximately balanced labels
- ZZ with reps=2 used to generate synthetic labels — strongest entanglement structure for demonstrating advantage
- SVC (not One-Class SVM) for synthetic benchmark — it's a classification problem with labeled training data

**Issues Encountered:**
- Previous session crashed, leaving all files in place. Verified everything works by running tests (244 pass), verification script (2.1s), synthetic benchmark (77.5s), and analysis (18.6s) — all successful.

**Next Steps:**
- Phase 5: IBM Quantum hardware execution — backend integration, batched job submission, zero-noise extrapolation, PSD correction

### Session 3 — 2026-02-15
**Phase:** 3 — ML Pipeline & Benchmarking
**Status:** COMPLETE

**Accomplished:**
- Implemented `src/utils/metrics.py`: AnomalyMetrics dataclass with AUROC, AUPRC, F1, precision, recall, FPR@95%recall, optimal threshold. `compute_anomaly_metrics()` handles all metric computation. `compute_metrics_table()` produces sorted DataFrame.
- Implemented `src/models/ocsvm.py`: QuantumOCSVM wrapping sklearn OneClassSVM with kernel='precomputed'. Negates decision_function so higher = more anomalous.
- Implemented `src/models/kpca.py`: KernelPCAAnomalyDetector using sklearn KernelPCA + KernelCenterer. Scores via reconstruction error (projection norm deficit): points projecting weakly onto training PCs score high.
- Implemented `src/models/baselines.py`: IsolationForestBaseline, AutoencoderBaseline (PyTorch, 3-layer: input→16→encoding_dim→16→input), LOFBaseline. All follow fit/predict_scores/evaluate interface.
- Implemented `src/kernels/factory.py`: `create_quantum_kernel()`, `create_classical_kernel()`, `create_all_kernels()` — builds kernels from YAML config dicts.
- Updated `configs/experiments/kernel_comparison.yaml` with model and baseline configuration.
- Created `experiments/run_experiment.py`: Full experiment orchestration with argparse, kernel caching, model training/evaluation, CSV output, and heatmap visualization.
- Created `scripts/verify_phase3.py`: Fast verification (3.6s) with ZZ + RBF kernels, OCSVM + KPCA + 3 baselines.
- Wrote 65 new tests across 5 test files (test_metrics, test_ocsvm, test_kpca, test_baselines, test_factory).
- Added `torch` dependency for autoencoder baseline.

**Benchmark Results (200 train, 100 test, 5 qubits, seed=42):**
| Model | AUROC | AUPRC | F1 |
|-------|-------|-------|-----|
| Isolation Forest | 0.9144 | 0.6988 | 0.6250 |
| OCSVM (RBF) | 0.9078 | 0.6707 | 0.6667 |
| OCSVM (HW-Efficient) | 0.8978 | 0.6687 | 0.7619 |
| LOF | 0.8967 | 0.7208 | 0.7059 |
| OCSVM (Covariant) | 0.8756 | 0.6560 | 0.6957 |
| OCSVM (ZZ) | 0.8744 | 0.4219 | 0.4848 |
| KPCA (IQP) | 0.8733 | 0.5129 | 0.5714 |
| KPCA (ZZ) | 0.8689 | 0.4658 | 0.5455 |
| OCSVM (IQP) | 0.8178 | 0.5196 | 0.5714 |
| KPCA (Covariant) | 0.7944 | 0.7101 | 0.7500 |
| KPCA (HW-Efficient) | 0.7800 | 0.6679 | 0.6667 |
| OCSVM (Polynomial) | 0.5544 | 0.3123 | 0.3750 |
| Autoencoder | 0.5411 | 0.1736 | 0.3030 |
| KPCA (RBF) | 0.1678 | 0.0636 | 0.1835 |
| KPCA (Polynomial) | 0.1289 | 0.0615 | 0.1835 |

**Key Findings:**
- Quantum kernel OCSVM models achieve AUROC 0.87-0.90, competitive with classical baselines
- OCSVM (HW-Efficient) achieves the best F1 score (0.7619) of all models
- OCSVM consistently outperforms KPCA for the same kernel
- Classical baselines (IsolationForest, LOF) are strong but quantum kernels are competitive
- Polynomial kernel performs poorly for anomaly detection (non-normalized values)
- KPCA with classical kernels (RBF, Polynomial) has inverted scores — likely due to projection norm approach
- Full experiment runs in ~154s (4 quantum kernels × ~20s each for train+test)

**Decisions Made:**
- Score convention: ALL models output higher = more anomalous (sklearn models negated)
- KPCA scoring: reconstruction error via projection norm deficit (max_train_norm_sq - test_norm_sq), NOT centroid distance — centroid distance fails when anomaly points have degenerate kernel values
- Autoencoder architecture: input→16→3→16→input with ReLU + MSE + Adam, 50 epochs
- LOF n_neighbors clamped to n_train-1 to handle small datasets
- Experiment runner uses KernelEstimator caching for re-run speed

**Issues Encountered:**
- `torch` was not installed — added via `uv add torch` (v2.10.0)
- KPCA centroid distance scoring failed: anomaly points with near-zero kernel values project to a degenerate point near the centroid in PCA space, giving them LOW scores. Fixed by switching to projection norm deficit (reconstruction error proxy) where weak projections = high anomaly score.

**Next Steps:**
- Phase 4: Expressibility analysis — effective dimension, kernel target alignment, geometric difference, synthetic advantage dataset

### Session 2 — 2026-02-15
**Phase:** 2 — Kernel Engine Expansion
**Status:** COMPLETE

**Accomplished:**
- Created `src/kernels/feature_maps/base.py`: BaseFeatureMap ABC with `n_qubits`, `name`, `reps`, `total_gate_count` abstract properties and `build_circuit()` method. Added `_validate_input()` helper.
- Refactored `ZZFeatureMap` to inherit from BaseFeatureMap, using properties instead of direct attributes. All Phase 1 tests still pass unchanged.
- Refactored `QuantumKernel` to accept `BaseFeatureMap` (not just `ZZFeatureMap`). Updated `name` property to use `feature_map.name`.
- Implemented `IQPFeatureMap`: H → RZ(x_i) → CX-RZ(x_i*x_j)-CX per rep. Uses RZ gates and raw product (not (π-x) form), producing fundamentally different kernels from ZZ. Verified statevectors differ from ZZ.
- Implemented `CovariantFeatureMap`: RY(x_i) → ring CX → RZ(x_i) per rep. Uses two rotation axes (RY + RZ) for broader Bloch sphere coverage. Ring entanglement wraps around: CX(n-1, 0).
- Implemented `HardwareEfficientFeatureMap`: RZ(x_i) → SX → RZ(x_i) → linear CX per rep. Verified ONLY uses IBM native gates {rz, sx, cx} — zero transpilation overhead on real hardware.
- Added shot-based sampler backend to QuantumKernel using AerSimulator + measurement. Tested: mean |diff| from statevector = 0.0042 at 1024 shots, correlation 0.9952.
- Implemented `RBFKernel` and `PolynomialKernel` behind BaseKernel interface using sklearn pairwise kernels. gamma='scale' auto-adapts to data.
- Created `configs/experiments/kernel_comparison.yaml` defining all 6 kernels.
- Wrote 85 new tests across 6 test files, bringing total to 123 (all passing).
- Verification script computes all 6 kernel matrices on 20 samples, validates all quantum kernels pass (symmetric, PSD, unit diagonal, [0,1] bounded), produces 2x3 heatmap grid and shot noise scatter plot.

**Kernel Comparison Results (20x20 matrix, 5 qubits, 2 reps):**
| Kernel | Mean Off-Diag | Min | Max | All Checks |
|--------|--------------|------|------|------------|
| ZZ | 0.058 | 0.000 | 1.000 | PASS |
| IQP | 0.053 | 0.000 | 1.000 | PASS |
| Covariant | 0.120 | 0.001 | 1.000 | PASS |
| HW-Efficient | 0.175 | 0.000 | 1.000 | PASS |
| RBF | 0.413 | 0.003 | 1.000 | PASS |
| Polynomial | 86.85 | 13.64 | 1237.2 | symmetric, PSD only |

**Decisions Made:**
- IQP uses RZ gates (not P) and raw product x_i*x_j (not (π-x_i)(π-x_j)) — intentionally different from ZZ
- Covariant always uses ring entanglement (no 'linear'/'full' option)
- Hardware-efficient uses RZ-SX-RZ sandwich (not H-RZ) to stay within IBM native gate set
- Sampler uses `AerSimulator().run(transpiled, shots=N)` — most reliable Qiskit 2.x API path
- Classical kernels correctly have non-unit diagonal (Polynomial) — documented in class docstrings

**Issues Encountered:**
- `__init__.py` imports failed before feature maps were implemented. Resolved by implementing all maps before updating exports.
- No other issues — clean implementation.

**Next Steps:**
- Phase 3: ML Pipeline — One-Class SVM with precomputed kernels, Kernel PCA anomaly detection, classical baselines, full benchmark suite

### Session 1 — 2026-02-15
**Phase:** 1 — Foundation
**Status:** COMPLETE

**Accomplished:**
- Created full project directory structure with all placeholder files for phases 1-6
- Set up pyproject.toml with uv, installed all dependencies (Qiskit 2.3.0, scikit-learn, etc.)
- Implemented `src/data/loader.py`: OpenML credit card fraud dataset loader with parquet caching
- Implemented `src/data/transforms.py`: QuantumPreprocessor with StandardScale → PCA → MinMax [0, 2π]
- Implemented `src/kernels/base.py`: BaseKernel ABC + validate_kernel_matrix() checking symmetry, PSD, unit diagonal, [0,1] bounds
- Implemented `src/kernels/feature_maps/zz.py`: Custom ZZ feature map matching Havlíček et al. (2019). Uses H → P(2*x_i) → CX-P(2*(π-x_i)(π-x_j))-CX structure. Cross-validated against Qiskit's built-in `zz_feature_map()` — exact match.
- Implemented `src/kernels/quantum.py`: QuantumKernel computing K(x1,x2) = |⟨0|U†(x2)U(x1)|0⟩|² via Statevector simulation. Exploits symmetry for K(X,X).
- Implemented `src/kernels/estimation.py`: KernelEstimator with .npy caching and JSON metadata sidecars
- Wrote 38 tests covering all modules (loader, transforms, base validation, ZZ feature map structure + cross-validation, quantum kernel properties)
- Verification script: loads real dataset (284,807 transactions), preprocesses, computes 20×20 kernel matrix — ALL CHECKS PASS
- Kernel matrix performance: ~1000 entries/sec on statevector simulator (5 qubits)

**Decisions Made:**
- Used Qiskit 2.3.0 (latest). The `ZZFeatureMap` class is deprecated; used `zz_feature_map()` function for cross-validation
- Feature map uses P (phase) gates matching Qiskit convention: P(2*x_i) single-qubit, CX-P-CX for ZZ entangling
- Added `certifi` dependency to fix macOS SSL certificate issue with OpenML downloads
- Kernel diagonal is set to exactly 1.0 (not computed) since K(x,x) = 1 is guaranteed analytically

**Issues Encountered:**
- macOS SSL certificate verification fails for OpenML API. Fixed by adding `certifi` and patching `ssl._create_default_https_context` in the loader
- Qiskit 2.3.0 deprecates `ZZFeatureMap` class in favor of `zz_feature_map()` function. Both still work, used function for tests.

**Next Steps:**
- Phase 2: Implement IQP, covariant, and hardware-efficient feature maps
- Phase 2: Add shot-based kernel estimation (not just statevector)
- Phase 2: Implement classical kernel baselines (RBF, polynomial) behind same BaseKernel interface

---

## Architecture Decisions Log

| Decision | Choice | Rationale | Date |
|----------|--------|-----------|------|
| Package manager | uv v0.10.2 | Speed, modern Python packaging | 2026-02-15 |
| Quantum framework | Qiskit 2.3.0 | Latest, best IBM hardware integration | 2026-02-15 |
| Angle encoding range | [0, 2π] | Maximum expressiveness for rotation gates | 2026-02-15 |
| Preprocessing order | StandardScale → PCA → MinMaxToAngles | Standard practice, preserves variance structure | 2026-02-15 |
| Kernel caching format | .npy + JSON sidecar | Fast load for arrays, human-readable metadata | 2026-02-15 |
| ZZ gate convention | P gates (not RZ) | Matches Qiskit built-in, identical statevectors | 2026-02-15 |
| SSL fix | certifi + ssl context patch | Required for macOS OpenML access | 2026-02-15 |
| IQP gate convention | RZ gates + raw product | Distinct from ZZ, matches IQP circuit definition | 2026-02-15 |
| Covariant entanglement | Ring only (no linear/full) | Follows group-theoretic motivation | 2026-02-15 |
| HW-efficient gate set | RZ, SX, CX only | Zero transpilation on IBM Eagle/Heron | 2026-02-15 |
| Sampler API | AerSimulator().run() | Most reliable Qiskit 2.x path for shot-based | 2026-02-15 |
| Classical kernels | sklearn pairwise | Efficient, well-tested implementations | 2026-02-15 |
| IBM runner | qiskit-ibm-runtime 0.45.1 | Official IBM Quantum SDK, SamplerV2 | 2026-02-15 |
| Noise model | Depolarizing with gate-specific errors | Simple, configurable, physically motivated | 2026-02-15 |
| Error rate ratios | 2q=10×1q, ro=2×1q | Standard IBM hardware ratios | 2026-02-15 |
| PSD correction | Eigenvalue clipping (primary) | Minimal perturbation to matrix | 2026-02-15 |
| ZNE extrapolation | Linear (primary) | Robust, no convergence issues | 2026-02-15 |
| Token security | Environment variable only | Never hardcoded, never logged | 2026-02-15 |
| Score convention | Higher = more anomalous | Consistent across all models; sklearn models negated | 2026-02-15 |
| KPCA scoring | Projection norm deficit | Centroid distance fails for degenerate kernel values | 2026-02-15 |
| Autoencoder | PyTorch, 3-layer MLP | Simple baseline, not the focus of the project | 2026-02-15 |

---

## Known Issues & Tech Debt
- SSL certificate fix in `loader.py` patches a global `ssl` context — could be scoped more narrowly
- Polynomial kernel diagonal is not 1.0 — expected behavior, documented in class docstring
- Classical kernel `compute_entry` uses gamma from previous `compute_matrix` call if available — works correctly but ordering matters
- KPCA with classical kernels shows poor AUROC (< 0.2) — the projection norm scoring doesn't work well for non-unit-diagonal kernels (Polynomial) or when normal/anomaly projections are close
- RuntimeWarning in metrics when P+R=0 at some thresholds — handled by np.where but warning still prints

---

## File Manifest (key files and their status)

| File | Status | Description |
|------|--------|-------------|
| `src/data/loader.py` | ✅ Complete | OpenML credit card fraud loader + parquet cache + anomaly split |
| `src/data/transforms.py` | ✅ Complete | QuantumPreprocessor: StandardScale → PCA → [0, 2π] |
| `src/kernels/base.py` | ✅ Complete | BaseKernel ABC + validate_kernel_matrix() |
| `src/kernels/feature_maps/base.py` | ✅ Complete | BaseFeatureMap ABC (n_qubits, name, reps, total_gate_count, build_circuit) |
| `src/kernels/feature_maps/zz.py` | ✅ Complete | ZZ feature map (Havlíček et al.) — inherits BaseFeatureMap |
| `src/kernels/feature_maps/iqp.py` | ✅ Complete | IQP feature map — H, RZ, CX-RZ-CX per rep |
| `src/kernels/feature_maps/covariant.py` | ✅ Complete | Covariant feature map — RY, ring CX, RZ per rep |
| `src/kernels/feature_maps/hardware_efficient.py` | ✅ Complete | HW-efficient — RZ-SX-RZ, linear CX (native gates only) |
| `src/kernels/quantum.py` | ✅ Complete | QuantumKernel: statevector + sampler backends |
| `src/kernels/classical.py` | ✅ Complete | RBFKernel + PolynomialKernel (sklearn-backed) |
| `src/kernels/estimation.py` | ✅ Complete | KernelEstimator with .npy caching |
| `src/kernels/factory.py` | ✅ Complete | Kernel factory: create_quantum_kernel, create_classical_kernel, create_all_kernels |
| `src/models/ocsvm.py` | ✅ Complete | QuantumOCSVM: One-Class SVM with precomputed kernels |
| `src/models/kpca.py` | ✅ Complete | KernelPCAAnomalyDetector: Kernel PCA with projection norm scoring |
| `src/models/baselines.py` | ✅ Complete | IsolationForest, Autoencoder (PyTorch), LOF baselines |
| `src/utils/metrics.py` | ✅ Complete | AnomalyMetrics, compute_anomaly_metrics, compute_metrics_table |
| `configs/experiments/kernel_comparison.yaml` | ✅ Complete | All 6 kernels + models + baselines config |
| `experiments/run_experiment.py` | ✅ Complete | Full benchmark orchestration with YAML config |
| `scripts/verify_phase1.py` | ✅ Complete | Phase 1 end-to-end verification |
| `scripts/verify_phase2.py` | ✅ Complete | Phase 2 verification: 6 kernels + shot noise analysis |
| `scripts/verify_phase3.py` | ✅ Complete | Phase 3 verification: ML pipeline (3.6s) |
| `tests/*` | ✅ Complete | 317 tests, all passing |
| `src/analysis/alignment.py` | ✅ Complete | KTA and centered KTA (Cristianini et al.) |
| `src/analysis/expressibility.py` | ✅ Complete | Effective dimension, participation ratio, eigenspectrum |
| `src/analysis/geometric.py` | ✅ Complete | Geometric difference metric (Huang et al.) |
| `src/data/synthetic.py` | ✅ Complete | Quantum advantage dataset generation |
| `experiments/run_synthetic_benchmark.py` | ✅ Complete | Synthetic classification benchmark |
| `experiments/run_analysis.py` | ✅ Complete | Full expressibility analysis on fraud data |
| `scripts/verify_phase4.py` | ✅ Complete | Phase 4 verification (2.1s) |
| `tests/test_analysis/*` | ✅ Complete | 56 tests for alignment, expressibility, geometric, synthetic |
| `src/analysis/noise.py` | ✅ Complete | Noise impact study: noisy kernel matrix, fidelity metrics, noise sweep |
| `src/analysis/psd_correction.py` | ✅ Complete | PSD projection: clip, nearest, shift methods + violation analysis |
| `src/analysis/zne.py` | ✅ Complete | Zero-noise extrapolation: linear/exponential, entry + matrix level |
| `src/hardware/ibm_runner.py` | ✅ Complete | IBMQuantumRunner + LocalNoiseRunner |
| `src/hardware/noise_models.py` | ✅ Complete | Depolarizing noise models, noise sweep, real backend fetch |
| `configs/hardware/noise_sweep.yaml` | ✅ Complete | Noise sweep + ZNE experiment config |
| `experiments/run_hardware.py` | ✅ Complete | Full noise study experiment (simulation + hardware modes) |
| `scripts/verify_phase5.py` | ✅ Complete | Phase 5 verification (0.5s) |
| `tests/test_hardware/*` | ✅ Complete | 20 tests for noise models + IBM runner |
| `tests/test_analysis/test_psd_correction.py` | ✅ Complete | 15 tests for PSD correction |
| `tests/test_analysis/test_zne.py` | ✅ Complete | 14 tests for ZNE |
| `tests/test_analysis/test_noise_study.py` | ✅ Complete | 9 tests for noise study |
| `tests/test_kernels/test_quantum_noisy.py` | ✅ Complete | 9 tests for noisy QuantumKernel |
| `dashboard/export_data.py` | ✅ Complete | Export Phase 3-5 results to JSON + JS data bundle |
| `dashboard/index.html` | ✅ Complete | Single-file React + Plotly.js dashboard (6 tabs, dark theme) |
| `dashboard/data_bundle.js` | ✅ Complete | Auto-generated JS with all data embedded (54.4 KB) |
| `tests/test_dashboard/test_export.py` | ✅ Complete | 24 tests for data export |
| `scripts/verify_phase6.py` | ✅ Complete | Phase 6 verification |
| `scripts/run_all.sh` | ✅ Complete | End-to-end pipeline runner |
| `README.md` | ✅ Complete | Comprehensive project README |
| `LICENSE` | ✅ Complete | MIT License |
