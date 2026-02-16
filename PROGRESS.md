# PROGRESS.md — Quantum Kernel Anomaly Detection

> This file is updated at the end of every Claude Code session. Read this FIRST to understand current project state.

---

## Project Status: ✅ Phase 1 — Complete

### Completed Phases
- **Phase 1: Foundation** — Completed 2026-02-15

### Current Phase: Phase 2 — Kernel Engine Expansion (Not Started)
**Objective:** IQP, covariant, and hardware-efficient feature maps. Shot-based kernel estimation. Classical kernel baselines.

### Phase 1 Task Checklist
- [x] Task 1.1: Project setup (uv, directory structure, pyproject.toml, .gitignore, git init)
- [x] Task 1.2: Data loading & preprocessing (loader.py, transforms.py)
- [x] Task 1.3: Abstract kernel interface (base.py with validation)
- [x] Task 1.4: ZZ Feature Map implementation (zz.py)
- [x] Task 1.5: Quantum kernel + statevector simulation (quantum.py, estimation.py)
- [x] Task 1.6: Tests + verification script (38 tests pass, heatmap generated)

### Phase 1 Exit Criteria
- [x] Full project directory structure exists with all placeholder files
- [x] Data loader downloads real credit card fraud dataset from OpenML with proper train/test split
- [x] Preprocessing pipeline: scale → PCA → angle encoding, with fit/transform pattern
- [x] ZZ feature map builds correct circuits (verified against Qiskit built-in)
- [x] Quantum kernel computes valid kernel matrices on statevector simulator
- [x] 20×20 kernel matrix passes all validation checks (symmetric, PSD, diagonal=1, bounded)
- [x] All tests pass (38/38)
- [x] Verification script runs end-to-end and produces heatmap

---

## Session Log

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

---

## Known Issues & Tech Debt
- SSL certificate fix in `loader.py` patches a global `ssl` context — could be scoped more narrowly
- Unused import `train_test_split` was removed from loader.py

---

## File Manifest (key files and their status)

| File | Status | Description |
|------|--------|-------------|
| `src/data/loader.py` | ✅ Complete | OpenML credit card fraud loader + parquet cache + anomaly split |
| `src/data/transforms.py` | ✅ Complete | QuantumPreprocessor: StandardScale → PCA → [0, 2π] |
| `src/kernels/base.py` | ✅ Complete | BaseKernel ABC + validate_kernel_matrix() |
| `src/kernels/feature_maps/zz.py` | ✅ Complete | ZZ feature map (Havlíček et al.) |
| `src/kernels/quantum.py` | ✅ Complete | QuantumKernel with statevector simulation |
| `src/kernels/estimation.py` | ✅ Complete | KernelEstimator with .npy caching |
| `scripts/verify_phase1.py` | ✅ Complete | End-to-end verification (ALL CHECKS PASS) |
| `tests/*` | ✅ Complete | 38 tests, all passing |
| `src/kernels/feature_maps/iqp.py` | ⬜ Phase 2 | IQP feature map |
| `src/kernels/feature_maps/covariant.py` | ⬜ Phase 2 | Covariant kernels |
| `src/kernels/feature_maps/hardware_efficient.py` | ⬜ Phase 2 | HW-efficient ansatz |
| `src/kernels/classical.py` | ⬜ Phase 3 | Classical kernel baselines |
| `src/models/ocsvm.py` | ⬜ Phase 3 | One-class SVM |
| `src/models/kpca.py` | ⬜ Phase 3 | Kernel PCA |
| `src/models/baselines.py` | ⬜ Phase 3 | Classical baselines |
| `src/analysis/*` | ⬜ Phase 4-5 | Expressibility, noise analysis |
| `src/hardware/*` | ⬜ Phase 5 | IBM Quantum integration |
