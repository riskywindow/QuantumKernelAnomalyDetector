# PROGRESS.md — Quantum Kernel Anomaly Detection

> This file is updated at the end of every Claude Code session. Read this FIRST to understand current project state.

---

## Project Status: ✅ Phase 2 — Complete

### Completed Phases
- **Phase 1: Foundation** — Completed 2026-02-15
- **Phase 2: Kernel Engine Expansion** — Completed 2026-02-15

### Current Phase: Phase 3 — ML Pipeline (Not Started)
**Objective:** One-Class SVM, Kernel PCA anomaly detection, classical baselines, full benchmark suite with AUROC/AUPRC/F1.

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

### Phase 2 Exit Criteria
- [x] BaseFeatureMap ABC exists and all four feature maps inherit from it
- [x] QuantumKernel accepts any BaseFeatureMap, not just ZZFeatureMap
- [x] All Phase 1 tests still pass (no regressions) — 38/38
- [x] IQP, Covariant, and Hardware-Efficient feature maps produce valid quantum circuits
- [x] Hardware-Efficient uses ONLY RZ, SX, CX gates
- [x] Shot-based sampler backend works and produces kernel estimates within noise tolerance of statevector
- [x] RBF and Polynomial classical kernels produce valid kernel matrices
- [x] All new tests pass — 123/123 total
- [x] Verification script runs end-to-end and produces both comparison heatmap and shot noise plot

### Phase 1 Task Checklist
- [x] Task 1.1: Project setup (uv, directory structure, pyproject.toml, .gitignore, git init)
- [x] Task 1.2: Data loading & preprocessing (loader.py, transforms.py)
- [x] Task 1.3: Abstract kernel interface (base.py with validation)
- [x] Task 1.4: ZZ Feature Map implementation (zz.py)
- [x] Task 1.5: Quantum kernel + statevector simulation (quantum.py, estimation.py)
- [x] Task 1.6: Tests + verification script (38 tests pass, heatmap generated)

---

## Session Log

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

---

## Known Issues & Tech Debt
- SSL certificate fix in `loader.py` patches a global `ssl` context — could be scoped more narrowly
- Polynomial kernel diagonal is not 1.0 — expected behavior, documented in class docstring
- Classical kernel `compute_entry` uses gamma from previous `compute_matrix` call if available — works correctly but ordering matters

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
| `configs/experiments/kernel_comparison.yaml` | ✅ Complete | All 6 kernels + sampler config |
| `scripts/verify_phase1.py` | ✅ Complete | Phase 1 end-to-end verification |
| `scripts/verify_phase2.py` | ✅ Complete | Phase 2 verification: 6 kernels + shot noise analysis |
| `tests/*` | ✅ Complete | 123 tests, all passing |
| `src/models/ocsvm.py` | ⬜ Phase 3 | One-class SVM |
| `src/models/kpca.py` | ⬜ Phase 3 | Kernel PCA |
| `src/models/baselines.py` | ⬜ Phase 3 | Classical baselines |
| `src/utils/metrics.py` | ⬜ Phase 3 | AUROC, AUPRC, F1 |
| `src/analysis/*` | ⬜ Phase 4-5 | Expressibility, noise analysis |
| `src/hardware/*` | ⬜ Phase 5 | IBM Quantum integration |
