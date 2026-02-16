# CLAUDE.md — Quantum Kernel Anomaly Detection

## IMPORTANT: Session Protocol

**At the start of every session:**
1. Read this entire CLAUDE.md file
2. Read PROGRESS.md to understand current state
3. Identify which phase/task to work on next
4. Begin implementation

**At the end of every session:**
1. Update PROGRESS.md with everything accomplished, decisions made, issues encountered, and what comes next
2. Ensure all code is tested and working
3. Commit all changes with descriptive commit messages

---

## Project Overview

We are building **quantum-kernel-anomaly**: a hybrid quantum-classical anomaly detection system that uses parameterized quantum circuits as feature maps to compute kernel functions for financial fraud detection.

### What This Project Does
1. Encodes classical data points into quantum states via parameterized circuits (quantum feature maps)
2. Computes kernel matrices K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|² measuring quantum state overlap
3. Feeds these kernel matrices into classical ML models (One-Class SVM, Kernel PCA) for anomaly detection
4. Benchmarks quantum kernels against classical kernels (RBF, polynomial) on credit card fraud data
5. Runs expressibility analysis proving what quantum kernels can represent that classical kernels cannot
6. Executes on real IBM Quantum hardware with noise analysis and error mitigation
7. Provides an interactive React dashboard for visualization

### Why This Matters
Quantum kernel methods have the strongest theoretical evidence for quantum advantage in ML (Huang et al. 2021 Nature, Liu et al. 2021 Nature). This project provides rigorous empirical validation on both real-world and synthetic data.

---

## Tech Stack & Dependencies

### Core Python Dependencies
```
python = "^3.11"
qiskit = "^1.3"
qiskit-aer = "^0.15"
qiskit-ibm-runtime = "^0.34"
qiskit-machine-learning = "^0.8"
pennylane = "^0.39"
scikit-learn = "^1.6"
torch = "^2.5"
numpy = "^1.26"
scipy = "^1.14"
pandas = "^2.2"
pyarrow = "^18.0"     # For parquet caching of downloaded datasets
```

### Analysis & Visualization
```
matplotlib = "^3.9"
seaborn = "^0.13"
plotly = "^5.24"
```

### Experiment Management
```
pyyaml = "^6.0"
wandb = "^0.19"       # Weights & Biases for experiment tracking
tqdm = "^4.67"
```

### Dashboard (Phase 6)
```
React, Plotly.js, FastAPI (later phase — do not install yet)
```

### Development Tools
```
pytest = "^8.3"
pytest-cov = "^6.0"
ruff = "^0.8"          # Linting and formatting
mypy = "^1.13"         # Type checking
pre-commit = "^4.0"
```

### Package Manager
Use **uv** for Python package management. It is significantly faster than pip/poetry.
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project
uv init
uv add qiskit qiskit-aer numpy scipy pandas pyarrow scikit-learn matplotlib seaborn plotly pyyaml tqdm
uv add --dev pytest pytest-cov ruff mypy
```

If uv is unavailable, fall back to pip:
```bash
pip install qiskit qiskit-aer numpy scipy pandas pyarrow scikit-learn matplotlib seaborn plotly pyyaml tqdm --break-system-packages
pip install pytest pytest-cov ruff mypy --break-system-packages
```

---

## Project Structure

```
quantum-kernel-anomaly/
├── CLAUDE.md                 # This file — project instructions
├── PROGRESS.md               # Session-by-session progress tracking
├── README.md                 # Project README (create in final phase)
├── pyproject.toml            # Project config and dependencies
├── configs/
│   ├── experiments/          # YAML experiment configs
│   │   └── default.yaml      # Default experiment configuration
│   └── hardware/             # IBM backend configs, noise models
│       └── simulator.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Dataset loading + preprocessing
│   │   ├── synthetic.py      # Synthetic advantage dataset generation (Phase 4)
│   │   └── transforms.py     # Feature scaling, dim reduction, angle encoding
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract kernel interface
│   │   ├── quantum.py        # Quantum kernel (wraps feature maps + backend)
│   │   ├── classical.py      # RBF, polynomial, etc. for baselines (Phase 3)
│   │   ├── feature_maps/
│   │   │   ├── __init__.py
│   │   │   ├── zz.py         # ZZFeatureMap (Phase 1)
│   │   │   ├── iqp.py        # IQP circuits (Phase 2)
│   │   │   ├── covariant.py  # Covariant kernels (Phase 2)
│   │   │   └── hardware_efficient.py  # HW-efficient ansatz (Phase 2)
│   │   └── estimation.py     # Kernel matrix computation, caching, batching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ocsvm.py          # One-class SVM wrapper (Phase 3)
│   │   ├── kpca.py           # Kernel PCA anomaly detector (Phase 3)
│   │   └── baselines.py      # Isolation forest, XGBoost, autoencoder (Phase 3)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── expressibility.py # Effective dimension, kernel alignment (Phase 4)
│   │   ├── geometric.py      # Geometric difference metric (Phase 4)
│   │   ├── noise.py          # Noise impact study, ZNE (Phase 5)
│   │   └── psd_correction.py # Kernel matrix PSD projection (Phase 5)
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── ibm_runner.py     # IBM Quantum job submission (Phase 5)
│   │   └── noise_models.py   # Realistic noise model construction (Phase 5)
│   └── utils/
│       ├── __init__.py
│       ├── caching.py        # Kernel matrix caching
│       ├── metrics.py        # AUROC, AUPRC, F1 (Phase 3)
│       └── plotting.py       # Visualization helpers
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   │   ├── __init__.py
│   │   ├── test_loader.py
│   │   └── test_transforms.py
│   ├── test_kernels/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_quantum.py
│   │   └── test_zz_feature_map.py
│   └── conftest.py           # Shared test fixtures
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_kernel_verification.ipynb
├── experiments/
│   ├── run_experiment.py     # Main experiment runner (Phase 3+)
│   └── results/              # Cached kernel matrices, model outputs
├── data/
│   └── raw/                  # Downloaded datasets (gitignored)
└── .gitignore
```

---

## Code Style & Conventions

### Python Style
- **Type hints everywhere** — all function signatures must have type annotations
- **Docstrings** — Google style docstrings for all public functions and classes
- **Abstract base classes** — use ABC for interfaces (kernel base, feature map base)
- **Dataclasses or Pydantic** — for configuration objects, never raw dicts
- **f-strings** — for string formatting
- **pathlib.Path** — never use os.path
- Max line length: 100 characters
- Use ruff for formatting and linting

### Architecture Principles
- **Interface-first design**: Define abstract interfaces before implementations
- **Dependency injection**: Kernels take backends as constructor args, models take kernel matrices
- **Configuration as data**: All hyperparameters in YAML configs, never hardcoded
- **Aggressive caching**: Kernel matrices are expensive — cache to disk with metadata
- **Reproducibility**: Set random seeds everywhere, log all experiment parameters

### Testing Standards
- Every module gets a corresponding test file
- Test with small examples first (2-3 qubits, 5-10 samples)
- Mathematical verification: for small cases, compute expected kernel values by hand/analytically
- Property-based tests: kernel matrices must be symmetric, PSD, diagonal = 1

### Git Conventions
- Commit after each logical unit of work
- Commit messages: `phase-N: description` (e.g., `phase-1: implement ZZ feature map`)
- Branch per phase if desired, but main is fine for solo dev

---

## Phase 1: Foundation (THIS SESSION)

### Objective
Set up the complete project structure, implement data loading and preprocessing for the credit card fraud dataset, implement the abstract kernel interface and the ZZ feature map, compute a small kernel matrix on a statevector simulator, and verify correctness.

### Tasks

#### Task 1.1: Project Setup
- [ ] Initialize the project with uv (or pip fallback)
- [ ] Create the full directory structure as shown above (create __init__.py files, empty placeholder files for future phases)
- [ ] Set up pyproject.toml with all dependencies
- [ ] Set up .gitignore (include data/raw/, experiments/results/, __pycache__, .pyc, *.npy, .env)
- [ ] Initialize git repository
- [ ] Create a minimal conftest.py with shared fixtures

#### Task 1.2: Data Loading & Preprocessing
- [ ] Implement `src/data/loader.py`:
  - Function `load_credit_card_fraud(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]`
  - Download the dataset via OpenML (no authentication required): `sklearn.datasets.fetch_openml('creditcard', version=1, as_frame=True, parser='auto')`. This returns the real Credit Card Fraud dataset (284,807 transactions, 492 frauds, 30 features). Cache it to `data_dir` as a parquet file after first download so subsequent loads are instant.
  - Do NOT implement any synthetic fallback — we use the real dataset only. If OpenML fails, we troubleshoot.
  - Function `prepare_anomaly_split(X, y, normal_train_size, test_size, seed)` that:
    - Separates normal (y=0) and anomaly (y=1) samples
    - Creates a training set of ONLY normal samples (this is how one-class SVM works)
    - Creates a test set with both normal and anomaly samples (stratified)
    - Returns: X_train, X_test, y_test (y_train is all zeros, not needed)

- [ ] Implement `src/data/transforms.py`:
  - Class `QuantumPreprocessor` with methods:
    - `fit_transform(X: np.ndarray, n_features: int) -> np.ndarray`: Full pipeline
    - `_standard_scale(X)`: Zero mean, unit variance
    - `_reduce_dimensions(X, n_features, method='pca')`: PCA to n_features dimensions. Support 'pca' method, with a hook for 'autoencoder' later.
    - `_rescale_to_angles(X, range_min=0, range_max=2*np.pi)`: Rescale features to [0, 2π] for use as rotation angles in quantum circuits. Use min-max scaling per feature.
    - `transform(X)`: Apply fitted pipeline to new data (for test set)
  - Must be sklearn-pipeline-compatible (fit/transform pattern)
  - Store fitted parameters (scaler mean/std, PCA components, min/max values) for consistent train→test transformation

#### Task 1.3: Abstract Kernel Interface
- [ ] Implement `src/kernels/base.py`:
  - Abstract class `BaseKernel(ABC)`:
    - `compute_matrix(X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray`: Compute the kernel matrix. If X2 is None, compute K(X1, X1).
    - `compute_entry(x1: np.ndarray, x2: np.ndarray) -> float`: Compute a single kernel entry.
    - `name: str` property
  - Validation helper `validate_kernel_matrix(K: np.ndarray) -> dict[str, bool]`:
    - Check symmetry: K == K.T (within tolerance)
    - Check PSD: all eigenvalues >= -tolerance
    - Check diagonal: all K[i,i] ≈ 1.0 (for normalized kernels)
    - Check bounds: all 0 <= K[i,j] <= 1 (for quantum kernels specifically)
    - Return dict of {property: pass/fail}

#### Task 1.4: ZZ Feature Map
- [ ] Implement `src/kernels/feature_maps/zz.py`:
  - Class `ZZFeatureMap`:
    - Constructor: `__init__(self, n_qubits: int, reps: int = 2, entanglement: str = 'linear')`
    - Method: `build_circuit(x: np.ndarray) -> QuantumCircuit`: Build the parameterized circuit encoding data point x
    - The ZZ feature map structure (per rep):
      1. H gate on all qubits
      2. Single-qubit RZ rotations: RZ(xᵢ) on qubit i
      3. Two-qubit entangling: RZZ(xᵢ · xⱼ) on connected pairs (linear or full entanglement)
    - Use Qiskit's `QuantumCircuit` to build circuits
  - NOTE: Qiskit has a built-in `ZZFeatureMap` in `qiskit.circuit.library`. You may use it as reference or wrap it, but implement the core circuit construction yourself so we understand and control every gate. Then validate against Qiskit's version.

#### Task 1.5: Quantum Kernel with Statevector Simulation
- [ ] Implement `src/kernels/quantum.py`:
  - Class `QuantumKernel(BaseKernel)`:
    - Constructor: `__init__(self, feature_map: ZZFeatureMap, backend: str = 'statevector')`
    - `compute_entry(x1, x2)`: 
      - Build circuit: U(x1) then U†(x2), measure probability of |0...0⟩
      - For statevector backend: use Qiskit Aer StatevectorSimulator, get exact probability
      - Return |⟨0|U†(x2)U(x1)|0⟩|²
    - `compute_matrix(X1, X2)`:
      - Efficiently compute all kernel entries
      - Exploit symmetry when X2 is None: only compute upper triangle
      - Show progress with tqdm
      - Return the full kernel matrix

- [ ] Implement `src/kernels/estimation.py`:
  - Class `KernelEstimator`:
    - Wraps kernel computation with caching
    - Method: `estimate(kernel, X1, X2, cache_key) -> np.ndarray`
    - Caching: Save/load kernel matrices as .npy files with JSON metadata sidecar (feature map config, backend, data hash, timestamp)
    - Cache directory: `experiments/results/kernels/`

#### Task 1.6: Verification & Testing
- [ ] Implement tests in `tests/`:
  - `test_transforms.py`:
    - Test that preprocessing preserves data shape
    - Test that output features are in [0, 2π]
    - Test that fit_transform → transform on new data is consistent
  - `test_base.py`:
    - Test kernel matrix validation on known valid/invalid matrices
  - `test_zz_feature_map.py`:
    - Test circuit construction for 2 qubits, verify gate count and structure
    - Test that circuit parameters match input data
  - `test_quantum.py`:
    - **CRITICAL**: 2-qubit verification. For x1 = x2, kernel should be exactly 1.0
    - For x1 ≠ x2, kernel should be in (0, 1)
    - Kernel matrix should be symmetric
    - Kernel matrix should be PSD
    - Compute a 5×5 kernel matrix on random data and validate all properties
    - Cross-validate: Compare our ZZ implementation against Qiskit's built-in ZZFeatureMap to ensure they produce identical circuits/kernels

- [ ] Create a small verification script `scripts/verify_phase1.py`:
    - Load the credit card fraud data via OpenML
    - Preprocess to 5 features
    - Take 20 samples
    - Compute 20×20 kernel matrix using ZZ feature map + statevector simulator
    - Print validation results (symmetric, PSD, diagonal=1, bounds)
    - Print a few example kernel values
    - Save a matplotlib heatmap of the kernel matrix to `experiments/results/phase1_kernel_heatmap.png`

### Exit Criteria (ALL must pass)
1. ✅ Full project directory structure exists with all placeholder files
2. ✅ Data loader downloads real credit card fraud dataset from OpenML with proper train/test split
3. ✅ Preprocessing pipeline: scale → PCA → angle encoding, with fit/transform pattern
4. ✅ ZZ feature map builds correct circuits (verified against Qiskit built-in)
5. ✅ Quantum kernel computes valid kernel matrices on statevector simulator
6. ✅ 20×20 kernel matrix passes all validation checks (symmetric, PSD, diagonal=1, bounded)
7. ✅ All tests pass
8. ✅ Verification script runs end-to-end and produces heatmap

### Common Pitfalls to Avoid
- **Don't forget the adjoint**: The kernel circuit is U(x1) followed by U†(x2), NOT U(x1) followed by U(x2). The dagger (inverse/adjoint) is critical.
- **Feature count must match qubit count**: If you PCA to 5 features, your feature map must use 5 qubits.
- **Angle encoding range matters**: Different papers use [0, π] vs [0, 2π]. We use [0, 2π] for maximum expressiveness. Be consistent.
- **Statevector gives exact probabilities**: Don't add measurement gates when using statevector simulator. Just compute the state and extract the |0...0⟩ amplitude.
- **Kernel matrix diagonal must be exactly 1.0**: K(x, x) = |⟨ψ(x)|ψ(x)⟩|² = 1 always. If it's not, your circuit construction is wrong.
- **PCA before angle encoding**: The order is standard_scale → PCA → min_max_to_angles. Not PCA → standard_scale.

---

## Future Phases (DO NOT IMPLEMENT YET — for context only)

### Phase 2: Kernel Engine Expansion
- IQP, covariant, and hardware-efficient feature maps
- Shot-based kernel estimation (not just statevector)
- Full classical kernel baselines behind same interface

### Phase 3: ML Pipeline
- One-Class SVM with precomputed kernels
- Kernel PCA anomaly detection
- Classical baselines (Isolation Forest, XGBoost, autoencoder)
- Full benchmark suite with AUROC/AUPRC/F1

### Phase 4: Expressibility Analysis
- Effective dimension computation
- Kernel target alignment
- Geometric difference metric
- Synthetic advantage dataset

### Phase 5: Hardware Execution
- IBM Quantum backend integration
- Batched job submission
- Zero-noise extrapolation
- PSD correction for noisy kernels

### Phase 6: Dashboard & Polish
- React + Plotly.js dashboard
- LaTeX writeup
- README, demo video

---

## Useful References (for understanding, not for copying code)
- Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature spaces" — Nature
- Huang et al. (2021) "Power of data in quantum machine learning" — Nature Communications  
- Liu et al. (2021) "A rigorous and robust quantum speed-up in supervised machine learning" — Nature Physics
- Abbas et al. (2021) "The power of quantum neural networks" — Nature Computational Science
- Qiskit documentation: https://docs.quantum.ibm.com/
- Qiskit Machine Learning: https://qiskit-community.github.io/qiskit-machine-learning/
