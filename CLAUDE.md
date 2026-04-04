# CosmoGasPeruser

## Project Overview
Computational astrophysics + ML research project analyzing diffuse gas in cosmological simulations (Sherwood simulation, z=0.3). Uses unsupervised clustering of sightline separability vectors to discover physically meaningful spectral features across 4 physics feedback classes (NoFeedback, StellarWind, WindAGN, WindStrongAGN).

## Architecture

### Core Library (`src/core/`)
All reusable logic lives here. Follows ABC pattern (`BaseTransformer`, `BaseModel`).
- `data.py` — Data ingestion (`DataIngestor`, `SignalClusteringData`)
- `probe.py` — RBF SVM micro-probing (24-dim separability vectors)
- `cluster.py` — K-Means with k-sweep and silhouette analysis
- `viz.py` — UMAP, spatial maps, comparison plots
- `audit.py` — Cross-run overlap, variance, feature attribution
- `models/rf_classifier.py` — Per-cluster Random Forest with GridSearchCV
- `base.py` — Abstract base classes
- `transforms.py` — PCA, DCT, Wavelet, Fisher transforms
- `provenance.py` — Git metadata capture for result traceability
- `utils.py` — MLflow cleanup, shape checks, directory helpers

### Scripts (`scripts/`)
Thin orchestration wrappers only. **No logic in scripts** — all computation delegates to `src/core/`.
- `run_all.py` — Master pipeline orchestrator (stages 1-6)
- `run_prep.py` — Stage 1: Data preparation
- `run_probe.py` — Stage 2: Micro-probing
- `run_cluster.py` — Stage 3: K-Means clustering
- `run_viz.py` — Stage 4: Visualization
- `run_audit.py` — Stage 5: Cross-run auditing
- `run_rf.py` — Stage 6: Random Forest classifiers

### Pipeline Execution
```bash
# Via DVC (preferred — tracks dependencies, skips unchanged stages)
dvc repro

# Via orchestrator (always re-runs)
PYTHONPATH=. uv run python scripts/run_all.py --stages 1,2,3,4,5,6 --run both --k 5
```

## Key Conventions

### Data
- **Do NOT modify `data/` directly** — managed by DVC
- Large files tracked via DVC, small results/docs tracked via Git
- 4 physics classes stored in directories named `1/`, `2/`, `3/`, `4/`
- Wavelet features: `data/processed/wavelet_db8_l6_d12/`
- Raw flux: `data/preprocessed/Sherwood_z0.3_inf/`

### Terminology (standardized)
- **Separability vector**: The 24-dim feature vector per sightline (6 OVO pairs x 4 class distances)
- File naming uses `fingerprints_*.npy` for historical reasons but code/docs should say "separability vector"
- **Sightline**: A single line-of-sight spectrum (2048 pixels)

### Dependencies
- Python 3.12+, managed via `uv`
- Install: `uv sync`
- Run commands: `uv run python <script>`

### DVC Pipeline
- Pipeline defined in `dvc.yaml` (6 stages)
- Dependency paths must match actual file locations in `src/core/`
- Results go to `results/signal_clustering_v2/`
- Metrics CSV files use `cache: false` so they're always readable

### MLflow
- Experiment tracking with nested parent-child runs
- Artifacts stored on S3 (`s3://cosmo-gas-peruser/mlflow-artifacts`)
- Backend store: local SQLite or remote server (see `.env.example` for options)
- Experiment names auto-generated per branch via `provenance.mlflow_experiment_name()`
- `mlruns/` and `mlartifacts/` are gitignored — NOT tracked by DVC
- See `.agent/skills/mlflow-sync-guide/SKILL.md` for multi-machine setup

### Git Workflow
- Feature branches: `feature/<name>`
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`
- Current active branch: `feature/signal-clustering-v2`
- Tag stable checkpoints: `git tag -a v0.X-name -m "description" <commit>`
- Existing tags: `v0.1-eda`, `v0.2-baseline-rf`, `v0.3-clustering-v1`, `v0.4-clustering-v2`

### Testing
- Framework: pytest
- Run: `uv run pytest tests/`
- Tests should cover `src/core/` modules

### Environment
- `.env` contains credentials — never commit secrets
- `.env` is in `.gitignore`
- `.env.example` documents required variables without real values

### CI/CD
- GitHub Actions workflow in `.github/workflows/ci.yml`
- Runs on push to `main` and `feature/**` branches
- Checks: import validation, pytest, DVC dep verification

### Provenance
- Use `src/core/provenance.py` to stamp results with git SHA, branch, timestamp
- `provenance_header(params)` — returns dict with commit, branch, dirty, timestamp + params
- `mlflow_experiment_name()` — returns branch-aware name like `CosmoGasPeruser/feature/signal-clustering-v2`
- Use `mlflow_experiment_name()` instead of hardcoded experiment names in scripts

### Legacy Code
- `src/main.py` — old config-driven orchestrator (superseded by `scripts/run_all.py`)
- `scripts/baseline/`, `scripts/clustering/`, `scripts/utils/` — old standalone scripts (superseded by pipeline stages)
- `src/core/models.py` — contains `BaselineRFClassifier`, `MicroProbingClassifier`, `SimpleTransformerClassifier` (legacy, not used by current pipeline)
