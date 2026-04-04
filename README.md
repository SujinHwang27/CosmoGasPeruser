# CosmoGasPeruser

ML-driven analysis of diffuse gas in cosmological simulations using unsupervised clustering of sightline separability vectors.

## Overview

This project analyzes the Sherwood cosmological simulation (z=0.3) to discover physically meaningful spectral features across 4 physics feedback classes:
- **Class 1**: NoFeedback
- **Class 2**: StellarWind
- **Class 3**: WindAGN
- **Class 4**: WindStrongAGN

It uses a novel **micro-probing** technique (RBF SVM decision distances) to encode each sightline's class separability into a 24-dimensional vector, then clusters these vectors to map the separability landscape of the spectrum.

## Project Structure
```
CosmoGasPeruser/
├── src/core/           # Core library (all reusable logic)
│   ├── data.py         # Data ingestion (DataIngestor, SignalClusteringData)
│   ├── probe.py        # RBF SVM micro-probing (24-dim separability vectors)
│   ├── cluster.py      # K-Means with k-sweep and silhouette analysis
│   ├── viz.py          # UMAP, spatial maps, comparison plots
│   ├── audit.py        # Cross-run overlap and feature attribution
│   └── models/         # ML models (Random Forest classifiers)
├── scripts/            # Thin orchestration wrappers (stages 1-6)
│   └── run_all.py      # Master pipeline orchestrator
├── data/               # Dataset storage (DVC-tracked)
├── results/            # Pipeline outputs (CSVs, plots, figures)
├── configs/            # Experiment YAML configurations
├── docs/               # Design documents and analysis reports
├── tests/              # Unit tests (pytest)
├── dvc.yaml            # DVC pipeline definition (6 stages)
└── pyproject.toml      # Python project metadata (uv)
```

## Setup

```bash
# Install dependencies
uv sync

# Pull data (requires DVC remote access)
dvc pull

# Copy .env.example to .env and fill in credentials
cp .env.example .env
```

## Running the Pipeline

```bash
# Via DVC (preferred — tracks dependencies, skips unchanged stages)
dvc repro

# Via orchestrator (always re-runs all stages)
PYTHONPATH=. uv run python scripts/run_all.py --stages 1,2,3,4,5,6 --run both --k 5

# Run specific stages only
PYTHONPATH=. uv run python scripts/run_all.py --stages 3,4 --run wavelet --k 5
```

### Pipeline Stages
| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `run_prep.py` | Load & normalize wavelet/absorption features |
| 2 | `run_probe.py` | RBF SVM micro-probing (24-dim separability vectors) |
| 3 | `run_cluster.py` | K-Means clustering with elbow analysis |
| 4 | `run_viz.py` | UMAP 2D/3D + spatial cluster maps |
| 5 | `run_audit.py` | Cross-run overlap and feature attribution |
| 6 | `run_rf.py` | Per-cluster Random Forest classifiers |

## Testing

```bash
uv run pytest tests/
```

## Configuration

Experiments are defined in YAML (`configs/`):
```yaml
name: "my_experiment"
data: "path/to/data"
transforms:
  - type: "dct"
    params: { mode: "full" }
model:
  type: "transformer"
  params: { lr: 0.001, epochs: 10 }
```

---
Contact: **Sujin Hwang** (sujinhwang000@gmail.com)
