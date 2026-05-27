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
├── CLAUDE.md           # Project kernel — conventions, points at the active LEDGER
├── src/core/           # Core library (all reusable logic)
│   ├── data.py         # Data ingestion (DataIngestor, SignalClusteringData)
│   ├── probe.py        # RBF SVM micro-probing (24-dim separability vectors)
│   ├── cluster.py      # K-Means with k-sweep and silhouette analysis
│   ├── viz.py          # UMAP, spatial maps, comparison plots
│   ├── audit.py        # Cross-run overlap and feature attribution
│   └── models/         # ML models (Random Forest classifiers)
├── scripts/            # Thin orchestration wrappers (stages 1-6)
│   └── run_all.py      # Master pipeline orchestrator
├── experiments/        # Per-track command centers (Research-OS)
│   └── <track>/LEDGER.md   # Single source of truth per experiment track
├── .claude/            # Agentic OS: agents/, commands/, skills/
├── templates/          # Scaffold for new experiment tracks
├── data/               # Dataset storage (DVC-tracked)
├── results/            # Pipeline outputs (CSVs, plots, figures)
├── docs/               # Design documents and analysis reports (historical archive)
├── tests/              # Unit tests (pytest)
├── dvc.yaml            # DVC pipeline definition (6 stages)
└── pyproject.toml      # Python project metadata (uv)
```

This repo follows the **Research-OS** discipline: each methodology is isolated on its own branch with an `experiments/<track>/LEDGER.md` (7-section command center: Pulse / Methodology / Logic / Data / Evaluation / Visualization / History). Specialist agents under `.claude/agents/` (PI, data, core, infra, analysis, paper, defense-panel) are coordinated through the active LEDGER. See `CLAUDE.md`.

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

## Experiment tracks

Each methodology is isolated on its own branch with a command-center LEDGER:

```bash
# Scaffold a new track (creates exp/<name> branch + experiments/<name>/LEDGER.md)
/new-experiment <name>          # in Claude Code

# Update the active track's LEDGER with the session's progress
/update-ledger                  # in Claude Code
```

Active track: `signal-clustering-v2` — see `experiments/signal-clustering-v2/LEDGER.md`. Reconstructed LEDGERs for the historical tracks (`eda-sherwood`, `baseline-random-forest`, `signal-clustering-v1`) live alongside it. MLflow runs are branch-aware (`CosmoGasPeruser/<branch>`) and synced across machines via DVC (`mlruns.dvc`).

---
Contact: **Sujin Hwang** (sujinhwang000@gmail.com)
