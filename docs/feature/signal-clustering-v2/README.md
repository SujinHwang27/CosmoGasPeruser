# Signal Clustering v2 — Production Pipeline

**Branch:** `feature/signal-clustering-v2` (tag: `v0.4-clustering-v2`)

## What This Experiment Does

6-stage DVC pipeline that clusters sightlines by their **24-dimensional separability vectors** (RBF SVM decision distances, intercepts dropped). Runs two parallel pipelines (wavelet features and raw absorption) and cross-validates their cluster assignments.

## Key Design Decisions (from v1 findings)

| Decision | Rationale |
|----------|-----------|
| 24-dim vectors (drop intercepts) | Intercepts encode scaling artifacts, not physics |
| K=5 default (sweep K=2-20) | Empirical elbow + silhouette analysis |
| Dual wavelet + raw runs | Cross-run audit validates representation-independence |
| DVC pipeline | Reproducible, dependency-tracked, skips unchanged stages |
| Per-level wavelet normalization | Z-score per level prevents D5/D6/A6 from dominating |

## Pipeline Stages

| Stage | Script | Core Module | Output |
|-------|--------|-------------|--------|
| 1. Prepare | `run_prep.py` | `data.py` | Normalized wavelet arrays + absorption field |
| 2. Probe | `run_probe.py` | `probe.py` | Separability vectors (16384 x 24) |
| 3. Cluster | `run_cluster.py` | `cluster.py` | Labels + centroids, K-sweep CSVs |
| 4. Visualize | `run_viz.py` | `viz.py` | UMAP 2D/3D, spatial maps, comparisons |
| 5. Audit | `run_audit.py` | `audit.py` | Cross-run contingency, overlap summary |
| 6. RF | `run_rf.py` | `rf_classifier.py` | Per-cluster Random Forest classifiers |

## Files

| File | Content |
|------|---------|
| `signal_clustering_project_plan_v4.md` | Full design spec — data assets, design decisions (D1-D3), stage-by-stage implementation plan, expected outputs |

## Caveats

- The plan references `src/cluster.py` and `src/viz.py` — actual locations are `src/core/cluster.py` and `src/core/viz.py`
- Plan proposes K=8 as primary with K=5 as stability check; actual pipeline uses K=5 as default after empirical results
- Plan mentions "fingerprints" throughout; current terminology is "separability vectors"
