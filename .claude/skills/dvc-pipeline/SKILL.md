---
name: dvc-pipeline
description: DVC pipeline management, adding stages, debugging reproducibility
---

# DVC Pipeline Management

## When to Use
When modifying pipeline stages, adding new stages, or debugging reproducibility issues.

## Pipeline Overview
6-stage pipeline defined in `dvc.yaml`:
1. **prepare** — Load & normalize wavelet/absorption features
2. **probe** — RBF SVM micro-probing (24-dim separability vectors)
3. **cluster** — K-Means with elbow analysis (K=2-20 sweep)
4. **viz** — UMAP 2D/3D + spatial cluster maps
5. **audit** — Cross-run overlap, variance audit, feature attribution
6. **rf** — Per-cluster Random Forest classifiers with GridSearchCV

## Key Commands
```bash
# Reproduce full pipeline (skips unchanged stages)
dvc repro

# Run specific stage
dvc repro <stage_name>

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag
```

## Adding a New Stage
1. Create core logic in `src/core/<module>.py`
2. Create thin script in `scripts/run_<stage>.py`
3. Add stage to `dvc.yaml` with correct `deps`, `outs`, `metrics`
4. Ensure `deps` paths match actual file locations exactly
5. Update `scripts/run_all.py` to include the new stage number
6. Run `dvc repro` to verify

## Common Pitfalls
- `deps` paths in `dvc.yaml` must be exact — `src/cluster.py` != `src/core/cluster.py`
- After moving files, update both `dvc.yaml` deps AND `dvc.lock` (via `dvc repro`)
- Metrics with `cache: false` are always readable in Git without `dvc pull`
- Large outputs should use default caching; small CSVs/plots use `cache: false`
