---
name: research-workflow
description: Research experiment lifecycle, results interpretation, multi-machine sync
---

# Research Workflow

## When to Use
When planning new experiments, reviewing results, or preparing for publication.

## Experiment Lifecycle
1. **Design** — Document hypothesis in `docs/feature/<branch-name>/`
2. **Implement** — Code in `src/core/`, script in `scripts/`, stage in `dvc.yaml`
3. **Run** — `dvc repro` for full pipeline, or target specific stages
4. **Analyze** — Check results in `results/signal_clustering_v2/`
5. **Iterate** — Modify parameters, re-run changed stages only
6. **Document** — Update docs, commit with conventional message

## Results Interpretation
- `sweep_*.csv` — Elbow plots: look for inertia drop + silhouette peak
- `cluster_stats_*.csv` — Cluster sizes, L2 norms, centroid magnitudes
- `cross_run_contingency_k5.csv` — Wavelet vs raw cluster agreement
- `rf_summary_*.csv` — Per-cluster RF accuracy (high = good separability)
- UMAP plots — Visual cluster separation in 2D/3D

## Multi-Machine Workflow
```bash
# After running experiments (Machine A)
dvc add mlruns && git add mlruns.dvc && git commit -m "mlflow: sync metadata"
dvc push && git push

# Before reviewing (Machine B)
git pull && dvc pull mlruns && mlflow ui
```

## Documentation Standards
- Every feature branch gets a doc folder: `docs/feature/<branch-name>/`
- Include a project plan with Mermaid pipeline diagram
- Update `docs/core_implementation_principles.md` for architectural changes
