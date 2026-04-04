# Signal Clustering v1 — Exploratory Discovery Phase

**Branch:** `feature/signal-clustering-analysis` (tag: `v0.3-clustering-v1`)

## What This Experiment Did

Trained per-sightline RBF SVMs across all 16,384 indices and extracted **30-dimensional separability vectors** (24 decision distances + 6 intercepts). Clustered these vectors with K-Means (K=8) to discover physically meaningful spectral regimes.

## Key Findings

1. **Signal Islands exist.** K=8 clustering revealed 3 rare clusters (~1,200 indices total) with fundamentally different "classification personality" — these indices discriminate physics models in ways the bulk spectrum does not.

2. **Transform-invariant.** Wavelet and DCT pipelines produced ~90% overlapping clusters, confirming the islands are physical, not mathematical artifacts.

3. **K=5 dissolves rare clusters.** Reducing to K=5 merges the fine-grained signal islands into bulk noise, providing evidence K=8 is the minimum resolution for this manifold.

4. **Intercepts add noise.** The 6 SVM intercepts encode margin center position (scaling artifact), not class arrangement geometry.

## What Led to v2

These findings motivated three design changes for v2:

- **30-dim → 24-dim:** Drop intercepts, keeping only decision distances (finding #4)
- **Manual scripts → DVC pipeline:** v1 used 8 standalone scripts run manually; v2 formalized this as a 6-stage DVC pipeline
- **K=8 → K=5 + sweep:** v2 runs K-sweep [2,20] empirically rather than assuming K=8, and uses K=5 as the default after elbow analysis showed it was sufficient for the cleaned 24-dim vectors
- **Dual-run validation:** v2 runs both wavelet and raw absorption inputs through the same 24-dim pipeline, with cross-run audit (finding #2)

## Files

| File | Content |
|------|---------|
| `signal_clustering_analysis.md` | Full technical writeup (theory, implementation, results, file paths) |
| `feature_extraction_plan.md` | Design doc for wavelet and DCT feature extraction (A=1-F rationale, DWT config) |
| `support/behavior_vector_strategy.md` | Design decision: sensitivity (30-dim) vs mechanism (3072-dim) vectors |
| `support/clustering_comparison_report.md` | K=5 vs K=8 comparative analysis with contingency tables and justification |

## Caveats

- File paths in these docs reference the v1 script locations (`scripts/clustering/`, `src/core/clustering.py`) which no longer exist in the current codebase
- The 30-dim vector format and K=8 default were superseded by v2's 24-dim / K=5 design
- The "Mechanism-based" L1-Linear SVM approach described in `behavior_vector_strategy.md` was selected but never implemented as an experiment; v2 continued with sensitivity-based (RBF) vectors
