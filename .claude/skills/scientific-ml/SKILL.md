---
name: scientific-ml
description: Scientific ML development, micro-probing algorithm, data shapes, reproducibility
---

# Scientific ML Development

## When to Use
When implementing new ML algorithms, modifying the probe/cluster/RF pipeline, or analyzing results.

## The Micro-Probing Algorithm
For each sightline index `s`:
1. Extract 4 class feature vectors at index `s`
2. For each of 6 one-vs-one class pairs (C(4,2)=6):
   - Train RBF SVM on just the 2 class vectors
   - Record signed decision distances from all 4 classes to hyperplane
3. Result: 24-dim separability vector (6 pairs x 4 distances)

This encodes the "separability landscape" — how distinguishable physics classes are at each spectral position.

## Adding a New ML Module
1. Create `src/core/<module>.py`
2. If it's a transform: inherit `BaseTransformer`, implement `fit_transform(X, y)`
3. If it's a model: inherit `BaseModel`, implement `train(...)` and `predict(X)`
4. Add type hints with shape comments
5. Write tests in `tests/test_<module>.py` using synthetic data
6. Wire into pipeline via `scripts/run_<stage>.py`

## Data Shapes (Sherwood z=0.3)
- Raw flux: `(16384, 2048)` per class — 4 classes
- Wavelet features: `(16384, 2048)` per class (db8, 6 levels, 12 detail)
- Separability vectors: `(16384, 24)` — one per sightline
- Cluster labels: `(16384,)` — integers 0..K-1

## Reproducibility Checklist
- [ ] Random seeds explicit in all stochastic operations
- [ ] Data loaded via `SignalClusteringData` (not raw np.load in scripts)
- [ ] Results saved with `cache: false` in DVC for Git readability
- [ ] MLflow run logged with params, metrics, and artifacts
