# Signal Clustering Analysis — Project Plan (v4)
## Cluster Sightlines by Their Pattern of class observation Across 4 Feedback Classes

---

## What We Have

| Asset | Path | Shape | Notes |
|---|---|---|---|
| Raw flux spectra | `data/preprocessed/Sherwood_z0.3_inf/{1,2,3,4}/flux.npy` | 4 × (16384, 2048) | F = e^{−τ}, float32, per-class |
| Wavelet features | `data/processed/wavelet_db8_l6_d12/{1,2,3,4}/data.npy` | 4 × (16384, 2048) | db8 DWT (D1-D6+A6), per-class |
| per-sightline-classifier script | `scripts/clustering/train_micro_classifiers.py` | — | RBF SVM per-sightline-probing, **to be updated to 24-dim** (drop intercepts) |
| Clustering module | `src/core/clustering.py` | — | Basic KMeans implementation |
| Visualization module | `src/core/viz.py` | — | Basic training curves only |
| Data loader | `src/core/data.py` | — | Generic DataIngestor class |

Partial implementation exists. **per-sightline-classifier script needs update from 30-dim to 24-dim.** Pipeline orchestration scripts and full visualization suite need to be written.

---

## Goal

Cluster the 16,384 sightline indices by **how the four physics classes are arranged relative to each other at that index**. Indices where the classes all look identical get grouped together. Indices where one class diverges sharply from the others — carrying a physics fingerprint — emerge as distinct clusters. The output is a map of the spectrum's separability landscape: which positions are informationally inert, which are transitional, and which are rare high-signal "islands" where feedback physics is most legible.

---

## Design Decisions

### D1 — Separability Fingerprint: RBF SVM, 24-dim

At each index $i$, train an RBF SVM for each of the 6 one-vs-one class pairs. Record the **signed decision distance** of all 4 class representatives from each hyperplane. This produces a 24-dimensional vector:

$$\mathbf{v}_i = [\delta_{01}, \delta_{02}, \delta_{03}, \delta_{12}, \delta_{13}, \delta_{23}] \in \mathbb{R}^{24}$$

where $\delta_{pq} \in \mathbb{R}^4$ = signed distances of classes C1–C4 from the $C_p$ vs $C_q$ boundary.

These 24 numbers are a complete, compact description of **where the four classes land** at index $i$ — the discrimination geometry. This is exactly what we want to cluster on.

**Intercepts are dropped (30-dim → 24-dim).** The intercept of each SVM encodes the margin center position, which is a function of feature scaling — not the arrangement of the classes. It adds noise without signal.

**Why not Linear SVM weights (3072-dim)?** L1 weights encode *which input features drove the boundary*, not *where the classes landed*. Two indices with different mechanisms but identical class arrangements should cluster together — they won't if we cluster on weights. At 3072 dimensions, K-Means also degrades: distance concentration makes all pairwise distances converge, and centroids lose geometric meaning. Mechanistic interpretation (which wavelet scale drives each cluster) is recovered post-hoc in Stage 5 via projection — it does not need to be an input to the clustering.

### D2 — Two Parallel Feature Inputs

The per-sightline-SVM accepts whichever feature vector we hand it. We run two independent pipelines:

| Run | Input to per-sightline-SVM | Dimensionality | What it probes |
|---|---|:---:|---|
| **Wavelet** (primary) | D1–D6 + A6 from existing wavelet data | 2048 | Full 7-level db8 DWT, per-class storage |
| **Raw** (cross-validation) | A = 1−F computed from raw flux | 2048 | Full spectral context including fine-line structure and inter-line correlations |

Both runs produce a 24-dim separability fingerprint. The clustering space is therefore identical. The scientific question is: **do both representations discover the same high-sensitivity indices?** High overlap confirms those indices are representation-independent physical realities. Divergence is also informative — it means fine-scale structure that the wavelet decomposition drops is carrying independent physics.

**Why D1 and D2 are excluded from the Wavelet run:** At the per-index scale, D1 (std = 0.000102, 99.5% sparse) and D2 (78.8% sparse) present near-zero noise values to the per-sightline-SVM. With only 4 data points, the SVM will fit a hyperplane to noise geometry rather than physics. The raw run captures this fine-scale information naturally, without the sparsity problem, because the 2048-pixel absorption vector is not sparse at the sightline level.

### D3 — Cluster Count: K=8

With 4 classes and 6 OvO pairs, the separability manifold has at most 6 degrees of freedom. K=8 is the expected optimum: it sits just above the effective manifold dimensionality and is the minimum resolution expected to preserve rare high-sensitivity clusters without fracturing bulk indices into noise. Confirm this empirically by sweeping K ∈ [2, 20] and checking the elbow. Run K=5 as a stability check — if a small cluster disappears at K=5, that is evidence K=8 is the correct minimum.
plementary Views

No single visualization is sufficient. Three views are required, each answering a different question:

| View | Output | Question Answered |
|---|---|---|
| **A — UMAP manifold** | 2D scatter + interactive 3D HTML | How are indices arranged in separability space? Do clusters form tight islands or diffuse clouds? |
| **B — Mean absorption profiles** | 5-panel figure, 4 overlaid curves per panel | What does the actual signal look like in each cluster's regime? Do the 4 class curves separate or collapse? |
| **C — Spatial index map** | 1D colored strip of the 2048-pixel axis | Where in the spectrum does each cluster live? |

---

## Pipeline

```
data/preprocessed/Sherwood_z0.3_inf/{1,2,3,4}/flux.npy  (4 × 16384 × 2048)
data/processed/wavelet_db8_l6_d12/{1,2,3,4}/data.npy  (4 × 16384 × 2048)
          │
          ▼
    compute A = 1 − F
          │
          ├──────────────────────────────────────────────────────┐
          │  WAVELET RUN (primary)                               │  RAW RUN (cross-validation)
          │                                                      │
    [S1-W] Prepare wavelet input                          [S1-R] Prepare raw input
          │  load existing wavelet data                         │  A = 1−F  (no further transform)
          │  load per-class wavelet features                    │  →  (16384 × 2048)
          │                                                      │
    [S2-W] Micro-Probing                                  [S2-R] Micro-Probing
          │  RBF SVM × 6 OvO per index                         │  RBF SVM × 6 OvO per index
          │  24 decision distances per index                    │  24 decision distances per index
          │  →  fingerprints_wavelet.npy  (16384 × 24)         │  →  fingerprints_raw.npy  (16384 × 24)
          │                                                      │
    [S3-W] Clustering                                     [S3-R] Clustering
          │  StandardScaler + KMeans(K=5)                       │  StandardScaler + KMeans(K=5)
          │  →  labels_wavelet.npy  (16384,)                   │  →  labels_raw.npy  (16384,)
          │                                                      │
          └──────────────────────┬───────────────────────────────┘
                                 │  both label arrays + raw spectra + wavelet features
                       ┌─────────▼──────────┐
                       │  [S4] Visualization │
                       │  View A: UMAP 2D+3D │
                       │  View B: Mean A(λ)  │
                       │  View C: Index map  │
                       └─────────┬──────────┘
                                 │
                       ┌─────────▼──────────┐
                       │  [S5] Auditing      │
                       │  cross-representation agreement  │
                       │  Variance audit     │
                       │  Scale attribution  │
                       └────────────────────┘
```

---

## Stage 1 — Prepare Inputs

**What to write:** `src/data.py` (existing `src/core/data.py` needs extension)

This module owns all data loading and transformation. Nothing downstream touches file paths directly.

### 1a — Wavelet Input (Primary Run)

Load the existing wavelet feature arrays from per-class directories. Use full D1-D6+A6 (no slicing). Apply **per-level z-score normalization** before downstream use.

**Per-level normalization:** Each wavelet level (D1, D2, D3, D4, D5, D6, A6) is normalized independently:
- Compute mean and std for each level across all sightlines
- Standardize: `X_level_normalized = (X_level - mean) / std`

```
Input:   data/processed/wavelet_db8_l6_d12/{1,2,3,4}/data.npy   4 × (16384, 2048)
Output:  data/feature_discovery/wavelet_class{1,2,3,4}.npy        4 × (16384, 2048)  float64  (per-level normalized)
```

### 1b — Raw Absorption Input (Cross-validation Run)

Load raw flux from per-class directories and compute the absorption field in memory.

```
Input:   data/preprocessed/Sherwood_z0.3_inf/{1,2,3,4}/flux.npy  4 × (16384, 2048)
Compute: A = 1 − F,  clip to [0, 1]
Output:  X_raw  (16384, 2048)  float32  (concatenated, held in memory)
```

### 1c — Sanity Checks

Before proceeding, verify:

- `X_wavelet` has no NaNs or Infs; values within expected range
- `X_raw` is in [0, 1]; no negative values; grand mean ≈ 0.18–0.22
- Class label distribution: exactly 16384 sightlines per class (4 × 16384 = 65536 total)

**Produce:**

| # | Output | Description |
|:---:|---|---|
| T1 | `stage1_input_summary.csv` | Shape, dtype, min, max, mean, std for X_wavelet and X_raw |
| P1 | `fig_absorption_vs_flux.png` | A(λ) vs F(λ) for 4 representative sightlines (one per class) |
| P2 | `fig_wavelet_scalogram.png` | D1-D6 + A6 coefficient amplitude for one sightline per class |

---

## Stage 2 — per-sightline-Probing

**What to write:** `src/probe.py` · `scripts/run_probe.py`

For each of the 16,384 indices $i$, and for each of the two feature inputs independently:

1. Extract the 4 class class obserevations at index $i$: one feature vector per class → shape (4, n_features)
2. For each of the 6 OvO pairs $(p, q)$: fit `SVC(kernel='rbf', C=1.0, gamma='scale')` on the 2 class vectors
3. Call `decision_function` on all 4 class vectors → 4 signed distances
4. Concatenate the 6 × 4 = 24 distances into $\mathbf{v}_i \in \mathbb{R}^{24}$
5. Do **not** record intercepts

Parallelize the outer loop over indices with `joblib.Parallel(n_jobs=-1)`.

**SVM note on the Raw run:** `gamma='scale'` sets $\gamma = 1 / (n\_features \times \text{Var}(X))$, which automatically adjusts to 2048-dim input. No hyperparameter change is needed between runs.

**What to write in `src/probe.py`:**
- `probe_index(i, X, class_labels) → np.ndarray shape (24,)`
- `run_probe(X, class_labels, n_jobs=-1) → np.ndarray shape (16384, 24)`

**What to write in `scripts/run_probe.py`:**
- CLI: `python run_probe.py --run wavelet|raw|both`
- Loads the appropriate input, calls `run_probe`, saves output

**Outputs:**

| File | Shape | dtype | Description |
|---|---|---|---|
| `data/feature_discovery/fingerprints_wavelet.npy` | (16384, 24) | float64 | Separability vectors from wavelet input |
| `data/feature_discovery/separability_vectors_wavelet.npy` | (16384, 24) | float64 | Separability vectors from wavelet input |
| `data/feature_discovery/separability_vectors_raw.npy` | (16384, 24) | float64 | Separability vectors from raw A input |

**Produce:**

| # | Output | Description |
|:---:|---|---|
| T2 | `stage2_separability_vector_summary.csv` | Per-dimension mean, std, min, max for both separability vector arrays |
| P3 | `fig_separability_vector_norms.png` | Side-by-side histograms of $\|\mathbf{v}_i\|_2$ for both runs on the same axes — high-norm indices are Signal Island candidates; early cross-representation agreement check |

---

## Stage 3 — Clustering

**What to write:** `src/cluster.py` · `scripts/run_cluster.py`

Run independently for each fingerprint array. The procedure is identical for both.

**Steps:**

1. `StandardScaler().fit_transform(fingerprints)` — zero mean, unit variance per dimension
2. Sweep K ∈ [2, 20]: for each K, fit `KMeans(n_clusters=K, random_state=42, n_init=20)`, record inertia and silhouette score
3. Identify elbow and silhouette peak. Confirm K=5 is the answer. If a different K is indicated, investigate before proceeding.
4. Fit final `KMeans(n_clusters=5, random_state=42, n_init=20)` on standardized fingerprints
5. Save cluster labels and centroids
6. Repeat steps 1–5 with K=5 (stability check)

**What to write in `src/cluster.py`:**
- `k_sweep(fingerprints, k_range) → DataFrame` with columns [k, inertia, silhouette]
- `fit_kmeans(fingerprints, k, seed=42) → (labels, centroids)`

**What to write in `scripts/run_cluster.py`:**
- CLI: `python run_cluster.py --run wavelet|raw|both --k 5`
- Runs sweep, plots elbow, fits final model, saves outputs

**Outputs:**

| File | Shape | dtype | Description |
|---|---|---|---|
| `data/feature_discovery/labels_wavelet_k5.npy` | (16384,) | int32 | K=5 cluster assignments, wavelet run |
| `data/feature_discovery/labels_raw_k5.npy` | (16384,) | int32 | K=5 cluster assignments, raw run |
| `data/feature_discovery/centroids_wavelet_k5.npy` | (5, 24) | float64 | K=5 centroids, wavelet run |
| `data/feature_discovery/centroids_raw_k5.npy` | (5, 24) | float64 | K=5 centroids, raw run |
| `data/feature_discovery/labels_wavelet_k8.npy` | (16384,) | int32 | K=8 stability check, wavelet |
| `data/feature_discovery/labels_raw_k8.npy` | (16384,) | int32 | K=8 stability check, raw |

**Produce:**

| # | Output | Description |
|:---:|---|---|
| P4 | `fig_elbow_wavelet.png` | Inertia + silhouette vs. K (2–20), wavelet run. Vertical line at chosen K. |
| P5 | `fig_elbow_raw.png` | Same for raw run |
| T3 | `cluster_stats_wavelet_k5.csv` | Per-cluster: size, % of total, mean L2 to centroid, max L2, centroid magnitude, top-3 most active dimensions |
| T4 | `cluster_stats_raw_k5.csv` | Same for raw run |
| T5 | `contingency_k5_vs_k5_wavelet.csv` | 5×8 membership table — confirms which K=5 clusters dissolve at K=5 |
| T6 | `contingency_k5_vs_k5_raw.csv` | Same for raw run |

---

## Stage 4 — Visualization Suite

**What to write:** `src/viz.py` · `scripts/run_viz.py`

All three views are produced for both runs independently, then laid out side-by-side for comparison. `run_viz.py` accepts `--run wavelet|raw|both` and `--view A|B|C|all`.

---

### View A — UMAP Manifold

**What to write in `src/viz.py`:** `plot_umap_2d(fingerprints, labels, title)` · `plot_umap_3d_html(fingerprints, labels, title)`

Fit UMAP on the standardized 24-dim fingerprints. Color each point by its K=8 cluster label.

**Parameters:** `n_neighbors=15, min_dist=0.1, n_components=2` (2D) and `n_components=3` (3D). Use `random_state=42` for reproducibility. Run once per fingerprint array.

**What to look for:** The bulk clusters (expected to be large) should form a dense core. Signal Island clusters (expected to be small, ~2–8% of indices) should appear as detached peninsulas or separate clouds. The spatial separation in UMAP should correlate with centroid distance in K-Means space — if it does not, the UMAP layout is misleading and should not be over-interpreted.

**Produce:**

| # | Output | Description |
|:---:|---|---|
| P6 | `fig_umap2d_wavelet_k5.png` | 2D UMAP, wavelet run, colored by K=5 label |
| P7 | `fig_umap2d_raw_k5.png` | 2D UMAP, raw run, colored by K=5 label |
| P8 | `fig_umap2d_comparison.png` | Side-by-side: wavelet K=5 \| raw K=5 \| wavelet K=5 \| raw K=5 |
| P9 | `umap3d_wavelet_k5.html` | Interactive 3D Plotly, wavelet run |
| P10 | `umap3d_raw_k5.html` | Interactive 3D Plotly, raw run |

---

### View B — Mean Absorption Profiles per Cluster

**What to write in `src/viz.py`:** `plot_cluster_profiles(X_raw, labels, class_labels, run_name)`

For each of the K=5 clusters, compute the mean absorption spectrum $\langle A(\lambda) \rangle$ for each of the 4 classes, using only the sightlines belonging to that cluster. Plot all 4 class means overlaid on one panel, with ±1σ shading.

Layout: 5-panel figure (2 rows × 3 columns, one panel per cluster). Use consistent class colors across all panels. Title each panel with cluster ID and size.

**What to look for:**
- **low-sensitivity positions clusters:** 4 class curves nearly coincide — real signal, no class-discriminative structure at these positions
- **Transition clusters:** mild but visible separation in specific wavelength windows
- **high-sensitivity positions:** strong, clearly spread class curves — the spectral positions where feedback model leaves a measurable imprint

This view uses the raw absorption field A for both runs (it is the physically interpretable signal regardless of which features fed the per-sightline-SVM).

**Produce:**

| # | Output | Description |
|:---:|---|---|
| P11 | `fig_profiles_wavelet_k5.png` | 5-panel mean A(λ) per cluster × class, wavelet cluster labels |
| P12 | `fig_profiles_raw_k5.png` | Same using raw run cluster labels |

---

### View C — Spatial Index Map

**What to write in `src/viz.py`:** `plot_spatial_map(labels, X_raw, run_name)`

A two-panel figure:

- **Top panel:** A 1D horizontal strip of width 2048 pixels. Each pixel position is colored by its K=5 cluster assignment. Height is decorative (~20px). Use the same cluster color palette as the UMAP plots.
- **Bottom panel:** The grand-mean absorption profile $\langle A(\lambda) \rangle$ across all 16,384 sightlines, plotted as a line. Overlay transparent vertical color bands corresponding to the spatial extent of each cluster (especially high-sensitivity positions), so the reader can immediately see which spectral features coincide with which cluster.

**What to look for:**
- Signal Island indices concentrated at specific pixel ranges → physically interpretable positions (DLA systems, Lyman-α forest edges, strong metal lines)
- low-sensitivity positions indices distributed across the whole axis → background IGM
- Whether the Wavelet and Raw runs agree on the spectral location of their high-sensitivity positions

**Produce:**

| # | Output | Description |
|:---:|---|---|
| P13 | `fig_spatial_map_wavelet_k5.png` | Spatial index map, wavelet cluster labels |
| P14 | `fig_spatial_map_raw_k5.png` | Spatial index map, raw cluster labels |
| P15 | `fig_spatial_map_comparison.png` | Both maps stacked vertically on the same pixel axis for direct comparison |

---

## Stage 5 — Auditing

**What to write:** `scripts/run_audit.py` · supporting functions in `src/audit.py`

### 5.1 — Cross-Representation Cluster Agreement

Compare the K=5 labels from the Wavelet run against the K=5 labels from the Raw run for all 16,384 indices.

**Steps:**
1. Build the 5×5 contingency matrix: `contingency[i, j]` = number of indices assigned to Wavelet cluster $i$ and Raw cluster $j$
2. For each Wavelet cluster, identify the best-matching Raw cluster (highest overlap count) and compute the overlap percentage
3. Produce the contingency heatmap and the Procrustes-aligned 3D UMAP overlay

**Visualizations:**

**P16a — Contingency Heatmap:** A 5×5 annotated heatmap of the contingency matrix. Each cell displays both the absolute count and the row-normalized percentage. Uses `YlOrRd` colormap. Rows = Wavelet clusters, Columns = Raw clusters.

**P16b — Shared 3D UMAP Overlay:** Both the wavelet and raw fingerprint arrays (each 16384 × 24) are concatenated into a single (32768 × 24) matrix, standardized together, and projected through a single 3D UMAP fit. The resulting embedding is split back into wavelet and raw halves, which now live in the exact same 3D coordinate system. The result is an interactive Plotly visualization:
- **Wavelet points:** 16,384 points colored by wavelet cluster, opacity=0.5
- **Raw points:** 16,384 points colored by matched wavelet cluster color, opacity=0.5
- Matched cluster pairs share the same color. Sightlines with similar fingerprint patterns across both representations will naturally occupy the same region of 3D UMAP space, producing denser opacity where the runs agree.

**Produce:**

| # | Output | Description |
|:---:|---|---|
| T5 | `cross_run_contingency_k5.csv` | 5×5 contingency table, Wavelet rows vs Raw columns |
| T6 | `cross_run_overlap_summary.csv` | Per-cluster: size (wavelet), best raw match, overlap %, regime |
| P16a | `fig_contingency_heatmap_k5.png` | 5×5 annotated heatmap of contingency matrix |
| P16b | `fig_shared_umap_overlay_k5.html` | Shared 3D UMAP overlay (interactive) |

### 5.2 — Separability Vector Greyscale Visualization

For each run (wavelet and raw), create a greyscale image of the separability vectors (fingerprints) grouped by cluster assignment.

**Method:**
1. Sort all 16,384 sightlines by their cluster label (0 through 4)
2. Stack the 24-dimensional fingerprint vectors as rows, producing a (16384 × 24) image matrix
3. Normalize values to [0, 1] for greyscale rendering
4. Display as a greyscale image: rows = sightlines (grouped by cluster), columns = fingerprint dimensions (0–23)
5. Draw red horizontal lines at cluster boundaries for visual separation

**What to look for:**
- Whether specific fingerprint dimensions show consistent high/low activation within a cluster (vertical bands within a cluster block)
- Whether different clusters activate different subsets of dimensions (distinct column patterns across cluster blocks)
- Whether wavelet and raw runs produce similar or different activation patterns

**Produce:**

| # | Output | Description |
|:---:|---|---|
| P17 | `fig_separability_greyscale_wavelet_k5.png` | Greyscale fingerprint image, wavelet run |
| P18 | `fig_separability_greyscale_raw_k5.png` | Greyscale fingerprint image, raw run |

---

## Complete Script and Module List

**Status:** Partial implementation exists. Items marked with ← update need modification from existing code.

| File | Purpose | Status |
|---|---|---|
| `src/data.py` | Load wavelet features (full D1-D6+A6), load raw flux, compute A=1−F, validate shapes and value ranges | ← update (extend existing) |
| `src/probe.py` | `probe_index()` · `run_probe()` — RBF SVM per-sightline-probing, 24-dim output | ← write |
| `src/cluster.py` | `k_sweep()` · `fit_kmeans()` — standardize, sweep K, fit final model | ← write |
| `src/viz.py` | `plot_umap_2d()` · `plot_umap_3d_html()` · `plot_cluster_profiles()` · `plot_spatial_map()` | ← write |
| `src/audit.py` | `cross_run_overlap()` · `compute_contingency()` · `plot_contingency_heatmap()` · `plot_shared_umap_overlay_3d()` · `plot_separability_greyscale()` | ← write |
| `scripts/run_probe.py` | CLI for Stage 2 — `--run wavelet\|raw\|both` | ← write |
| `scripts/run_cluster.py` | CLI for Stage 3 — `--run wavelet\|raw\|both --k 5` | ← write |
| `scripts/run_viz.py` | CLI for Stage 4 — `--run wavelet\|raw\|both --view A\|B\|C\|all` | ← write |
| `scripts/run_audit.py` | CLI for Stage 5 — cross-run agreement audit | ← write |
| `scripts/run_all.py` | End-to-end orchestrator — calls all four scripts in order | ← write |
| `scripts/clustering/train_micro_classifiers.py` | RBF SVM per-sightline-probing | ← update (30-dim → 24-dim) |

---

## Complete Output Catalogue

### Data Files

| File | Shape | dtype | Produced by |
|---|---|---|---|
| `data/feature_discovery/fingerprints_wavelet.npy` | (16384, 24) | float64 | `run_probe.py --run wavelet` |
| `data/feature_discovery/fingerprints_raw.npy` | (16384, 24) | float64 | `run_probe.py --run raw` |
| `data/feature_discovery/labels_wavelet_k5.npy` | (16384,) | int32 | `run_cluster.py --run wavelet --k 5` |
| `data/feature_discovery/labels_raw_k5.npy` | (16384,) | int32 | `run_cluster.py --run raw --k 5` |
| `data/feature_discovery/centroids_wavelet_k5.npy` | (5, 24) | float64 | `run_cluster.py --run wavelet --k 5` |
| `data/feature_discovery/centroids_raw_k5.npy` | (5, 24) | float64 | `run_cluster.py --run raw --k 5` |
| `data/feature_discovery/labels_wavelet_k8.npy` | (16384,) | int32 | `run_cluster.py --run wavelet --k 8` |
| `data/feature_discovery/labels_raw_k8.npy` | (16384,) | int32 | `run_cluster.py --run raw --k 8` |

### Tables

| ID | File | Produced by |
|:---:|---|---|
| T1 | `results/signal_clustering_v2/stage1_input_summary.csv` | `run_probe.py` (pre-run validation) |
| T2 | `results/signal_clustering_v2/stage2_fingerprint_summary.csv` | `run_probe.py` |
| T3 | `results/signal_clustering_v2/cluster_stats_wavelet_k5.csv` | `run_cluster.py` |
| T4 | `results/signal_clustering_v2/cluster_stats_raw_k5.csv` | `run_cluster.py` |
| T5 | `results/signal_clustering_v2/cross_run_contingency_k5.csv` | `run_audit.py` |
| T6 | `results/signal_clustering_v2/cross_run_overlap_summary.csv` | `run_audit.py` |

### Figures

| ID | File | View | Produced by |
|:---:|---|---|---|
| P1 | `figs/signal_clustering_v2/fig_absorption_vs_flux.png` | — | `run_probe.py` (validation) |
| P2 | `figs/signal_clustering_v2/fig_wavelet_scalogram.png` | — | `run_probe.py` (validation) |
| P3 | `figs/signal_clustering_v2/fig_fingerprint_norms.png` | — | `run_probe.py` |
| P4 | `figs/signal_clustering_v2/fig_elbow_wavelet.png` | — | `run_cluster.py` |
| P5 | `figs/signal_clustering_v2/fig_elbow_raw.png` | — | `run_cluster.py` |
| P6 | `figs/signal_clustering_v2/fig_umap2d_wavelet_k5.png` | A | `run_viz.py` |
| P7 | `figs/signal_clustering_v2/fig_umap2d_raw_k5.png` | A | `run_viz.py` |
| P8 | `figs/signal_clustering_v2/fig_umap2d_comparison.png` | A | `run_viz.py` |
| P9 | `figs/signal_clustering_v2/umap3d_wavelet_k5.html` | A | `run_viz.py` |
| P10 | `figs/signal_clustering_v2/umap3d_raw_k5.html` | A | `run_viz.py` |
| P11 | `figs/signal_clustering_v2/fig_profiles_wavelet_k5.png` | B | `run_viz.py` |
| P12 | `figs/signal_clustering_v2/fig_profiles_raw_k5.png` | B | `run_viz.py` |
| P13 | `figs/signal_clustering_v2/fig_spatial_map_wavelet_k5.png` | C | `run_viz.py` |
| P14 | `figs/signal_clustering_v2/fig_spatial_map_raw_k5.png` | C | `run_viz.py` |
| P15 | `figs/signal_clustering_v2/fig_spatial_map_comparison.png` | C | `run_viz.py` |
| P16a | `figs/signal_clustering_v2/fig_contingency_heatmap_k5.png` | — | `run_audit.py` |
| P16b | `figs/signal_clustering_v2/fig_shared_umap_overlay_k5.html` | — | `run_audit.py` |
| P17 | `figs/signal_clustering_v2/fig_separability_greyscale_wavelet_k5.png` | — | `run_audit.py` |
| P18 | `figs/signal_clustering_v2/fig_separability_greyscale_raw_k5.png` | — | `run_audit.py` |

---

## Directory Structure

```
project/
│
├── .dvc/                                         # DVC cache and config
├── *.dvc                                         # DVC tracking files at root (preprocessed.dvc, processed.dvc, etc.)
├── dvc_signal_clustering.yaml                    # DVC pipeline definition
│
├── data/
│   ├── preprocessed/
│   │   └── Sherwood_z0.3_inf/
│   │       ├── 1/flux.npy  (16384, 2048)  ← exists
│   │       ├── 2/flux.npy
│   │       ├── 3/flux.npy
│   │       └── 4/flux.npy
│   ├── processed/
│   │   └── wavelet_db8_l6_d12/
│   │       ├── 1/data.npy  (16384, 2048)  ← exists
│   │       ├── 2/data.npy
│   │       ├── 3/data.npy
│   │       └── 4/data.npy
│   └── feature_discovery/
│       ├── fingerprints_wavelet.npy             (16384 × 24)    ← produced by S2
│       ├── fingerprints_raw.npy                 (16384 × 24)    ← produced by S2
│       ├── labels_wavelet_k5.npy                (16384,)        ← produced by S3
│       ├── labels_raw_k5.npy                    (16384,)        ← produced by S3
│       ├── centroids_wavelet_k5.npy             (5 × 24)        ← produced by S3
│       ├── centroids_raw_k5.npy                 (5 × 24)        ← produced by S3
│       ├── labels_wavelet_k5.npy                (16384,)        ← produced by S3
│       └── labels_raw_k5.npy                    (16384,)        ← produced by S3
│
├── src/
│   ├── data.py                                  ← write: load, transform, validate
│   ├── probe.py                                 ← write: RBF SVM per-sightline-probing
│   ├── cluster.py                               ← write: k-sweep, KMeans
│   ├── viz.py                                   ← write: UMAP, profiles, spatial map
│   └── audit.py                                 ← write: overlap, attribution, variance
│
├── scripts/
│   ├── run_probe.py                             ← write: Stage 2 CLI
│   ├── run_cluster.py                           ← write: Stage 3 CLI
│   ├── run_viz.py                               ← write: Stage 4 CLI
│   ├── run_audit.py                             ← write: Stage 5 CLI
│   └── run_all.py                               ← write: end-to-end orchestrator
│
├── results/
│   └── signal_clustering_v2/                   ← branch-specific results
│       ├── stage1_input_summary.csv             ← T1
│       ├── stage2_fingerprint_summary.csv       ← T2
│       ├── cluster_stats_wavelet_k5.csv         ← T3
│       ├── cluster_stats_raw_k5.csv             ← T4
│       ├── contingency_k5_vs_k5_wavelet.csv     ← T5
│       ├── contingency_k5_vs_k5_raw.csv         ← T6
│       ├── cross_run_contingency_k5.csv         ← T7
│       ├── cross_run_overlap_summary.csv        ← T8
│       └── variance_audit.csv                  ← T9
│
└── figs/
    └── signal_clustering_v2/                    ← branch-specific figures
        ├── fig_absorption_vs_flux.png           ← P1
        ├── fig_wavelet_scalogram.png            ← P2
        ├── fig_fingerprint_norms.png            ← P3
        ├── fig_elbow_wavelet.png                ← P4
        ├── fig_elbow_raw.png                    ← P5
        ├── fig_umap2d_wavelet_k5.png            ← P6
    ├── fig_umap2d_raw_k5.png                    ← P7
    ├── fig_umap2d_comparison.png                ← P8
    ├── umap3d_wavelet_k5.html                   ← P9
    ├── umap3d_raw_k5.html                       ← P10
    ├── fig_profiles_wavelet_k5.png              ← P11
    ├── fig_profiles_raw_k5.png                  ← P12
    ├── fig_spatial_map_wavelet_k5.png           ← P13
    ├── fig_spatial_map_raw_k5.png               ← P14
    ├── fig_spatial_map_comparison.png           ← P15
    ├── fig_cross_run_overlap.png                ← P16
    ├── fig_wavelet_attribution_heatmap.png      ← P17
    ├── fig_raw_attribution_heatmap.png          ← P18
    └── fig_variance_ratios.png                  ← P19
```

---

## Key Findings from Audit (Stage 5)

### 5.1 cross-representation Cluster Agreement

| Wavelet Cluster | Size | % | Best Raw Match | Overlap % | Regime |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 3302 | 20.2% | 0 | 37.2% | low-sensitivity positions |
| 1 | 3190 | 19.5% | 3 | 37.2% | low-sensitivity positions |
| 2 | 2385 | 14.6% | 1 | 60.5% | transition |
| 3 | 1606 | 9.8% | 2 | 17.9% | transition |
| 4 | 562 | 3.4% | 4 | 31.1% | **signal_island** |
| 5 | 455 | 2.8% | 7 | 12.1% | **signal_island** |
| 6 | 2577 | 15.7% | 6 | 4.1% | low-sensitivity positions |
| 7 | 2307 | 14.1% | 5 | 14.7% | transition |

**Key Observations:**
- low-sensitivity positions clusters (0, 1, 6) show moderate overlap (~37%) with raw clusters
- Signal Island clusters (4, 5) show low overlap (12-31%), indicating wavelet and raw representations capture different aspects of high-sensitivity positions
- Cluster 2 (transition) shows highest overlap (60.5%), suggesting similar separability patterns are discovered by both representations

### 5.2 Post-Hoc Feature Attribution

**Wavelet Attribution:**
- Signal Island clusters (4, 5) show elevated activation in D5, D6, and A6 wavelet levels
- low-sensitivity positions clusters dominated by lower-frequency components (D3, D4)

**Raw Spectral Attribution:**
- Signal Island clusters concentrated in specific pixel bins corresponding to high-variance spectral regions
- low-sensitivity positions clusters show more uniform deviation patterns across the spectrum

### 5.3 Inter-Class Variance Audit

| Run | Cluster | Size | Variance | Normalized | Regime |
|:---:|:---:|:---:|:---:|:---:|:---:|
| wavelet | 4 | 562 | 0.000123 | **10.2x** | signal_island |
| wavelet | 5 | 455 | 0.000128 | **10.6x** | signal_island |
| raw | 4 | 460 | 0.000709 | **58.6x** | signal_island |
| raw | 6 | 758 | 0.000086 | **7.1x** | signal_island |
| raw | 7 | 431 | 0.000140 | **11.6x** | signal_island |

**Key Observations:**
- high-sensitivity positions show 7-58x higher inter-class variance than low-sensitivity positions clusters
- Raw representation is more discriminative: variance amplification up to 58x vs 10x for wavelet
- This confirms the Signal Island designation: these are positions where the 4 physics classes are most separable

---

## Execution Summary

**Pipeline Completed:** 2026-03-09

| Stage | Status | Notes |
|:---:|:---:|:---|
| 1 | ✓ | Input preparation (wavelet + raw absorption) |
| 2 | ✓ | per-sightline-probing (24-dim fingerprints, 16384 sightlines) |
| 3 | ✓ | K-Means clustering (K=5 primary, K=5 stability check) |
| 4 | ✓ | Visualization (UMAP, profiles, spatial maps) |
| 5 | ✓ | Auditing (overlap, attribution, variance) |

**Bug Fixes Applied:**
- Fixed probe.py: was iterating over wrong dimension (2048 → 16384 sightlines)
- Fixed viz.py: corrected class label indexing for cluster profile plotting
- Fixed audit.py: corrected raw attribution to work with per-class data
