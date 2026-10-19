# LEDGER — eigen-voigt-embedding

**Branch:** `exp/eigen-voigt-embedding` (off `exp/signed-response-embedding`, which carries
`FluxPowerSpectrumTransform`). **Opened:** 2026-10-18 by user request (un-park).
**Question:** do graph-Laplacian eigenvector embeddings (kNN graph → normalised Laplacian →
eigenvectors; K-Means on them = spectral clustering), computed on (i) the raw spectra and
(ii) Voigt-profile line features extracted with a Python library (VoigtFit), carry per-sightline
feedback-class information beyond PCA at matched dimension, P_F(k) and the mean-flux scalars?
**Binding user directive:** smoke run, then full run. No smoke-scale scientific gate may stop
the full run; the smoke checks engineering correctness only.

---

## 1. The Pulse (Progress & Roadmap)

| Stage | Description | Status |
|---|---|---|
| EV-0 — orientation + data audit | PI design review; data-engineer locality/noise/pairing audit | ✅ **DONE** |
| EV-1 — Voigt catalogs | VoigtFit decomposition of all 4 × 16384 sightlines, fiducial + τ-rescaled | 🚀 **NEXT** — smoke ✅, full running |
| EV-2 — embeddings + evaluation | grouped CV: PCA vs Laplacian eigenmaps on raw / Voigt / P_F(k); confound controls; clusters | 🚀 **NEXT** — smoke ✅, full queued behind EV-1 |
| EV-3 — interpretation + panel | read full results against §5 ceilings; defense-panel on the result | ⏳ **PENDING** |

### ✅ Completed Milestones
- **2026-10-18**: track opened; PI APPROVE-WITH-CAVEATS (provisional); data audit complete.
- **2026-10-18**: VoigtFit pipeline validated (exact synthetic recovery incl. saturated/LLS/DLA-strength lines); 4 VoigtFit defects found and patched.
- **2026-10-18**: smoke run (256 skewers × 4 classes) of EV-1 (both variants) and EV-2 (2 seeds) end to end; full run launched.

---

## 2. Methodology & Architecture

- **Data.** Sherwood z=0.3 `_inf` flux, 4 classes × 16384 sightlines × 2048 px, Δv = 2.6365 km/s,
  periodic (box 5399.56 km/s). Verified this session: `flux == roll(exp(-tau), 1)` to 1e-15 —
  **noiseless, no LSF**; sightline index i is the same skewer in all classes (axis.npy identical;
  same-index τ correlation 0.99 / 0.97 / 0.71 for c2 / c3 / c4 vs c1).
- **Data variants.** `fiducial` = stored flux. `taurescaled` = roll(exp(−A_c τ_c), 1) with A_c set so
  every class has the pooled mean flux 0.977875: A = [1.0297, 0.8196, 0.8579, 1.4914]. This absorbs
  any UVB / photoionisation-rate-like mean-flux difference; the whole pipeline is re-run on it.
- **Voigt features (`src/core/voigt.py`).** VoigtFit 3.23.2 (lmfit least squares) driven by an
  automatic search. The sightline is rolled to put its most transmissive pixel at the edge, then split
  into independent absorption regions (1 − F > 0.005, ±5 px, merged). In each region, components are
  seeded at τ_eff peaks, with a curve-of-growth grid seed for saturated segments; components are added
  at residual peaks while max|ΔF| falls, until max|ΔF| < tol = 0.01. Per-sightline features (31):
  line counts at log N ≥ 12/13/14, total log N, b and log N quantiles, N-weighted b, broad/narrow
  fractions, b–N slope, median line spacing, 12-cell (b, log N) occupancy, has_lines.
- **Embeddings (`src/core/eigen_embedding.py`).** kNN (k = 15) graph with self-tuning Gaussian
  weights. Random-walk eigenvectors from D^-1/2 W D^-1/2. Graph fit on the training fold only, test
  points embedded by Nyström extension. Masked variant drops same-skewer edges. Spectral clustering =
  K-Means on the first K eigenvectors. Dsilva+2018 harmonic test.
- **Evaluation (`src/core/eigen_eval.py`, orchestration in `src/core/eigen_voigt.py`).** See §5.
- **Entry points.** `scripts/run_eigen_voigt_catalog.py` (EV-1), `scripts/run_eigen_voigt_eval.py`
  (EV-2), `scripts/run_eigen_voigt_full.sh` (detached, resumable full driver).

---

## 3. The Logic (Decision Log)

- **[D-01] Track opened; smoke checks engineering only**: the user asked for eigenvector embeddings
  (spectral-clustering style) on raw spectra and on Voigt-library features, with an explicit
  instruction that no small-run gate may be used to avoid running the whole experiment. Smoke = code
  runs, outputs finite, fitter recovers known lines, runtime extrapolation; a smoke failure means
  fix-and-rerun, never close. Interpretation thresholds (§5) apply to the full run only. PI sign-off
  APPROVE-WITH-CAVEATS, PI-only, PROVISIONAL (rule 15); defense-panel review of the protocol runs in
  parallel with the full run and may change metrics but not cancel the run.
- **[D-02] linewidth-cddf Q-C1 "signal-limited" verdict superseded**: Q-C1 (2026-06-02) assumed
  σ_F = 0.01 noise; the data is noiseless (verified above). On the noiseless Sherwood grid the fitter
  recovers every cell Q-C1 reported failing (e.g. b = 15, log N = 15: Q-C1 +70% b-bias → here
  b = 15.000, log N = 15.000; b = 40, log N = 18.5 → 39.98 / 18.502). Q-C3's UVB-degeneracy concern does
  not transfer as a kill (no CDDF-vs-literature claim here; all classes share one UVB), but transfers
  as a confound — hence the mandatory τ-rescaled variant. The 2× line-density shortfall vs
  Davé+1999 is noise-independent and remains a caveat for any comparison with observed line counts.
- **[D-03] VoigtFit as the fitter, with four documented patches**: VoigtFit is the library fitter of
  record; four defects found this session are patched in `_install_vf_patches` (no fork):
  (1) its Tepper-García H(a, x) is 0/0 at x = 0 and wrong by up to 1e12 for |x| < 1e-4 (1–3% for
  1 < |x| < 5) → exact Faddeeva profile for |x| < 10, Tepper-García beyond (rel. err < 4e-4);
  (2) a float resolution makes it evaluate on a padded 3× log grid and interpolate (0.13 km/s shift,
  3× cost) → point-sampled exact grid when res = 0, which is the correct forward model for Sherwood;
  (3) `nan_policy='omit'` → `'raise'`; (4) Lyα constants aligned to VoigtFit's own line list
  (λ0 = 1215.6696 Å float32, f = 0.416), so v and log N carry that convention (vs NIST: 0.12 km/s,
  4e-4 dex). Whole-sightline joint fits (≤ 82 components, 246 parameters) segfaulted scipy's C MINPACK
  deterministically (class-2 sightline 7307); independent-region fitting removes this and is 10×
  faster. A residual crash is survived by the catalog writer (pool rebuilt, sightline re-queued,
  per-sightline isolation, a final crash recorded as `n_fit_errors = −1`). A `least_squares`
  fallback was tested and rejected (~30× slower).
- **[D-04] Pre-registered full-run evaluation** (from the PI review; interpretation criteria, not
  run gates):
  - **Leakage.** Grouped by skewer, 5 folds × 10 seeds; graph, PCA, scalers and residualisers fit on
    the training fold only. Masked and unmasked graphs, with the same-skewer neighbour share reported.
    All comparators recomputed in this harness (the old 0.451 baseline used ungrouped splits).
  - **Representations.** Raw flux: PCA(d) vs eigenmap(d), d ∈ {2, 4, 8, 16, 32}. Voigt features and
    P_F(k): direct, PCA(d) and eigenmap(d). Scalars: ⟨F⟩ and [⟨F⟩, ⟨τ_eff⟩]. Total equivalent width
    is affine in ⟨F⟩, so ⟨τ_eff⟩ is the non-redundant second scalar. Raw-pixel RF (seed 0) as the
    recomputed v0.2-style baseline.
  - **Metrics.** 4-class balanced accuracy and the 6 pairwise BAs, with logistic regression and RF.
    ΔBA(eigenmap − PCA) at matched d with a group-bootstrap 95% CI. Fold-internal residualisation on
    [⟨F⟩, ⟨F⟩², ⟨τ_eff⟩], plus the τ-rescaled re-run. Headroom over the scalar baseline. A
    Voigt-converged-only sensitivity.
  - **Cluster diagnostics** (K = 4, 5; d = 8): stability ARI under 80% skewer-subsampling vs PCA +
    K-Means; largest-cluster fraction; η²(⟨F⟩); AMI vs class and vs the v2 K = 5 labels;
    cross-input AMI vs a permutation null; Spearman of the top-5 eigenvectors with ⟨F⟩ / ⟨τ_eff⟩;
    harmonic residuals.
- **[D-05] Compute: local, not Juno**: Juno was reachable, but the user's own `v6` job arrays were
  queued against `QOSMaxJobsPerUserLimit`, so submitting would compete with them. After [D-03] the
  catalogs cost ~40 min per variant locally (9 workers), and EV-2 is resumable per split. RF =
  150 trees, min_samples_leaf 5, max_samples 0.5 (compute-sized; identical for every representation).
- **[D-06] Smoke run: engineering checks pass; full run launched**: EV-1 smoke (256 skewers × 4,
  both variants): no crashes, convergence 97.7–99.2% (fiducial) and 98.4–98.8% (τ-rescaled). EV-2 smoke
  (2 seeds): 185 configurations per split, no NaN, every training graph connected, no short
  neighbour lists. Smoke-scale BA values are not interpreted (n = 1024). Full driver launched detached.

---

## 4. The Data (Lineage & Governance)

| Area | File | Metadata |
|---|---|---|
| **Input flux / τ** | `data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,tau,wave,vel}.npy` | (16384, 2048) float64; flux = roll(exp(−τ), 1); noiseless |
| **v2 K=5 labels** | `data/feature_discovery/labels_raw_k5.npy` | (16384,) — AMI target, indexed by skewer |
| **τ-rescale factors** | `results/eigen_voigt_embedding/artifacts/tau_rescale_factors.json` | A_c, target ⟨F⟩ = 0.977875 |
| **Voigt catalogs** | `results/eigen_voigt_embedding/artifacts/catalog_{fiducial,taurescaled}_{smoke,full}/class{1..4}.npz` | flattened (sightline, v, b, logN) + per-sightline diag + config; DVC-track the full ones (> 10 MB) |
| **EV-2 outputs** | `results/eigen_voigt_embedding/ev2_{variant}_{mode}/` | split_metrics.csv, summary_by_config.csv, le_minus_pca.csv, headroom_vs_scalar.csv, cluster_*.{csv,json}; oof_seed*.npz (DVC) |
| **Logs** | `results/eigen_voigt_embedding/logs/` | ev1_full.log, full.log |

### Responsibility Matrix
| Area | Owner |
|---|---|
| Voigt pipeline, embeddings, harness | core-implementer (implemented in-session by the orchestrating agent) |
| Data audit / loaders | data-engineer |
| Evaluation interpretation | PI → defense-panel |
| DVC tracking of artifacts | infrastructure-manager |

---

## 5. Evaluation Plan

Verb ceilings, binding on the full-run write-up:
- **"Beyond PCA"** requires the eigenmap − PCA ΔBA 95% CI to exclude 0 with Δ ≥ 0.02 at matched d, and
  the gain must survive both the ⟨F⟩ residualisation and the τ-rescaled re-run.
- **"Exceeds the per-sightline ceiling"** requires p16 above the recomputed grouped raw-pixel baseline.
- Anything that does not survive the confound controls is reported as **"re-derives ⟨F⟩"**.
- No "invariant" / "representation-independent" wording (rule 9).
- C2–C3 is expected near chance at a single sightline (pk-feedback-classifier [D-26]); it is reported,
  not spun.
- Cluster structure counts only if the largest cluster is < 70%, η²(⟨F⟩) < 0.30 and stability beats
  PCA + K-Means at matched d.

Prior-failure ledger this track inherits ([D-37] / rule 4): signal-clustering-v2 [D-13];
pk-feedback-classifier [D-01], [D-10]/[D-15], [D-26]; confusability / response-magnitude /
cluster-environment probes (NULL); signed-response S3 (TEMPERED). Named degeneracies to watch:
eigenvectors that track ⟨F⟩; one bulk cluster plus a thin tail; disconnected graphs; harmonic
eigenvectors; an unbounded Voigt component count.

---

## 6. Visualization & Artifacts

- **Run ID**: none (CPU in-process; no MLflow run yet — wire one for EV-2 full if results are reported).
- **Primary trace**: `results/eigen_voigt_embedding/ev2_fiducial_full/summary_by_config.csv` (pending).
- **Insights**: pending the full run.

---

## 7. Session History & Next Handoff

### Session Snapshot: October 18, 2026 (track open → smoke → full run launched)
- PI orientation + design review; data-engineer audit (noiseless, shared skewers, DLA-strength τ).
- VoigtFit validated and patched ([D-03]); region-based fitter; 14 unit tests pass.
- Smoke EV-1 (both variants) + EV-2 complete ([D-06]); full driver running detached:
  `nohup bash scripts/run_eigen_voigt_full.sh > results/eigen_voigt_embedding/logs/full.log 2>&1 &`.

### Immediate Next Steps
1. Let the full run finish (EV-1 τ-rescaled ≈ 40 min, then EV-2 fiducial and τ-rescaled; resumable).
2. Read the results against §5; PI interpretation; defense-panel on the result.
3. DVC-track the full catalogs and OOF arrays; wire an MLflow run for EV-2 if results are reported.

### Blockers
- None. Laptop CPU is shared with the user's `graph-clustering` jobs, so wall time may stretch.
