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
| EV-1 — Voigt catalogs | VoigtFit decomposition of all 4 × 16384 sightlines, fiducial + τ-rescaled | ✅ **DONE** — 98.2–98.9% converged, 0 crashes |
| EV-2 — embeddings + evaluation | spatially-blocked CV: PCA vs Laplacian eigenmaps on raw / Voigt / P_F(k); confound controls; clusters | ✅ **DONE** — pre-registered **NULL** ([D-08], [D-09]) |
| EV-3 — interpretation + panel | read full results against §5 ceilings; post-hoc checks; figures | ✅ **DONE** (PI-only); defense-panel on the result optional ([D-09]) |

### ✅ Completed Milestones
- **2026-10-18**: track opened; PI APPROVE-WITH-CAVEATS (provisional); data audit complete.
- **2026-10-18**: VoigtFit pipeline validated (exact synthetic recovery incl. saturated/LLS/DLA-strength lines); 4 VoigtFit defects found and patched.
- **2026-10-18**: smoke run (256 skewers × 4 classes) of EV-1 (both variants) and EV-2 (2 seeds) end to end; full run launched.
- **2026-10-18**: defense-panel protocol review (NEEDS WORK) → 7 amendments applied before any full EV-2 result ([D-07]).
- **2026-10-19**: full run complete (both variants, 10 seeds × 5 spatial-block folds, 293 configs/split). Eigenvector embeddings: NULL ([D-08], [D-09]). Exploratory lead: Voigt line widths on the 2–3 axis.

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
- **[D-07] Defense-panel protocol review (NEEDS WORK) → amendments, applied before any full-run EV-2
  result existed**: the driver was stopped while still waiting on EV-1, so no full EV-2 split ran
  under the old protocol. The EV-1 catalogs are unaffected; every finding concerns EV-2.
  - **#1 KILLER, fixed: Voigt imputation leaked the label.** Features were built per class, so missing
    entries got a within-class median, a constant an RF can map back to the class. Features are now
    NaN-preserving, imputed with the training-fold median inside each split. The kNN metric uses a
    label-free pooled fill. Missingness flags `has_dv_nn` and `has_bN_slope` were added (33 features).
  - **#2, fixed: linear residualisation was lopsided against PCA.** It is replaced by (resid=1)
    fold-internal additive cubic splines (48 knots) on rank-transformed scalars, which removes harmonics
    of rank ⟨F⟩; a unit test confirms cos(6πu) is removed. A new increment test (resid=2) gives the
    classifier [scalars ∪ embedding].
  - **#5, fixed: spatial leakage.** Skewers are randomly placed (~0.8 Mpc/h spacing, 3 projection axes),
    so CV folds and the bootstrap now use 192 spatial blocks (axis × 8×8 transverse cells of
    7.5 Mpc/h), not individual skewers.
  - **#6, fixed: train/test embedding mismatch.** Training rows are now embedded by the same
    leave-self-out Nyström map as the test rows. The masked graph is primary; unmasked is secondary.
  - **#7, fixed: scalar naming.** The second scalar is renamed ⟨τ⟩_clip, with τ clipped at 30 (not 700),
    so it is not a count of underflowed pixels.
  - **#4, fixed: statistics.** Seed-averaged correctness with block bootstrap CIs, a simultaneous max-T
    band over all Δ rows, a Nadeau–Bengio corrected t, and a pre-declared primary cell (§5).
  - **#10 / #12 / #13.** Voigt detection-threshold sensitivity added (log N ≥ 12.5, 13; direct
    features). Seed-0 probabilities saved. Raw-pixel RF baseline run on 3 seeds.
  - **Clusters.** An ⟨F⟩-only K-Means null was added.
  - **Carried to EV-3 as post-hoc checks:** #3 (τ-rescaling can create class signal: compare
    per-feature class AUC fiducial vs rescaled, and ba_14 / ba_24); #8 (read the RF Δ as the geometry
    test and report the non-harmonic count next to d); #9 (converged-only filters predictions without
    refitting; report ⟨F⟩ of kept vs dropped skewers).
  - **Re-smoke:** 293 configurations per split, end to end, no NaN.
- **[D-08] Full fiducial EV-2 (10 seeds × 5 spatial-block folds): eigenmaps do NOT beat PCA**
  (pre-registered primary cell: masked eigenmap vs PCA, RF, d = 8).
  - **resid = 0.** ΔBA = −0.049 (raw), −0.022 (Voigt), −0.023 (P_F(k)); all 95% CIs below zero, so
    eigenmaps are significantly worse.
  - **resid = 2 (increment).** −0.007 / −0.015 / −0.018, also below zero.
  - **resid = 1 (spline-residualised).** Positive: +0.026 (Voigt), +0.021 (P_F(k)), simultaneous lower
    bounds > 0, but −0.012 for raw. This is PCA losing more when ⟨F⟩ is removed, not a better
    representation: both trail the direct features, and the increment test is negative. The §5
    criteria require all three modes, so **"beyond PCA" fails for every input**.
  - **Exploratory rows.** The only other cells with a simultaneous lower bound above 0 are
    logistic-regression rows (raw, increment mode, d ≥ 8, Δ ≤ 0.016). Per panel #8 a logistic gain can
    come from nonlinearity alone, so these are not a geometry result.
  - **Ceiling.** The recomputed spatially-blocked raw-pixel RF baseline is BA 0.477 (p16 0.470,
    15 splits), above the old ungrouped 0.451. Primary cells (0.29–0.42) fall below it, so **"exceeds
    the ceiling" fails**. The best compact representations are direct P_F(k) (0.441, increment mode)
    and direct Voigt features (0.432–0.439), each ≈ +0.10 above the ⟨F⟩, ⟨τ⟩ scalars (0.340).
  - **Exploratory, not pre-registered (verb ceiling: "observed"; needs its own pre-registered binary
    test).** On the historically null 2–3 pair, direct Voigt features give pairwise BA 0.576 (p16 0.572;
    unchanged in increment mode), vs raw pixels 0.544, P_F(k) 0.540 and scalars 0.515; on 1–3 they give
    0.584. This matches the catalog-level result (EV-3): line-width features separate 2 vs 3 at
    univariate AUC 0.589 fiducial / 0.588 τ-rescaled, and class 4 broadens lines (median b 31.1 vs
    24.8 km/s) in both variants, consistent with Nasir+2017 ("AGN … producing some broader lines").
    Line-count separability largely vanishes under τ-rescaling (|AUC − 0.5| for 2–4 falls 0.283 → 0.007),
    so counts mostly carry ⟨F⟩, whereas widths survive.
  - **Voigt converged-only skewers.** 95.2% kept; ⟨F⟩ kept 0.9786 vs dropped 0.9635, so the dropped
    skewers are more absorbed (panel #9).
- **[D-09] τ-rescaled EV-2 + clusters → track verdict: eigenvector embeddings NULL; Voigt line widths are
  the durable, exploratory lead.**
  - **τ-rescaled primary cell.** Eigenmap − PCA is negative in every mode for every input: raw −0.016 /
    −0.001 / −0.007, Voigt −0.031 / −0.012 / −0.027, P_F(k) −0.043 / −0.030 / −0.041 (resid 0 / 1 / 2).
    The fiducial spline-residualised positives do not survive. **"Beyond PCA" fails in both variants
    for all inputs.**
  - **The control works.** The ⟨F⟩, ⟨τ⟩ scalars fall from 0.340 to 0.282 BA (chance 0.25) after
    rescaling.
  - **Why eigenmaps lose: they re-derive ⟨F⟩.** Leading eigenvector vs ⟨F⟩: P_F(k) φ1 Spearman 0.91
    (both variants); Voigt φ1–φ5 |ρ| 0.52–0.77. The harmonic test leaves 1 (P_F(k)), 4–5 (Voigt) and
    3 (raw) non-harmonic coordinates among the first 8. Same-skewer neighbour share is low (raw 14%,
    P_F(k) 7%, Voigt 2–3%), and masked and unmasked results agree.
  - **Spectral clusters fail the §5 criteria in both variants (K = 4).**
    - Voigt eigenmap: degenerate, largest cluster 0.95–0.96.
    - Raw eigenmap: ⟨F⟩-dominated, η² 0.56, largest cluster 0.79.
    - P_F(k) eigenmap: stability 0.948 / 0.924, below PCA + K-Means at 0.968 / 0.956. AMI vs class
      0.018 / 0.007, against 0.008 / 0.001 for the ⟨F⟩-only null.
    - AMI vs the v2 K = 5 labels ≤ 0.016.
  - **Raw-pixel RF rises under rescaling** (0.477 → 0.505, with 1–3 0.554 → 0.616). This is the
    rescaling-artefact direction panel #3 warned about, so it is not read as feedback signal.
  - **Exploratory lead (not pre-registered; verb ceiling "observed").** Direct Voigt features give
    2–3 pairwise BA 0.575 (p16 0.571) τ-rescaled vs 0.576 fiducial, unchanged. That compares with raw
    pixels 0.544 / 0.544 and P_F(k) 0.528–0.537. It holds at log N ≥ 12.5 (0.570) and weakens at
    log N ≥ 13 (0.539). The catalogs locate it in line widths: b-feature AUC 0.589 → 0.588, and class 4
    median b 31.1 → 31.3 km/s vs ~24.8 for class 1. Caveats: the pairwise BA comes from a 4-class
    model, the data are noiseless with no LSF, and the cut was not chosen in advance.
  - **Net.** The user's eigenvector idea is tested in full and is a clean, well-controlled NULL on
    raw spectra, on Voigt features and on P_F(k). The Voigt-library feature extraction is the
    productive half: it gives a physically interpretable per-sightline representation that matches
    P_F(k) on 4-class BA (0.42–0.44) and carries a mean-flux-robust line-width signal on the
    hardest pair. CEILING: no representation beats the spatially-blocked raw-pixel RF (0.477
    fiducial). Not a paper trigger; parked verdict unchanged. PI-only, non-provisional for the NULL
    (narrowing, rule 15); the 2–3 lead is explicitly not a claim.

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

Verb ceilings, binding on the full-run write-up (amended by [D-07], fixed before any full result):
- **Primary cell (pre-declared):** masked Laplacian eigenmap vs PCA, random forest, d = 8, for each
  input (raw / Voigt / P_F(k)). All other cells are exploratory and are read only through the
  simultaneous band.
- **"Beyond PCA"** for an input requires, in BOTH the fiducial and the τ-rescaled variant:
  - resid = 0: Δ ≥ 0.02 (point estimate; a practical-significance bar set by the PI before data), with
    the simultaneous 95% band's lower bound above 0 and Nadeau–Bengio one-sided p < 0.05;
  - resid = 1 (spline-residualised): block-bootstrap 95% CI lower bound above 0;
  - resid = 2 (increment test): Δ > 0 with block-bootstrap 95% CI lower bound above 0.
- **"Exceeds the per-sightline ceiling"** requires the primary cell's seed-averaged BA to exceed the
  recomputed spatially-blocked raw-pixel RF baseline, with Nadeau–Bengio p < 0.05.
- Anything that does not survive the confound controls is reported as **"re-derives ⟨F⟩"**.
- No "invariant" / "representation-independent" wording (rule 9).
- C2–C3 is expected near chance at a single sightline (pk-feedback-classifier [D-26]); it is reported,
  not spun.
- Cluster structure counts only if the largest cluster is < 70%, η²(⟨F⟩) < 0.30, stability beats
  PCA + K-Means at matched d, and AMI vs class beats the ⟨F⟩-only K-Means null.

Prior-failure ledger this track inherits ([D-37] / rule 4): signal-clustering-v2 [D-13];
pk-feedback-classifier [D-01], [D-10]/[D-15], [D-26]; confusability / response-magnitude /
cluster-environment probes (NULL); signed-response S3 (TEMPERED). Named degeneracies to watch:
eigenvectors that track ⟨F⟩; one bulk cluster plus a thin tail; disconnected graphs; harmonic
eigenvectors; an unbounded Voigt component count.

---

## 6. Visualization & Artifacts

- **Run ID**: none (CPU in-process, no MLflow run; provenance in each `ev2_*_full/meta.json`, git-stamped).
- **Primary trace**: `results/eigen_voigt_embedding/ev2_{fiducial,taurescaled}_full/le_minus_pca.csv`
  (+ `summary_by_config.csv`, `headroom_vs_scalar.csv`, `cluster_diagnostics.csv`, `diagnostics.json`).
- **Figures**: `results/eigen_voigt_embedding/figs/fig_le_minus_pca_full.png` (eigenmap − PCA vs d, all
  inputs × variants × confound modes), `fig_representation_ba_full.png` (per-sightline BA per
  representation), `fig_voigt_b_by_class.png` (Voigt b CDFs by class).
- **Catalog post-hoc**: `results/eigen_voigt_embedding/ev3_catalog_posthoc.json`.
- **Insights**: graph-Laplacian eigenvectors are a worse compression than PCA here and their leading
  coordinates restate ⟨F⟩; the Voigt decomposition's lasting value is the line-width channel, which
  survives mean-flux equalisation and leads on the StellarWind-vs-WindAGN axis (exploratory).

---

## 7. Session History & Next Handoff

### Session Snapshot: October 18, 2026 (track open → smoke → full run launched)
- PI orientation + design review; data-engineer audit (noiseless, shared skewers, DLA-strength τ).
- VoigtFit validated and patched ([D-03]); region-based fitter; 14 unit tests pass.
- Smoke EV-1 (both variants) + EV-2 complete ([D-06]); full driver running detached:
  `nohup bash scripts/run_eigen_voigt_full.sh > results/eigen_voigt_embedding/logs/full.log 2>&1 &`.

### Session Snapshot: October 19, 2026 (full run complete → verdict)
- Full EV-1 (2 × 65536 sightlines) and EV-2 (2 variants × 50 splits) complete; [D-08], [D-09] recorded.
- Catalogs + OOF arrays DVC-tracked and pushed (`artifacts.dvc`, `ev2_*_full/oof_*.npz.dvc`).

### Immediate Next Steps (user-owned; none auto-dispatched)
1. If the 2–3 line-width lead is worth pursuing: pre-register a dedicated binary 2-vs-3 test on Voigt
   b-features (spatial-block CV, both variants, a detection-threshold sweep, an LSF/noise-degraded
   variant), with a defense-panel pass BEFORE running it.
2. Optional: defense-panel review of the [D-09] verdict text (a narrowing NULL needs no panel under
   rule 15; the exploratory lead is not a claim).
3. Not a paper trigger; parked project verdict unchanged.

### Blockers
- None.
