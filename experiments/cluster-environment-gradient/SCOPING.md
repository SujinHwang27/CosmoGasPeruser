# SCOPING — cluster-environment-gradient (thin exploratory probe)

Status: SCOPED (read-only PI ruling). NOT a stage. NO §1 Pulse entry.
Branch: `exp/cluster-environment-gradient` (off `feature/signal-clustering-v2` — see §6).
Caps: ONE script · ONE figure · ONE D-XX · NO new sim data · NO paper trigger.
Provenance of this ruling: PI-only, deferred-panel-review (does not gate compute or a paper claim).

> **Does not modify the reframe-suite numbers of record or the park-state
> disposition.** Origin: user challenge (2026-06) that v1's "Void/Forest/Dense"
> cluster labels are an *interpretive assumption* (abundance + cosmological prior
> + feedback-variance heuristic — `signal_clustering_analysis.md` §4.4 lines
> 251-270), NOT a measured per-cluster absorption result. This probe TESTS that
> assumption by measurement.

## 0. One-line question
Do cluster labels — clustered on absorption CONTENT — sort sightlines into a
monotone dense→void absorption-strength gradient (per-cluster distributions of
mean 1−F, saturated-pixel fraction, EW, line count, A6), and does that ordering
survive across representations? We MEASURE this; we do NOT inherit v1's
"Void/Forest/Dense" labels.

## 1. WHICH LABELS TO TEST (ruling + fallback order)

The environment question is about absorption CONTENT, so the labels under test
must come from CONTENT features, not separability geometry.

**PRIMARY (honest test) — re-cluster fresh on CONTENT features.** K-Means
(K=5, seed=42, matching v1/v2 convention) on per-sightline CONTENT, two arms:
  - **raw-content arm:** cluster a per-sightline content summary — the Tier-1
    absorption scalars the probe already measures (`total_ew`, `depth_mean`,
    `mean_1mF`, `saturated_frac`, `line_count`), z-scored. (Do NOT cluster the
    raw 2048-dim flux as headline — curse of dimensionality + noise.) This is the
    sanity floor: cluster on content, then test whether content-clusters order by
    content. If they don't even do THAT, the hypothesis is dead on arrival.
  - **wavelet arm:** cluster the wavelet content representation
    (`data/feature_discovery/wavelet_class{1..4}.npy` stacked, the per-level
    z-scored db8/l6 features; or the 512-dim `wavelet_db8_l6_d12/data.npy` if
    present and aligned). This is the REAL test: cluster on absorption morphology,
    test whether it orders by absorption strength.
  These two give the raw-vs-wavelet cross-representation test directly.

**SECONDARY (optional, within the ONE-script cap)** — IF the v1 content-cluster
label arrays are recoverable via DVC pull (`data/feature_discovery/
clustering_results*/`, wavelet-512 + DCT-480 clusterings v1 labeled), apply the
SAME battery to them → free wavelet-vs-DCT check tying back to the exact
partitions v1 named. Use ONLY if they load + align to the 16384-row order;
else SKIP silently (not a kill).

**EXCLUDED** — the v2 separability-vector clusters
(`data/feature_discovery/labels_{wavelet,raw}_k5.npy`, clustered on the 24-dim
`fingerprints_*.npy`). Wrong feature (feedback geometry, not content). MAY be run
as a clearly-tagged CONTRAST arm only ("separability-clusters, NOT content"); not
the test.

## 2. METRICS + GRADIENT TEST

Per-sightline content metrics (reuse `scripts/eda_sherwood.py`; add the 2 new),
computed on the SAME 16384-row order as the labels (see §7):
  - **M1 `mean_1mF`** = mean(1−flux) over 2048 px  [NEW — cleanest scalar; ties
    to A6 per `signal_clustering_analysis.md` line 223]
  - **M2 `depth_mean`** = mean of `absorption_depths()`  [exists]
  - **M3 `saturated_frac`** = fraction of px with (1−F) > τ  [NEW; τ=0.8 i.e.
    flux<0.2 pre-committed; report sensitivity at τ=0.9]
  - **M4 `total_ew`** = `total_equivalent_width()`  [exists]
  - **M5 `line_count`** = `len(detect_local_minima())`  [exists]
  - **M6 `A6_energy`** = per-sightline A6 coefficient energy (sum sq A6)  [exists]

Per cluster report the full DISTRIBUTION (median, IQR, violin/box). Rank clusters
by median(M1) → candidate dense→void ordering.

**GRADIENT PASS test (pre-committed):**
  - **(G-a) Monotonicity:** order clusters by median(M1); for each of M2..M6 the
    cluster medians must be monotone under that ordering. Spearman ρ between
    cluster-rank(M1) and cluster-median(Mj). Bar: |ρ| ≥ 0.9 for ≥4 of 5 metrics.
  - **(G-b) Separation:** adjacent clusters (in M1 order) distributionally
    distinct — KS distance between adjacent-cluster M1 distributions ≥ 0.2 for
    ≥ K−2 of the K−1 adjacencies (≤1 fused adjacency). Guards against a smooth
    continuum mislabeled as discrete environments.

**CROSS-REPRESENTATION stability (pre-committed):**
  - **(X)** Build the rank ordering independently for raw-content vs wavelet arms
    (+ DCT if SECONDARY). Match clusters by contingency overlap
    (`src.core.cluster.contingency_matrix`). STABLE if rank-correlation of
    matched-cluster median(M1) across arms ρ ≥ 0.8 AND best-overlap matching
    assigns ≥ 60% of mass consistently.

## 3. PRE-COMMITTED PASS / NULL / KILL (rule-5 symmetric)

**PASS** (confirms the hypothesis at the MEASUREMENT level): (G-a) AND (G-b) AND
(X). Report: "content-clusters sort sightlines into a monotone, distributionally-
separated, representation-stable absorption-strength gradient." Carry the §5
ceiling: this is necessary-but-not-sufficient for "dense gas vs void."

**NULL** (honest negative — valid end state, report as-observed): any of —
ordering non-monotone (ρ bar missed on ≥2 metrics); gradient is a smooth
continuum not discrete (G-b missed on ≥2 adjacencies); ordering does NOT survive
raw-vs-wavelet (X missed). Report verbatim: "content-clusters do NOT order into a
monotone/separated/stable absorption gradient; v1's Void/Forest/Dense labels are
not supported by measured per-cluster absorption." Do NOT soften, do NOT re-run
searching for a passing K. K=5 pre-committed; K only a one-line sensitivity note.

**KILL** (malformed probe — do NOT log a finding): inherits/restates v1's
environment labels as if measured (any "Void/Forest/Dense" asserted as result not
hypothesis) → KILL. Per-sightline metrics NOT verifiably aligned to label
row-order (§7 fails) → KILL. Headline run on the EXCLUDED separability labels as
if content → KILL.

## 4. ANTI-SCOPE-CREEP + REUSE (hard boundary)
REUSE: `scripts/eda_sherwood.py` extractors; `src.core.data` (canonical row
order); wavelet `wavelet_class*.npy`; `src.core.cluster.fit_kmeans` +
`contingency_matrix` (K=5, seed=42); on-disk label .npy (SECONDARY/CONTRAST only).
ADD (only new logic permitted): `mean_1mF` (M1) + `saturated_frac` (M3) — two
one-line functions in `src/core/` (NOT in the script, per CLAUDE.md); ONE
orchestration script `scripts/run_cluster_env_gradient.py` (thin wrapper).
DO NOT: load/regenerate sim data; touch `data/`; add MLflow/dvc.yaml stages;
sweep K beyond K=5 + one footnote; produce >1 figure; add >1 D-XX.

## 5. GUARDRAILS (binding)
- NOT a paper trigger under PASS/NULL/KILL. Paper-trigger user-owned.
- Does NOT reopen the parked reframe-suite verdict.
- **CEILING CLAIM (mandatory in the D-XX + any prose):** a measured absorption
  gradient supports "clusters sort by absorption strength." It does NOT establish
  "dense gas cloud vs cosmic void" — there is no ground-truth halo/density/optical-
  depth catalog in this project. The environment LABELS remain interpretive even
  on PASS ([D-37]).
- Honest-reporting BOTH directions: lead with observed Spearman/KS numbers before
  any narrative; do not inflate a partial pass; do not flagellate a clean NULL.

## 6. BRANCH
`exp/cluster-environment-gradient`, branched off **`feature/signal-clustering-v2`**.
(PI ruling said off `main`; deviated because `main` is ~146 commits behind and
lacks the clustering code/data this probe needs. `feature/signal-clustering-v2`
is the natural clustering-track home and is still isolated from the parked
pk-feedback/reframe manuscript state — the isolation intent is preserved.)
Artifacts: `experiments/cluster-environment-gradient/{SCOPING.md, artifacts/}`;
one figure under `results/cluster_environment_gradient/figs/`.

## 7. HARD PRECONDITION (gate before any compute)
Assert IN-SCRIPT that the per-sightline metric vector and the cluster-label vector
index the SAME 16384(×4) sightlines in the SAME order (same loader, same class
concatenation as `scripts/run_cluster.py`). `eda_sherwood.py` aggregates to
per-class means and discards per-sightline index, so the probe MUST recompute
per-sightline metrics through the canonical loader order and assert
`len == labels.shape[0]` + class-block alignment. Misalignment → KILL.

## 8. Status
RUN & CLOSED — **NULL (informative)**, 2026-06-23. Implemented in
`src/core/env_metrics.py` + `src/core/cluster_env_probe.py`, driven by
`scripts/run_cluster_env_gradient.py`; outputs in `results/cluster_environment_gradient/`.

## 9. [D-01] — Outcome: NULL (pre-committed valid end state)

Observed (K=5, seed=42, 65,536 sightlines; numbers before narrative):

- **Wavelet arm** (512-dim D3–D6+A6 of A=1−F, standardized — the REAL test;
  noisy D1/D2 dropped per v1 line 220; an earlier full-2048-dim run collapsed
  99.97% into one cluster and was discarded as degenerate):
  - cluster sizes 32206 / 28265 / 3100 / 1897 / **68** (largest 49% — non-degenerate).
  - **G-a (monotonicity) PASS:** Spearman(M1-rank, cluster-median Mj) =
    depth_mean 1.00, total_ew 1.00, A6_energy 1.00, line_count 0.95,
    saturated_frac 0.71 → 4/5 ≥ 0.9.
  - **G-b (separation) FAIL:** the four low-absorption clusters (median 1−F
    0.017–0.024) are NOT distributionally distinct (adjacent-cluster KS ≈ 0
    between them); only the tiny 68-sightline saturated cluster (median 1−F 0.53)
    stands apart. Monotone ordering YES; discrete environment strata NO.
- **Cross-representation (X) FAIL on membership:** raw-summary vs wavelet arms
  agree on the absorption ORDERING (matched-cluster median-M1 Spearman 0.95) but
  only ~50% of sightlines co-cluster (matched mass 0.496 < 0.60 bar). Ordering
  stable; membership not (echoes the v2 cross-run overlap instability).
- **Raw-summary arm** (sanity floor, near-circular — clusters on the absorption
  scalars): G-a + G-b pass trivially; sizes 35285/24431/5656/134/30 — same shape:
  two huge low-absorption bulks + tiny saturated groups.

**VERDICT — NULL.** The absorption axis IS a real organizing dimension (clusters
order monotonically by it), but the population is overwhelmingly ONE low-absorption
bulk (~92% of sightlines in two clusters, median 1−F ≈ 0.018) plus a thin
saturated/DLA tail (68 sightlines, median 0.53) — NOT five discrete, separated,
representation-stable dense→void environment classes. **v1's "Void/Forest/Dense"
labels are NOT supported as measured discrete strata.** CEILING (binding): even the
real monotone ordering supports only "sightlines vary in absorption strength," not
"dense gas cloud vs cosmic void" (no halo/density catalog). Consistent with v1's
77%-bulk and the EW-distribution overlap. NOT a paper trigger; parked verdict
unchanged.
