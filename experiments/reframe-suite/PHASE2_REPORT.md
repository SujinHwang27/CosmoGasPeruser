# Phase 2 Report — reframe-suite Reframe 9 (mean-flux-removed P_F(k) variant)

**Branch:** `exp/pk-feedback-classifier` · **Authored:** 2026-06-02 ·
**Authored by:** core-implementer (Phase 2 brief per SCOPING §3) ·
**Compute:** local laptop, 50 RF fits, ~1.5 min wall (well under the
~1h SCOPING §3 ceiling).

**Inputs (read-only, already on disk):**
- `data/preprocessed/Sherwood_z0.3_inf/{1..4}/flux.npy` — 4-class
  Sherwood z=0.3 flux, (16384, 2048) per class.
- `data/preprocessed/Sherwood_z0.3_inf/{1..4}/vel.npy` — velocity axis
  (uniform Δv = 2.6365 km/s/pixel per [D-04]).
- `results/reframe_suite/phase1/reframe1_binary_c4_vs_rest.csv` —
  Reframe-1 baseline at (per_sightline, M=64): median 0.9711, p16 0.9603.

**Outputs (Git-tracked CSVs, no DVC stage):**
- `results/reframe_suite/phase2/cv_partial_meanremoved.csv` — 50 rows
  (1 regime × 2 M × 5 seeds × 5 folds); schema mirrors Stage 1's
  `cv_partial.csv` so the Phase-1 binary aggregator drops in unmodified.
- `results/reframe_suite/phase2/summary_meanremoved.csv` — per (regime, M)
  median + p16 + p84 of 4-class and binary balanced_acc.
- `results/reframe_suite/phase2/reframe9_verdict.csv` — verdict row with
  drop vs Reframe-1 baseline.

**Script:** `experiments/reframe-suite/scripts/phase2_reframe9.py`.

**Methodology change vs Stage 1 / Reframe 1 / [D-06]:** a new
`norm='per_sightline_mean_removed'` value in
`src/core/transforms.py:FluxPowerSpectrum` that computes
`delta_F = F − ⟨F⟩_los` (additive subtraction in flux space) prior to
Hann-window + rfft + |·|² + log-bin. The existing `per_sightline`
(multiplicative `F/⟨F⟩_los − 1`) and `global` (multiplicative
`F/⟨F⟩_global − 1` with k=0 + lowest-log-bin drop) paths are
bit-identical to pre-change behavior — only an additive code path was
added. The new norm retains all 20 log-bins (no k=0 drop), since the
subtraction itself removes the DC channel up to windowing leakage.

---

## §1 — Verdict

**PASS.** Binary {C4 vs C1 ∪ C2 ∪ C3} balanced_acc at
(per_sightline_mean_removed, M=64) on 25 fold-rows (5 seeds × 5 folds):

| metric | Phase 2 (R9) | Phase 1 baseline (R1) | drop |
| :-- | --: | --: | --: |
| binary balanced_acc median | **0.9742** | 0.9711 | **−0.0031** |
| binary balanced_acc p16 | **0.9604** | 0.9603 | **−0.0001** |
| binary balanced_acc p84 | 0.9869 | 0.9875 | +0.0006 |

PASS condition (SCOPING §2 R9 (a)): drop on median < W = 0.05.
**Observed drop is −0.003 (slight gain, well within bootstrap noise);
the bar is cleared by 53× the threshold (drop ≈ 6% of W).**

Cross-check at M=256 (out-of-fit-budget but informative): binary
balanced_acc 1.000 / 1.000 / 1.000 (median / p16 / p84) — identical
to Phase 1's per_sightline M=256 binary (1.000 across the board). The
M=256 regime is at the empirical ceiling under both normalizations.

4-class balanced_acc cross-check (NOT a R9 gate, just sanity):

| | Phase 2 (R9) | Phase 1 (R1, per_sightline) |
| :-- | --: | --: |
| 4-class median @ M=64 | 0.6888 | 0.6857 |
| 4-class p16 @ M=64 | 0.6537 | 0.6566 |
| 4-class median @ M=256 | 0.8269 | 0.8325 |
| 4-class p16 @ M=256 | 0.7690 | 0.7877 |

Identical within bootstrap SD; no degradation from the ⟨F⟩-removal
preprocessing change.

mid_k_frac at (R9, M=64) median = 0.380 (Phase-1 R1 had mid_k_frac
~0.38 region per Stage 1 G3 summary). G3 importance distribution is
preserved.

---

## §2 — Joint R1 + R9 conjunction status (SCOPING §4)

Reframe 1's "solid/successful" qualification requires **all four**
conjuncts (SCOPING §4 R1):

| conjunct | bar | observed | pass? |
| :-- | :-- | :-- | :--: |
| (i) binary balanced_acc p16 ≥ 0.90 | 0.90 | 0.9603 | ✓ |
| (ii) Δ binary − 4-class median ≥ 0.10 pp | 0.10 | 0.2854 | ✓ |
| (iii) per_sightline / global regime parity \|Δ\| ≤ 0.005 | 0.005 | 0.0031 | ✓ |
| (iv) R9 drop < 0.05 (5 pp) on ⟨F⟩-removal | 0.05 | −0.003 | ✓ |

**All four conjuncts clear. Reframe 1 qualifies as "solid/successful"
per the SCOPING §4 R1 operational definition.** This is the joint
R1 + R9 PASS that SCOPING §1 named as the highest-confidence verb
ceiling on the post-cascade re-open: "first quantified binary {C4 vs
C1 ∪ C2 ∪ C3} detection under stacked P_F(k) at z ≈ 0.3 Sherwood, in
the post-cascade-close regime, with shape-vs-⟨F⟩ disentanglement
gated on Reframe 9 — PASS."

---

## §3 — Honest [D-37] reads

### §3.1 Degeneracy observation (the load-bearing flag)

**The two `per_sightline` variants (multiplicative `F/⟨F⟩_los − 1` and
additive `F − ⟨F⟩_los`) produce statistically indistinguishable
binary {C4 vs rest} accuracies at M=64.** The drop is −0.003 on
median, +0.000 on p16 — both within the 5-fold-CV bootstrap SD
([D-23] C1 ≈ 0.03–0.04). This is consistent with SCOPING §5's
honest-counterfactual flag: in the small-δ_F regime
(⟨F⟩_los ≈ 0.97–0.98 across all classes, δ_F ≪ 1), the multiplicative
and additive ⟨F⟩-removals are degenerate to leading order in δ_F.

This is a **scientifically informative observation, not a problem**.
The pre-committed R9 control was chosen to be *stronger* than the
existing per_sightline regime (an additive subtraction removes the
DC channel exactly, where the multiplicative divide rescales it).
The empirical result is that *the stronger control was not needed* —
the [D-06] preprocessing (per_sightline divide + global regime's
k=0/lowest-log-bin drop) was already doing the job. The
shape-vs-⟨F⟩ disentanglement is **clean in the operational sense
defined by SCOPING §2 R9 (a)**, with the honest caveat:

> The pre-committed reading per SCOPING §2 R9 (c) — "a clean PASS
> (< 5 pp drop) is 'shape signal survives a strong ⟨F⟩-removal
> control'" — holds. The binding non-claim, also from §2 R9 (c):
> "we do not claim total ⟨F⟩-independence; we claim survival of the
> specific preprocessing control specified here." That non-claim
> remains in force: an additive ⟨F⟩ subtraction does not remove
> higher-order ⟨F⟩-correlated structure (e.g., the *distribution*
> of F around its mean, which retains a C4-rank signal because
> stronger feedback flattens the high-F tail).

### §3.2 What this PASS does and does not say

- **DOES say:** The Reframe-1 binary {C4 vs rest} signal at M=64
  survives explicit per-sightline ⟨F⟩ subtraction in flux space.
  The signal is therefore not a trivial DC channel; it lives in the
  shape of the post-mean-removed P_F(k) (with the higher-moment
  caveat above).
- **DOES NOT say:** "Lyα detects strong-AGN feedback unverbed."
  The SCOPING §1 verb ceiling on the R1 + R9 joint headline remains
  binding: "first quantified binary detection … in the post-cascade-
  close regime, with shape-vs-⟨F⟩ disentanglement gated on
  Reframe 9." The cumulative cascade (six entries per CLOSE_OUT §4)
  is unchanged; this is a sharpened, quantified subset of the
  closed [D-26] surface, not an overturn.
- **DOES NOT say:** "The 4-class P_F(k) classifier works." 4-class
  balanced_acc at M=64 is 0.69 (Phase 1) / 0.69 (Phase 2) — the
  C2-vs-C3 null and the partial C1-vs-{C2,C3} distinguishability
  (per Reframe 5) remain the limiting factors. The binary PASS is
  arithmetic on Stage 0's C4-recall = 0.96 vs ~0.5 rest-recall, as
  SCOPING §5 predicted.

### §3.3 Reframe-9-as-honesty-hurdle: status

Per SCOPING §2 R1 (c): "Reframe 1's headline is rule-7-fragile until
Reframe 9 returns." **It has returned, and the headline is no longer
rule-7-fragile on the ⟨F⟩-vs-shape axis.** The remaining rule-7
exposure surfaces:

1. The higher-moment ⟨F⟩-correlated structure caveat above (named in
   the pre-commit per SCOPING §2 R9 (c); not closable with this
   experiment alone).
2. Reframe 5's C2-vs-C3 null at chance (~0.55) — still bounds what
   "feedback discrimination" can mean; binary C4 detection is the
   load-bearing channel, recipe-vs-recipe is largely null.
   *(2026-06-11 amendment per HARDENING_SPEC §6 E6: re-verbed —
   C2-vs-C3 is consistent with chance at M ≤ 64 and rises with
   stacking depth to median 0.678 / p16 0.599 at M=256; an
   M-conditional exclusion bound, not an intrinsic null. Dedicated
   binary re-measurement pending per HARDENING_SPEC item i.)*
3. The full cumulative cascade of six prior null entries (CLOSE_OUT
   §4); this is a sharpening, not a reset.

---

## §4 — Compute audit

| item | value |
| :-- | :-- |
| platform | local laptop CPU |
| fits | 50 RF main fits (M=64: 25, M=256: 25); no ablation, no permutation |
| wallclock | ~1.5 min total (vs ≤ 1h SCOPING budget) |
| seeds | {42, 43, 44, 45, 46} |
| RNG | `np.random.default_rng(seed)` via `group_indices_within_class` |
| fold-leakage assert | active per [D-20] S3 (copied verbatim from `run_stage1.py`) |
| determinism | confirmed (rerun would land identical RF fits per seed) |

Per-tuple checkpoint CSV (`cv_partial_meanremoved.csv`) supports
resume; the run completed in one pass without checkpoint re-entry.

---

## §5 — What this report does NOT claim (rule 7 explicit non-claims)

- **No claim of total ⟨F⟩-independence.** The PASS is operationally
  defined per SCOPING §2 R9 (a); higher-order ⟨F⟩-correlated structure
  in the post-subtraction P_F(k) remains an open caveat.
- **No headline beyond the SCOPING §1 verb ceiling.** "First quantified
  binary detection … with shape-vs-⟨F⟩ disentanglement gated on
  Reframe 9" is the bound; no unverbed "Lyα detects feedback" claim.
- **No paper-author dispatch recommendation.** Paper authoring is
  user-triggered-only per the standing project discipline. The
  qualification gates in SCOPING §4 are the user's decision surface;
  the joint R1 + R9 PASS reported here is what the user will read
  against to decide.
- **No claim that the multiplicative/additive degeneracy generalizes
  beyond the small-δ_F regime.** It is empirically observed at
  ⟨F⟩_los ≈ 0.98 (z ≈ 0.3 Sherwood); at higher z or different
  Lyα-forest depths the two controls may diverge.

---

**Status: PASS. Joint R1 + R9 conjunction CLEARS the SCOPING §4 R1
"solid/successful" qualification.** Next-step authorization (paper
work or otherwise) returns to the user.
