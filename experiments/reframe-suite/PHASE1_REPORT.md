# Phase 1 Report — reframe-suite (Reframes 1 + 5 + 7)

**Branch:** `exp/pk-feedback-classifier` · **Authored:** 2026-06-02 ·
**Authored by:** support-researcher (Phase 1 brief per SCOPING §3) ·
**Compute:** local laptop, no new training, ~30 s wallclock incl.
line-peak counts.

**Inputs (all read-only, already on disk):**
- `results/pk_feedback_classifier/stage1/cv_partial.csv` (700 rows = 2
  regimes × 7 M × 10 seeds × 5 folds; `confusion_flat` is row-normalized
  recall, per `experiments/pk-feedback-classifier/run_stage1.py:298-303`)
- `data/feature_discovery/labels_{wavelet,raw}_k5.npy` (each (16384,))
- `data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,tau}.npy`

**Outputs:**
- `results/reframe_suite/phase1/reframe1_binary_c4_vs_rest.csv`
- `results/reframe_suite/phase1/reframe5_lattice.csv`
- `results/reframe_suite/phase1/reframe7_cluster_axes.csv`

**Analysis script:**
`experiments/reframe-suite/scripts/phase1_reframes.py` — deterministic;
no RNG.

---

## §1 — Schema verification (honest pre-flight)

- `cv_partial.csv` is **700 rows, not 1400** (the brief stated 1400). The
  Stage-1 grid was 2 regimes × 7 M × 10 seeds × 5 folds = 700 fold-level
  rows, with 50 rows per (regime, M) cell. The `confusion_flat` column
  exists and parses to a 4×4 row-normalized recall matrix.
- Confusion-matrix indices 0..3 correspond to true labels 1..4 (NoFB,
  StellarWind, WindAGN, WindStrongAGN) per `run_stage1.py:299`
  (`labels_all = np.array([1, 2, 3, 4])`).
- K=5 cluster labels are 0..4 in both files; both arrays are shape
  (16384,) — these are **per-position** labels (each entry indexes a
  spatial position with one realization in each of the 4 classes), not
  per-(class, position). Reframe 7 physical axes are reported as
  class-averaged headline + per-class breakouts (see §4 honesty
  hurdle).

---

## §2 — Reframe 1 verdict

**PASS.** Binary {C4 vs C1 ∪ C2 ∪ C3} at (per_sightline, M=64) on the 50
fold-rows:

| metric | value |
| :-- | --: |
| binary balanced_acc median | **0.9711** |
| binary balanced_acc p16 (lower CI edge) | **0.9603** |
| binary balanced_acc p84 | 0.9875 |
| 4-class balanced_acc median (same cell) | 0.6857 |
| Δ binary − 4-class median | **+0.2854 pp (× 100)** |
| global @ M=64 binary median | 0.9741 |
| regime parity \|per_sightline − global\| | 0.0031 (≤ 0.005 ✓) |

PASS conditions (SCOPING §2 R1(a)):
- p16 ≥ 0.90 ✓ (0.9603)
- Δ vs 4-class median ≥ +0.10 pp ✓ (+0.285)
- Regime parity sub-check (per_sightline ≈ global, \|Δ\| ≤ 0.005) ✓ (0.0031)

## §3 — Reframe 5 verdict

**PASS.** Per_sightline @ M=64 pairwise distinguishability lattice (p16):

| pair | median | p16 | clears 0.75? |
| :-- | --: | --: | :--: |
| C1-C2 | 0.880 | **0.842** | ✓ |
| C1-C3 | 0.802 | **0.741** | ✗ (just under) |
| C1-C4 | 0.989 | **0.978** | ✓ |
| C2-C3 | 0.553 | 0.501 | (excluded — known null) |
| C2-C4 | 1.000 | **0.980** | ✓ |
| C3-C4 | 0.952 | **0.926** | ✓ |

Count gate: **4/5** non-(C2-C3) pairs clear p16 ≥ 0.75 (PASS ≥ 3). C1-C3
just misses (0.741 vs 0.75) — flagged for honest framing but doesn't fail
the count.

Structural gate: NoFB-vs-feedback {C1-C2, C1-C3} contains a clear C1-C2
PASS at 0.842 p16 ✓; feedback-vs-feedback {C2-C4, C3-C4} both clear
(0.980, 0.926) ✓. The lattice produces a non-trivial pattern: WindAGN
and WindStrongAGN are both highly separable from C2 (StellarWind),
WindStrongAGN is separable from WindAGN, and only C2-C3
(StellarWind-vs-WindAGN) is near chance.

## §4 — Reframe 7 verdict

**FAIL.** No (cluster_set × physical_axis) cell clears the η² ≥ 0.20
"large effect" bar.

Class-averaged headline (six (cluster_set × axis) cells, in η² order):

| cluster_set | axis | η² | F | p |
| :-- | :-- | --: | --: | --: |
| wavelet_k5 | mean_flux | **0.1246** | 582.8 | 0 |
| raw_k5 | mean_flux | 0.0570 | 247.7 | 6.5e-207 |
| raw_k5 | line_count | 0.0067 | 27.5 | 9.4e-23 |
| wavelet_k5 | line_count | 0.0053 | 22.0 | 4.3e-18 |
| raw_k5 | integrated_tau | 0.0029 | 12.0 | 9.9e-10 |
| wavelet_k5 | integrated_tau | 0.0026 | 10.6 | 1.3e-08 |

Strongest signal: `wavelet_k5 × mean_flux` at η² = 0.125 — "medium
effect" by Cohen's bins (0.06 < η² < 0.14), well-separated from
chance (p ≈ 0) but **below the pre-committed 0.20 bar**. Per-class
breakouts (the labels are per-position, so a per-class axis is also
honest): the strongest single (class, axis) is `wavelet_k5 × mean_flux
× class=1` at η² = 0.131 — same axis, same magnitude. The clusters
**lean toward mean-flux ordering but do not strongly track it**; they
do **not** track ∫τ dv or line counts.

Per SCOPING §2 R7(b) FAIL routing: Reframe 7 records as a **second
independent confirmation of [D-13]'s representation-disagreement
reading** — the K=5 clusters do not encode a clean physical axis at
the large-effect level. Adjacent finding, no headline.

---

## §5 — Honesty hurdles (the [D-37] reads)

**R1 — ⟨F⟩ vs shape (the load-bearing hurdle).** The PASS bar clears
trivially: arithmetic on Stage 0's C4 recall (0.96) and rest recall
(≈ 0.4–0.6) lands a binary balanced_acc near 0.97, exactly as the PI
predicted in SCOPING §5. The decisive question — *whether this signal
is shape-encoded or ⟨F⟩-carry-over* — is **NOT settled by Reframe 1
alone**. The regime-parity check (per_sightline ≈ global within 0.003)
is consistent with both physical readings: a shape signal that survives
per-sightline normalization, OR an ⟨F⟩-encoded carry-over channel that
survives both normalizations through preserved shape-correlated
structure ([D-07] reading). **Reframe 9 is the only experiment that
disentangles these.**

**R5 — beyond "C4 is easy."** The lattice is non-trivial: C2-C3 is near
chance (0.55, as known from [D-26]), but {C1-C2, C1-C3} produce a
*partial* NoFB-vs-feedback distinguishability (0.880, 0.802 medians —
C1-C3 just under the structural bar at p16 = 0.741). Most strikingly,
{C2-C4, C3-C4} both clear cleanly, meaning **WindStrongAGN is
distinguishable from BOTH StellarWind and WindAGN at p16 ≥ 0.92**.
This **is** a structural pattern beyond "C4 is easy" — it sharpens to
"C4 is easy AND C2/C3 are partially distinguishable from C1 AND C2 is
indistinguishable from C3." But the same ⟨F⟩-carry-over hurdle from R1
applies to the lattice: the cell-by-cell numbers could be shape-encoded
or ⟨F⟩-rank-encoded; R5 alone cannot say which. R9 disentangles for
the C4-pair rows; the C2-C3 null is robust regardless.

**R7 — do clusters trivially track ⟨F⟩?** Partially. The strongest η²
sits on `mean_flux` (0.125 wavelet, 0.057 raw), an order of magnitude
above ∫τ and line counts. This **is** the predicted SCOPING §5 outcome
shape — clusters lean toward mean-flux ordering — but they don't track
it at the "large effect" threshold the pre-commit required. Honest
framing: *the K=5 wavelet clusters weakly encode mean-flux structure
(medium-effect, ~12.5% of variance explained); they do not encode a
clean thermal-state or line-count axis*. This is consistent with
[D-13]'s representation-disagreement reading rather than overturning
it.

---

## §6 — Phase-2 gate decision

Per SCOPING §3 Phase-1 → Phase-2 gate: gated on **Reframe 1 PASS**.

- Reframe 1: PASS (p16 = 0.96, Δ = +0.29 pp, regime parity ✓)
- ⇒ **Phase 2 (Reframe 9) is AUTHORIZED.**

The minor-compute Reframe-9 brief (~250 fits, ≤ 1h laptop wallclock per
SCOPING §3) is what resolves the R1 + R5 ⟨F⟩-vs-shape honesty hurdle.
**Without R9 the R1 headline remains rule-7-fragile**; the qualification
gate in SCOPING §4 binds the joint R1 + R9 PASS as the "solid/successful"
criterion for paper authoring (user-triggered-only — not recommended here).

---

## §7 — Limitations and what this report does NOT claim

- **Reframe 1's binary signal is not yet shape-attributed.** The
  ⟨F⟩-carry-over reading remains alive until R9 returns.
- **Reframe 5's lattice carries the same ⟨F⟩ caveat** on the C4-pair
  rows. The C2-C3 null and the partial C1-vs-feedback distinguishability
  are robust to the ⟨F⟩ question (they involve no C4).
- **Reframe 7's R7(a) PASS bar (η² ≥ 0.20) was missed.** The medium-effect
  mean-flux signal is informative but does not clear pre-commit.
- **K=5 per-position labels caveat.** Each cluster label indexes a
  spatial position shared across all 4 physics classes, not a
  single-class sightline. The class-averaged axis is the natural
  headline; per-class breakouts (in `reframe7_cluster_axes.csv` with
  `aggregation` = `class_1..class_4`) give the same qualitative
  ordering (mean_flux dominates) and the same FAIL verdict.
- **No CIs on Reframe 7 effect sizes** were computed — the brief
  specified F-stat + p-value + η² only; bootstrap CIs would tighten the
  read but the FAIL is comfortably below the bar (top 0.125 vs gate 0.20).
- **No headline-verb claim is made.** Per SCOPING §1 verb ceilings, R1
  remains hedged at "first quantified binary detection … gated on R9";
  R5 at "first quantified pairwise distinguishability lattice"; R7 at
  "first physical-axis interpretation."
