# SCOPING — confusability-residual (thin exploratory probe)

> **This does NOT modify the reframe-suite numbers of record or the park-state
> disposition** (`experiments/reframe-suite/HARDENING_OUTCOME.md`: H1 HARDENS,
> H2 weak-signal, H3 NARROWED, H4 quantified). It is a no-new-compute,
> no-new-prior, NON-paper exploratory probe. Paper-trigger remains user-owned.

Origin: user side-question (2026-06-20) — "can a classifier reroute predictions
from confusion-matrix analysis (binary flip → 4-class)? would multi-label output
sort out the most confusing sightlines?" PI relevance ruling: see below.

## Verdict (PI, 2026-06-20)
- **Part 1 — post-hoc rerouting (Hungarian-permutation / Bayes column-argmax): DECLINED.**
  Our Episode-4 RF confusion structure is a **sink collapse**, not label
  misalignment (RF_Raw: true-2 → 86% pred-1, true-3 → 78% pred-1; Classes 2/3
  recall 0.04/0.05; Class-4 recall 0.88/0.94/0.63). Rerouting recovers
  *misalignment* only; it cannot manufacture separability the features don't
  carry. The only salvage ("merge {1,2,3}, do C4-vs-rest") is the **already-
  qualified** reframe-suite R1 binary detector (CLOSE_OUT §1, p16 0.960).
  → Do NOT implement rerouting code, not even "to show it fails."
- **Part 2 — set-valued confusability: confirmation-only EXCEPT one slice.**
  Conformal sets / entropy that return "{1,2,3} confuse, {4} separates" merely
  re-derive the confusion matrix. The single genuinely un-asked question is the
  **residual** one below.

## The ONE core question (entire scope)
> After regressing out per-sightline **mean flux** AND **line density**, does RF
> `predict_proba` confusability retain structure that correlates with any third
> physical quantity at |Spearman| ≥ 0.20?

Not "does confusability correlate with physics" (that is the tautology — mean
flux is *the* C4 separator; line density is the eda-identified real separator, so
raw confusability will re-derive the H3 Spearman-0.94 mean-flux confound). Only
the **orthogonal residual** can clear the "adds information beyond what we know"
bar.

## Pre-committed criteria (rule-5 symmetric)
- **PASS (informative):** residual confusability shows |Spearman| ≥ 0.20 with an
  identifiable third quantity that is not a re-encoding of the two removed
  covariates → record as an exploratory D-XX finding (still NOT a paper trigger).
- **NULL (expected, valid end state):** residual ≤ 0.20 against all tested
  quantities → "confusability re-derives the known mean-flux / line-density
  confound; no orthogonal structure." Report plainly (rule-7); do not spin.
- **HARD KILL:** if the analysis only reproduces "{1,2,3} confuse, {4} separates"
  with no residual test run → duplicate of the confusion matrix; abandon, no D-XX.

## Reuse (no new compute, no new data, no new prior)
- RF: re-fit mirrors the recorded baseline hyperparameters (n_estimators=100,
  max_depth=25, max_features=sqrt, min_samples_leaf=50, min_samples_split=100,
  seed 42) — same fit as `export_episode4_rf_confusion_matrices`
  (on `service/data-export`, the cross-check anchor); add a `predict_proba` read.
  `src/core/models/rf_classifier.py` is the hook (`predict_proba` not yet used in
  the RF path).
- Per-sightline covariates (mean flux, line density): eda Tier-1 extractors
  (`scripts/eda_sherwood.py`) + `results/eda/`.
- Confusability score: entropy / margin of the 4-vector; split conformal sets are
  a **descriptive intermediate only** (one diagnostic figure cap).

## Anti-scope-creep boundary (binding)
- NO new simulation data; NO new feature extraction beyond Tier-1 + the existing RF.
- NO rerouting (Hungarian/Bayes) code — Part 1 is declined by argument.
- NO conformal-prediction headline; conformal sets capped at one diagnostic figure.
- NO touching the parked manuscript state or reframe-suite numbers of record.
- **ONE script, ONE figure, ONE D-XX entry.** If it grows a §1 Pulse table or a
  second figure family, scope has breached — stop and re-scope with the PI.

## Guardrails (binding, from the PI ruling)
1. Does not reopen the parked verdict, under any outcome.
2. NOT a paper trigger under any outcome (PASS = adjacent exploratory finding;
   rule-14(iii) does NOT license routing toward publication). Worker agents must
   not recommend "write this up."
3. [D-37] honest framing both directions — the **expected** outcome is the NULL
   (confusability = the mean-flux/line-density confound); report it as the
   first-pass observation, not buried under conformal-set machinery.
4. rule-14 fragility: set-valued outputs + author-defined score are structurally
   fragile → demoted to NO-headline-claim **by construction**, pre-committed here.
5. PI sign-off on this scope is PROVISIONAL (rule-15); any worker dispatch citing
   it inherits PROVISIONAL status. Promotion to a finding needs a fresh review.

## Status
**RUN & CLOSED — NULL (2026-06-20).** Single pass executed
(`scripts/probe_confusability_residual.py`); PI close-out sign-off given
(provisional status lifted via in-session re-verification). Outcome below.

## Outcome — D-01 (NULL, the pre-committed expected end state)

**Verdict: NULL.** After regressing out per-sightline mean flux AND line density,
RF `predict_proba` confusability retains **no** orthogonal residual structure above
the pre-committed |Spearman| ≥ 0.20 bar against any non-re-encoding third quantity.
Confusability re-derives the known mean-flux / line-density confound. **NOT a paper
trigger** (guardrail 2); does **not** reopen the parked reframe-suite verdict
(guardrail 1).

What was run (one script, one figure — boundary respected):
- **RF anchor (faithful).** Reproduced the recorded RF_Raw fit — the cross-check
  anchor `export_episode4_rf_confusion_matrices` (on `service/data-export`): raw
  flux, single 80/20 stratified holdout, seed 42, recorded HP. Holdout accuracy
  **0.4509 vs recorded 0.4514 (Δ=−0.0005)** → the fit is the correct anchor.
  `predict_proba` read on the 13,108 held-out sightlines.
- **Confusability** = Shannon entropy + top-two margin of the 4-vector.
- **Covariates** recomputed per sightline with the eda Tier-1 extractor. NOTE: the
  `results/eda/` CSVs are CLASS-level aggregates, so the SAME Tier-1 method was
  re-run per sightline (reuse of method, not new feature extraction — in scope).
  `total_ew` (≈ linear in mean_flux; raw ρ=−1.00) and `gap_mean` (≈ 1/line_density;
  raw ρ=−0.99) were excluded from PASS as confound re-encodings, reported as
  diagnostics.
- **Partial Spearman** of confusability vs each pre-registered candidate
  (depth_mean/std/max, ew_mean/std, gap_std), controlling for [mean_flux,
  line_density].

Honest deviation narrative ([D-37], recorded in execution order):
1. The **first** run used a **linear** rank-control and produced a borderline
   PASS: `gap_std` at ρ=−0.2589 — but **only under the margin score**; the entropy
   score gave −0.1667 (< 0.20). The pre-committed **rule-4 fragility** gate
   (must clear under *both* author-defined scores) therefore demoted it to
   **PASS-FRAGILE → no-headline** independently of anything below.
2. The test was then **hardened against my own borderline-positive first result**:
   the control basis was extended to [mf, ld, mf², ld², mf·ld]. Rationale —
   `gap_std` is a dispersion statistic of a quantity (gap_mean ~ 1/line_density),
   so its dependence on line_density is expected to be curved; the faithful reading
   of "regress out mean flux AND line density" is to strip their **full** (incl.
   quadratic/interaction) association, not just the linear part. Under this control
   `gap_std` drops to ρ=−0.1951 (margin) / −0.1026 (entropy) — both < 0.20.
3. **Verdict is invariant to the linear-vs-nonlinear choice**: even granting the
   lenient linear value (0.259), rule-4 fragility already demotes the candidate
   (entropy 0.167 < 0.20). The nonlinear hardening **corroborates** the NULL; it
   does not create it. Both control columns are reported transparently.

Artifacts (under `results/confusability_residual/`): `verdict.txt`,
`partial_correlations.csv` (linear + nonlinear columns, both scores),
`per_sightline_analysis.csv` (audit dump of the single analysis frame),
`figs/residual_partial_spearman.png` (the one figure).

No successor work authorized: no paper-author dispatch, no reframe-suite touch,
no §1 Pulse, no rerouting code (Part 1 remains declined-by-argument).
