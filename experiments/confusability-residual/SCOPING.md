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
SCAFFOLDED — not yet run. Next step (user-gated): one worker pass —
`core-implementer` (predict_proba + confusability score), `data-engineer`
(per-sightline covariate join), `support-researcher` (residual/partial-correlation
test + the single figure). Owner: PI-coordinated.
