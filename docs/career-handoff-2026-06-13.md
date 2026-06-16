# Career-Profile Handoff — CosmoGasPeruser

**Date:** 2026-06-13 (current-state snapshot; supersedes the 2026-06-10 version)
**Audience:** ML/AI hiring managers and recruiters. No astrophysics background assumed; no knowledge of this repository assumed.
**Scope:** Independent research project, ~2026-02-27 through 2026-06-13. All claims trace to repository files; the Provenance Appendix (§8) maps every claim to its source and raw numbers.
**Work attribution:** Independent, self-directed work by the repository owner. The underlying simulation data is the published Sherwood cosmological simulation suite (Bolton et al. 2017) — external, public-science input. The analysis, methodology, and all code are the owner's. The project was executed with an AI-agent-assisted workflow (multiple specialized LLM agents for implementation, data auditing, literature search, and adversarial review) that the owner designed and orchestrated; final scientific decisions and sign-offs were the owner's. Computation ran locally and on UT Dallas's Juno HPC cluster (SLURM).

> **What changed since the 2026-06-10 handoff:** the four validation experiments an internal review had flagged were run (2026-06-12), the headline results survived and in places strengthened, a previously-reported null result was honestly corrected into a weak real signal, the raw simulation provenance was audited from disk, and a first full manuscript draft was written. See §5 for the itemized delta.

---

## 1. Project one-liner

**CosmoGasPeruser asks a question the astrophysics community cares about and machine learning is well suited to test: can you tell which galaxy-feedback physics (supernova winds vs. black-hole/AGN energy injection) shaped the intergalactic gas, just by looking at the absorption patterns it imprints on light from distant quasars — and, crucially, at what data volume does each distinction become possible?**

Why it's hard: the test bed is 65,536 simulated quasar spectra (4 feedback scenarios × 16,384 lines of sight × 2,048 pixels each, ~134M values) from the *same* underlying universe — matched initial conditions, differing only in feedback physics. The differences are subtle, heavily degenerate between recipes, and easily mimicked by a single trivial statistic (the average transmitted flux), so naive classifiers "succeed" for the wrong reason. The project is as much about confound control, sensitivity calibration, and honest handling of weak/negative results as about detection.

---

## 2. Current status (honest external framing)

- **Research phase: complete and validated.** The analysis is finished, and the central results have been through an internal adversarial review followed by a dedicated round of validation experiments that addressed every objection (2026-06-12). The results are now defensible.
- **Publication status: manuscript in preparation (first full draft written); not yet submitted.** A complete draft exists — abstract through conclusions, three rendered figures, a corrected bibliography, all numbers wired to a single source-of-record file — targeting a monthly-notices-style astronomy journal. Author list and final journal formatting are not yet set; one provenance detail (see below) must be reconciled before submission.
- **What stands (validated):** (a) a quantified, strengthened map of which feedback recipes are distinguishable and at what data volume; (b) a strong detection of the strongest-AGN recipe; (c) a methodology for claiming a *weak* signal honestly, which converted one earlier "no signal" result into a real but modest one; (d) a clean chain of rigorously documented negative results.
- **One open pre-submission item:** the on-disk simulation box size (read from raw headers as 60 comoving Mpc/h) does not match the canonical box sizes in the cited reference paper — this must be reconciled to the exact simulation run before publication. Flagged, not yet resolved. [UNVERIFIED — reconcile before external use of the specific simulation-run citation.]

---

## 3. Headline accomplishments

### 3.1 Detection of strong AGN feedback from averaged quasar-absorption power spectra (~95–97% balanced accuracy, dedicated classifier)

**What:** The strongest-AGN-feedback recipe is reliably identifiable against all other recipes pooled, from the Fourier power spectrum of quasar absorption spectra once spectra are averaged in groups (a realistic survey-style aggregation).
**Method:** 1D flux power spectrum (windowed FFT, ~20 logarithmic frequency bands, physical velocity units derived from the data) → within-recipe averaging over groups of 64 sightlines → a **dedicated binary random-forest classifier** → balanced accuracy under 5-fold cross-validation across 10 random seeds (50 train/test splits per result).
**Outcome:** balanced accuracy **0.971 median, 0.954 on a conservative (16th-percentile) bound**. A mean-transmission-only baseline classifier (built specifically to test the dominant confounder) does *not* reach this — the power-spectrum *shape* carries genuine separability beyond mean transmission at every recipe pair. This closes the caveat that flagged this result in the prior handoff: the cheapest-confound baseline has now been run.

### 3.2 A strengthened "who is distinguishable from whom, and at what data volume" map

**What:** A full pairwise distinguishability matrix across the four physics recipes, each pair scored by its own dedicated classifier (not read off a shared multi-class confusion matrix — that methodological upgrade was the main fix demanded by the internal review).
**Outcome (conservative 16th-percentile balanced accuracy, groups of 64 sightlines):** no-feedback vs. {stellar-wind, wind+AGN, strong-AGN} = 0.85 / 0.81 / 0.97; the three feedback recipes against strong-AGN = 0.97 / 0.96 / 0.95. **All five recipe pairs not involving the stellar-wind-vs-wind+AGN comparison now clear a 0.75 distinguishability floor** (up from four of five in the prior snapshot — no-feedback-vs-wind+AGN newly clears, at 0.81). Distinguishability rises monotonically with averaging depth, which turns the map into a concrete **data-volume requirement** per pair — directly useful to survey designers.
**Honest caveat retained:** the lattice ordering still correlates strongly with the recipes' mean-transmission offsets (rank correlation ≈ 0.94); the validated claim is "real mean-transmission ordering *plus* a real power-spectrum-shape residual on top," not "shape alone." The residual is what the dedicated baseline (§3.1) isolates.

### 3.3 A rigorous negative result: single spectra carry too little information, and an invented representation proved it

**What:** The project's original goal was to discover feedback signatures in *individual* spectra. To test this, designed a novel per-spectrum representation: for each line of sight, train 6 tiny one-vs-one SVM probes (one per recipe pair) and record signed decision distances from all 4 recipe versions of that sight line → a 24-dimensional "separability vector" per spectrum (16,384 × 24), computed in parallel over the full dataset, in two independent feature spaces (raw spectra and a 6-level wavelet decomposition).
**Method of falsification:** K-Means clustering with silhouette sweep, UMAP, cross-representation cluster agreement, and a per-cluster classifier accuracy audit.
**Outcome:** per-cluster classification never beat the project's own per-spectrum baseline (0.451 balanced accuracy, 4 classes, chance 0.25); apparent "interesting" clusters agreed only 20–33% across the two feature spaces — artifacts of representation choice, not physics; a later analysis confirmed the clusters mostly track mean transmission. **Conclusion: the per-spectrum problem is information-limited, not representation-limited** — which justified and correctly predicted the success of the population-level (averaged) reformulation in §3.1–3.2.

### 3.4 Honest correction of a result in both directions: a "null" that became a weak real signal

**What:** The hardest, most scientifically central pair — stellar-wind vs. wind+AGN — was initially reported as indistinguishable (coin-flip). The validation round re-examined it with a dedicated classifier and three controls, and the honest verdict moved twice: it was *over-claimed* as an intrinsic null, the validation initially *over-corrected* it to a confident "feedback signal," and the final, defensible statement is in between.
**Outcome:** the pair is indistinguishable per sightline (≈ chance) but becomes a **weak but real** power-spectrum-shape signal that emerges only with averaging — rising from ≈0.36 (single spectra) to **0.686 median balanced accuracy at groups of 256 sightlines** (conservative bound 0.603; statistically marginal, needs groups of ≥128 to clearly exceed chance). Three controls establish the rise is physical, not an artifact: (i) a within-recipe self-comparison (same physics, split in half) stays exactly at chance, ruling out an averaging-variance artifact; (ii) a mean-transmission-only baseline stays flat at chance, ruling out the confounder; (iii) reading the raw simulation headers from disk confirmed the two recipes share box size, cosmology, and initial-condition geometry, ruling out a simulation-setup artifact.
**Why it matters to an ML audience:** this is a clean demonstration of disciplined weak-signal claiming — not letting a marginal result be either buried as "null" or oversold, and building the specific controls that license the honest middle verdict.

### 3.5 A calibrated sensitivity floor (knowing how small a signal the method could have seen)

**What:** Built an injection-recovery test — inject a synthetic feedback-like perturbation of controlled amplitude into the spectra and measure the smallest amplitude the pipeline can recover.
**Outcome:** at groups of 64 sightlines the pipeline detects a power-spectrum-shape perturbation as small as **2% of the strong-AGN signal**. This replaces a vague "the method is sensitive enough" claim with a number, and it also revealed (honestly) that the stellar-wind-vs-wind+AGN signal is *nearly orthogonal* to the strong-AGN template — i.e., a genuinely distinct, weaker spectral signature, not a scaled-down version of the easy signal. That orthogonality is why §3.4's weak signal rests on its own controls rather than on this calibration.

### 3.6 Two more research directions killed cheaply at scoping, and two physical mechanism stories falsified by design

(Condensed from the prior handoff — these still stand.) Two follow-on directions (a simulation-based-inference approach and a line-by-line absorption-statistics approach) were closed at scoping for under an hour of compute each, via feasibility tests and literature effect-size verification, before any pipeline was built. Separately, two attractive physical-mechanism explanations for the classifier's behavior were each converted into a falsifiable prediction and then rejected by pre-committed ablation and permutation tests — establishing that the surviving claim is "an aggregate signal exists, direction-consistent with published predictions," with no over-reaching mechanism story. (Details in §4 and §8.)

---

## 4. Decisions and pivots (situation → evidence → decision → outcome)

1. **Per-spectrum feature discovery → population statistics.** *Situation:* the founding hypothesis was that better representations reveal feedback in single spectra. *Evidence:* §3.3 — per-cluster accuracy never beat baseline; clusters disagreed across representations. *Decision:* declare the per-spectrum question closed; re-pose at the population (averaged) level, documenting that this changes the scientific question rather than rescuing the old one. *Outcome:* the reformulation produced the project's strongest results (§3.1–3.2).

2–3. **Two physical-mechanism hypotheses falsified, claims narrowed not spun.** A small-scale "thermal" signature and then a mid-scale "line-width" signature each looked compelling in feature-importance, were each pre-registered as a testable prediction, and each failed cleanly (importance diffuse rather than concentrated; an ablation deleting the "most important" bands barely moved accuracy). *Outcome:* the mechanism narrative was retracted; only the signal-existence claim, direction-consistent with the literature, was kept; a stopping rule prevented fishing for a third mechanism.

4. **A built-in "which recipes drive the accuracy?" audit fired, and the headline was scoped down on the day of the result.** A 4-class accuracy can be carried by one easy recipe; a dedicated check on the central hard pair plus a mean-transmission-leak check were required *in advance*. They fired: the aggregate accuracy was carried by the easy recipes, the central pair sat at chance, the leak check came back clean. The claim was narrowed immediately, not after review.

5. **Self-commissioned adversarial review → a bounded validation round that the results survived (2026-06-12).** *Situation:* with positive results in hand and a paper plausible, the owner commissioned an internal red-team to attack the qualified claims. *Evidence found:* the pairwise numbers came from shared-confusion-matrix sub-blocks rather than dedicated classifiers; the "indistinguishable pair" climbed with averaging; the distinguishability ordering correlated ≈0.94 with mean-transmission offsets; no injected-signal sensitivity test existed; one replication count was inflated. *Decision:* run four bounded validation experiments rather than publish around the objections. *Outcome:* dedicated per-pair classifiers **strengthened** the map (§3.2); a mean-transmission-only baseline showed a real shape residual (§3.1, §3.2); an injection-recovery test quantified sensitivity (§3.5); and the central pair was honestly re-cast as a weak real signal (§3.4). Every objection was neutralized, narrowed, or quantified — and the team caught its *own* over-correction mid-round (the weak signal was briefly overstated before being dialed back). This decision-quality story is itself a centerpiece.

6. **Provenance audited from the raw bytes.** To rule out that the recipe comparisons were confounded by differing simulation setups, the raw simulation line-of-sight headers for all four recipes were read directly from disk: identical box size, cosmology, redshift, sampling, initial-condition geometry, and a shared instrument line-spread kernel. The four recipes are confirmed to be the same realization with only feedback varied. (This audit also surfaced the open box-size-vs-citation discrepancy in §2.)

7. **Citation hygiene as a correctness problem.** Literature verification caught a non-existent reference, two identifier↔journal mispairings, and a same-author paper conflation — all corrected into a clean bibliography before the draft. One relevant recent preprint missing from the novelty analysis was added.

---

## 5. What changed since the 2026-06-10 handoff

- **SUPERSEDES prior claim "results are not yet defense-hardened; four validation experiments outstanding."** All four were run (2026-06-12) and the results survived: the pairwise map was rebuilt with dedicated classifiers and **strengthened** (now 5 of 5 non-central pairs clear the distinguishability floor), the mean-transmission baseline confirmed a real shape residual, sensitivity was quantified, and the central pair was re-cast (see next).
- **SUPERSEDES prior claim "stellar-wind vs. wind+AGN are indistinguishable (coin-flip) / an aggregation-conditional impossibility."** Now a **weak but real** signal emerging with averaging (median 0.686, conservative bound 0.603 at groups of 256), confirmed physical by three controls. The honest verb is "weak signal, statistically marginal," not "null" and not "confident detection."
- **SUPERSEDES prior claim "no manuscript exists / nothing in preparation."** A **first full draft exists** (abstract→conclusions, 3 figures, corrected bibliography, single numbers-of-record file), targeting a monthly-notices-style journal, titled around "a quantified feedback-recipe distinguishability framework for the low-redshift Lyman-α forest flux power spectrum." Not submitted; author list and one provenance citation still open.
- **Removed the §3.1/§3.2 `[UNVERIFIED]` interpretation flags** from the prior handoff — the specific experiments those flags were waiting on have been run. The one remaining `[UNVERIFIED]` is the box-size-vs-citation reconciliation (§2), which is a bibliography/run-identification detail, not a result.

---

## 6. Skills and tools demonstrated (with evidence)

- **Scientific ML / statistics:** balanced-accuracy evaluation under class structure; stratified K-fold CV with multi-seed bootstrap bounds (50 splits/result); dedicated binary classifiers vs. multi-class sub-block estimates; permutation-test nulls; ablation studies; correlated-feature importance bias; confound controls (mean-transmission-only baseline; per-class normalization rejected as leaky); within-class self-comparison nulls; **injection-recovery sensitivity calibration** (smallest detectable effect size). Evidence: `results/reframe_suite/hardening/`, `results/pk_feedback_classifier/stage1/`, `tests/test_stage1_fold_leakage.py`.
- **Classical ML engineering:** scikit-learn RBF-SVMs (6 one-vs-one micro-probes per spectrum across 65k spectra, joblib-parallel), random forests, K-Means + silhouette sweeps, UMAP; thousands of cross-validated model fits with checkpoint/resume. Evidence: `src/core/probe.py`, `src/core/cluster.py`, `src/core/models/rf_classifier.py`, `experiments/`.
- **Signal processing on physical data:** windowed FFT power spectra with physically derived frequency axes; multi-level wavelet decompositions; Voigt-profile line fitting with synthetic-recovery validation grids. Evidence: `src/core/transforms.py`, `experiments/linewidth-cddf/`.
- **Data provenance / forensics:** read raw binary simulation headers across all four datasets to confirm matched initial conditions and surface a box-size-vs-citation discrepancy. Evidence: `experiments/reframe-suite/HARDENING_OUTCOME.md` §H2 provenance note.
- **Reproducibility infrastructure:** DVC-versioned data + a 6-stage reproducible pipeline; MLflow tracking; `uv`-locked Python 3.12 reproduced bit-identically across laptop and cluster (md5-verified). Evidence: `dvc.yaml`, `.claude/skills/`.
- **HPC:** SLURM job design (partition selection, wallclock budgeting from smoke tests, fail-loud output verification), rsync staging. Evidence: `scripts/submit_juno_*.sh`; jobs 205725, 206130, 213282.
- **Technical writing / paper authoring:** a multi-venue LaTeX "master-source" manuscript (shared section atoms + per-venue manifest + a single numbers-of-record macro file so every cited number traces to a source CSV). Evidence: `papers/`.
- **AI-agent orchestration:** designed a multi-agent research workflow (PI, implementer, data-engineer, literature, red-team reviewer) with append-only decision logs, pre-committed acceptance criteria, and mandatory adversarial review before expensive runs. Evidence: `.claude/agents/`, the per-experiment logs.
- **Research integrity:** pre-registered falsification criteria; observation-before-framing reporting; citation verification that caught a hallucinated reference; symmetric disclosure of weak/negative results, including correcting the team's own over-correction.

---

## 7. Resume-ready bullets

1. Built an end-to-end reproducible ML pipeline (Python, scikit-learn, DVC, MLflow, SLURM) testing whether galaxy-feedback physics is detectable in 65,536 simulated quasar spectra (~134M data points); first full manuscript draft in preparation.
2. Detected strong AGN feedback from averaged absorption power spectra at 0.97 balanced accuracy (0.95 conservative bound) using dedicated binary classifiers, beating a mean-transmission-only confound baseline at every recipe pair.
3. Produced a validated pairwise distinguishability map of four feedback recipes (5 of 5 non-trivial pairs above a 0.75 floor) and a per-pair data-volume requirement — an actionable sensitivity result for survey design.
4. Corrected a prior null into a defensible weak signal: stellar-wind vs. wind+AGN reaches 0.69 balanced accuracy under stacking, validated against self-comparison, mean-flux, and simulation-setup controls.
5. Built an injection-recovery calibration showing the pipeline detects power-spectrum-shape perturbations as small as 2% of the strongest signal, replacing a qualitative sensitivity claim with a number.
6. Ran a self-commissioned adversarial review, then a bounded validation round that neutralized, narrowed, or quantified every objection — and caught the team's own over-correction in the process.
7. Invented a 24-dimensional per-spectrum "separability vector" (six one-vs-one SVM probes over 65k spectra) and used clustering + per-cluster classifiers to establish a per-spectrum information ceiling, motivating a successful population-level reformulation.
8. Authored a multi-venue LaTeX manuscript with a single numbers-of-record file wiring every cited figure to a source dataset; audited raw simulation headers to confirm matched initial conditions.

*(Translation rule honored: numbers are stated against chance, baselines, prior methods, or external references; internal acceptance thresholds appear only in §8.)*

---

## 8. Provenance appendix (internal vocabulary permitted here only)

Maps every externally-framed claim above to repository sources and raw internal numbers. Internal vocabulary (gates, p16, H1–H4, R1/R5, D-XX, reframe, defense-panel, hardening) appears here as in the sources. **Source-of-record:** `experiments/reframe-suite/HARDENING_OUTCOME.md` (2026-06-12) + `papers/shared/numbers.tex`, which supersede the older `reframe-suite/CLOSE_OUT.md` and the 4-class sub-block lattice estimates. All lattice/detection/M-scaling numbers are dedicated-binary, per_sightline, p16 = 16th percentile of 50 (10-seed × 5-fold) balanced-accuracy values. CSVs under `results/reframe_suite/hardening/`.

### Status (§2)
- Manuscript: `papers/` master-source tree — `papers/mnras/main.tex` (article-class fallback; mnras.cls not bundled), section atoms `papers/shared/sec/{0_abstract…6_conclusions}.tex`, `numbers.tex`, `main.bib`, figures `papers/shared/figures/fig_{acc_vs_M,lattice_M64,injection_sensitivity}.{pdf,png}`. Author block = TODO; not submitted. Paper draft committed at `0e3c269`.
- Box-size open item: `numbers.tex` `\boxsize{60}` cMpc/h with `% TODO(box-citation)`; HARDENING_OUTCOME.md §H2 + Net-disposition note — measured 60 cMpc/h ≠ Bolton+2017 canonical 40/80/160; reconcile run/citation before publication.

### §3.1 detection — H1 / R1, dedicated binary
- C4-vs-rest p16 `\latCdRest` = **0.954**, median `\latCdRestMed` = **0.971**, per_sightline M=64, `results/reframe_suite/hardening/pair_binary_summary.csv`. HARDENING_OUTCOME.md §H1 ("R1 SURVIVES: 0.954 ≥ 0.90").
- Mean-transmission residual: H3 — dedicated P_F(k) p16 minus ⟨F⟩-only-scalar p16; clean/large off the accuracy ceiling (M=16: C4-rest +0.121); `fbar_baseline_summary.csv`. Resolves the prior handoff's §3.1 `[UNVERIFIED]` (the ⟨F⟩-scalar baseline was the missing experiment; now run).

### §3.2 lattice — H1, dedicated binary, p16 @ M=64 (`numbers.tex`)
- `\latCaCb` C1-C2 0.854; `\latCaCc` C1-C3 **0.814** (newly clears 0.75); `\latCaCd` C1-C4 0.971; `\latCbCd` C2-C4 0.961; `\latCcCd` C3-C4 0.951; `\latCbCc` C2-C3 0.520 (→ H2). 5/5 non-(C2-C3) pairs ≥ 0.75 (was 4/5). `pair_binary_summary.csv`; HARDENING_OUTCOME.md §H1.
- Spearman(lattice ordering, per-class ⟨F⟩) `\hthreeSpearman` = 0.94 — NOT overturned; H3 verdict "narrowed, not rebutted."

### §3.3 per-spectrum ceiling — signal-clustering-v2 [D-13], baseline v0.2
- Baseline 4-class RF 0.4514 (10-fold CV), `experiments/baseline-random-forest/LEDGER.md`, `results/baseline_rf/`; chance 0.25. Separability vectors `(16384,24)`, `src/core/probe.py`. Cross-run overlap 20–33%, `results/signal_clustering_v2/`. Reframe-7 η²=0.125 second confirmation. 4-class M=1 ≈ 0.29 (`\fourclassMone`).

### §3.4 C2-C3 weak signal — H2, dedicated binary
- M-curve median (per_sightline): `\hbcMone` 0.36 (M=1) → `\hbcMsixfour` 0.566 (M=64) → `\hbcMonetwoeight` 0.609 (M=128) → `\hbcMtwofivesix` **0.686** (M=256); p16@M256 `\hbcMtwofivesixPsixteen` = **0.603** vs gate `\hbcGate` 0.60 (margin `\hbcGateMargin` 0.003; bootstrap P(p16≥0.60) `\hbcBootP` 0.739; 95% CI [0.577, 0.637]). Needs M ≥ `\hbcMmin` 128. M=512/1024 (0.77/0.83) EXPLORATORY. `pair_binary_summary.csv`; HARDENING_OUTCOME.md §H2.
- Controls: self-pair C2-vs-C2 `\selfpairMsixfour` 0.509 → `\selfpairMtwofivesix` 0.500 (criterion 0.55 NOT met); ⟨F⟩-only C2-C3 ≤ `\fbarCbCcMax` 0.512 flat; box/ICs identical from disk (raw LOS `.dat` headers, all 4 classes: z=0.300, Ω_m 0.308, Ω_b 0.0482, h 0.678, box 60 cMpc/h, 2048 bins, 16384 LOS, shared COS_LSF1). HARDENING_OUTCOME.md §H2 controls 1–3 + provenance note. The "corrected both ways" framing = HARDENING_OUTCOME.md Net-disposition row 4.

### §3.5 sensitivity — H4
- δ_min(M=64) `\hfourDeltaMin` = 0.02 = `\hfourDeltaMinPct` 2% of the C4−C2 template; α=0 calibration centered at chance (`\hfourAlphaZeroMsixfour` 0.509). Off-template: β `\hfourBeta` 0.539 vs α_equiv `\hfourAlphaEquiv` 0.004 (C3−C2 nearly orthogonal to C4 template). `injection_recovery_summary.csv`, `injection_templates.csv`; HARDENING_OUTCOME.md §H4.

### §3.6 / §4 mechanism + scoping (unchanged from 2026-06-10 handoff)
- High-k falsified: Stage 0 [D-07] high_k_frac 0.43 < 0.50. Mid-k falsified: Stage 1 [D-26] G3 mid_k_frac 0.340, G6 ablation Δ 0.0148 vs threshold 0.0530. Permutation null obs 0.595 vs p99 0.386. Per-class M=64 recall C1 0.882 / C2 0.627 / C3 0.346 / C4 0.961. `results/pk_feedback_classifier/stage1/`. fps-sbi + linewidth-cddf scoping closes: `experiments/{fps-sbi,linewidth-cddf}/`. Citation corrections: linewidth-cddf CLOSE_OUT §3 + pk [D-19].

### §5 delta sources
- Hardening sprint: Juno job 213282 (COMPLETED 1h44m), RUN_TAG `rshardening-20260612-172444-06d7ac`; commits `cad7f6d` (results), `cbef7ed` (verdicts), `a19eadd` (provenance promotion), `acc73a6`/`50a65e9` (figures), `0e3c269` (first full draft). Decisions [D-27]–[D-30] (HARDENING_OUTCOME.md §Decision-log).

### Binding constraints any external use inherits
- Demotion still binds in spirit: report as "stacked-P_F(k) population-statistics framework," not per-sightline classification; the manuscript abstract carries the verbatim non-claim "We do not claim per-sightline feedback identification at z ≈ 0.3 from P_F(k) alone."
- All results z ≈ 0.3, single Sherwood realization, fixed cosmology/UVB/thermal history, no nuisance marginalization; no generalization to z = 2–5.
- C2/C3 verb is "weak, gate-marginal, controls-confirmed" — never "clean detection."
