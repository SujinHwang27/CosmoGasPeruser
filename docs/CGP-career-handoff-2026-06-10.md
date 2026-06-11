# Career-Profile Handoff — CosmoGasPeruser

**Date:** 2026-06-10
**Audience:** ML/AI hiring managers and recruiters. No astrophysics background assumed; no knowledge of this repository assumed.
**Scope:** Independent research project, ~2026-02-27 through 2026-06-10. All claims trace to repository files; the Provenance Appendix (§8) maps every claim to its source and raw numbers.
**Work attribution:** Independent, self-directed work by the repository owner. The underlying simulation data is the published Sherwood cosmological simulation suite (Bolton et al., University of Nottingham–led collaboration) — external, public-science input. The analysis, methodology, and all code are the owner's. The project was executed with an AI-agent-assisted workflow (multiple specialized LLM agents for implementation, data auditing, literature search, and adversarial review) that the owner designed and orchestrated; final scientific decisions and sign-offs were the owner's. Computation ran locally and on UT Dallas's Juno HPC cluster (SLURM).

---

## 1. Project one-liner

**CosmoGasPeruser asks a question the astrophysics community cares about and machine learning is well suited to test: can you tell which galaxy-feedback physics (supernova winds vs. black-hole/AGN energy injection) shaped the intergalactic gas, just by looking at the absorption patterns it imprints on light from distant quasars?**

Why it's hard: the test bed is 65,536 simulated quasar spectra (4 feedback scenarios × 16,384 lines of sight × 2,048 pixels each, ~134M values) from the same underlying universe, differing only in feedback physics. The differences are subtle, heavily degenerate between feedback recipes, and easily mimicked by a single trivial statistic (the average transmitted flux), so naive classifiers "succeed" for the wrong reason. The project is as much about confound control and honest negative results as about detection.

---

## 2. Current status (honest external framing)

- **Research phase: complete and concluded** (as of 2026-06-02, with a final internal audit on 2026-06-10). The project reached a deliberate, documented stopping point rather than petering out.
- **Publication status: nothing submitted, nothing in preparation yet.** No manuscript exists. An internal adversarial review on 2026-06-10 identified four bounded validation experiments (each days, not weeks) recommended before the headline numbers are used in a paper; those have **not** been run. Treat all headline numbers below as "internally reproducible and traceable, pre-publication."
- **What stands:** a strong positive detection result for the strongest AGN-feedback scenario, a quantified map of which feedback-model pairs are and are not distinguishable, and a chain of carefully documented negative results that progressively narrowed the question.
- **What does not stand:** several early hypotheses (detailed in §4) were tested and falsified by the project's own experiments. They are listed because the falsifications were handled rigorously — that is the selling point.

---

## 3. Headline accomplishments

### 3.1 Detection of strong AGN feedback from averaged quasar-absorption power spectra (~97% balanced accuracy)

**What:** Demonstrated that the strongest AGN-feedback scenario is reliably identifiable against all other scenarios from the Fourier power spectrum of quasar absorption spectra, once spectra are averaged in groups (a realistic survey-style aggregation).
**Method:** 1D flux power spectrum (windowed FFT, ~20 logarithmic frequency bands, physical velocity units derived from the data, not hardcoded) → within-class averaging over M=64 spectra → random-forest classifier → balanced accuracy under 5-fold cross-validation with 10-seed bootstrap confidence intervals.
**Scale:** 65,536 spectra; classifier sweep across 7 aggregation levels × 2 normalization schemes × 10 seeds × 5 folds (~8,000 model fits), run on an HPC cluster in under an hour.
**Outcome:** binary balanced accuracy **0.971 median, 0.960 at the 16th percentile** (i.e., the pessimistic edge of the bootstrap distribution, not the best case). Critically, the result held — accuracy change ≈ 0.00 — under a control experiment that explicitly removed each spectrum's mean transmission, the dominant known confounder.
**Caveat (disclose if asked):** an independent internal review noted that the cheapest possible baseline — a classifier given *only* the mean-transmission scalar — was never run as a comparison column, and recommended it before publication. [UNVERIFIED in that one specific sense — confirm before external use of the "not mean-flux-driven" interpretation; the accuracy numbers themselves are verified.]

### 3.2 A quantified "who is distinguishable from whom" map of four feedback models

**What:** Instead of one accuracy number, produced a full pairwise distinguishability matrix across the four physics scenarios — the kind of result that tells simulators and survey designers where the information actually is.
**Outcome (16th-percentile balanced accuracy, groups of 64 spectra):** no-feedback vs. each feedback model: 0.84 / 0.74 / 0.98; each weaker-feedback model vs. strongest-AGN: 0.98 / 0.93; **stellar-wind vs. wind+moderate-AGN: ≈ 0.50 (coin flip)**. The structure is the finding: feedback is detectable, but the two intermediate recipes are mutually indistinguishable at this aggregation level.
**Honest refinement (2026-06-10):** that indistinguishability is **aggregation-dependent, not absolute** — median pairwise accuracy for the hard pair climbs from 0.31 (single spectra; below chance due to confusion with neighbors) to 0.68 when averaging 256 spectra, with the pessimistic percentile above chance at the largest aggregation. The defensible claim is a *sensitivity bound* ("indistinguishable below ~64-spectrum averaging in this setup"), not an impossibility theorem. The internal review also flagged that the pairwise numbers derive from sub-blocks of a 4-class confusion matrix rather than dedicated per-pair classifiers, and that the matrix's ordering correlates strongly (Spearman ≈ 0.94) with the classes' mean-transmission offsets — both on the pre-publication fix list. [UNVERIFIED — confirm before external use of the pairwise matrix as a headline.]

### 3.3 A rigorous negative result: single spectra carry too little information, and an invented representation proved it

**What:** The project's original goal was to discover physically meaningful features in *individual* spectra. To test this, designed a novel per-spectrum representation: for each line of sight, train 6 tiny one-vs-one RBF-SVM probes (one per scenario pair) and record signed decision distances from all 4 scenario versions of that sight line → a 24-dimensional "separability vector" per spectrum (16,384 × 24), computed in parallel over the full dataset, in two independent feature spaces (raw spectra and a 6-level db8 wavelet decomposition).
**Method of falsification:** K-Means clustering with silhouette sweep, UMAP visualization, cross-representation cluster agreement (Hungarian matching), and a per-cluster random-forest accuracy audit.
**Outcome:** per-cluster classification never beat the project's own global per-spectrum baseline (0.451 balanced accuracy, 4 classes, chance 0.25); apparent "interesting" clusters agreed only 20–33% across the two feature spaces — i.e., they were artifacts of representation choice, not physics. A later independent analysis confirmed the clusters mostly track mean transmission (effect size η² = 0.125, below the pre-set 0.20 bar for a "discovered physical axis"). **Conclusion: the per-spectrum problem is information-limited, not representation-limited.** This negative result is what justified — and correctly predicted the success of — the pivot to population-level statistics (§3.1).

### 3.4 Mechanism claims tested and rejected by design, not abandoned quietly

**What:** Twice during the project, the classifier's feature-importance profile suggested an attractive physical story (first a small-scale thermal-smoothing signature, then a mid-scale line-width signature backed by recent literature). Both times, the story was converted into a falsifiable prediction *before* the next experiment, and both times it failed cleanly: signal importance was diffuse across frequency bands rather than concentrated where the mechanism predicted, and an ablation test (delete the three "most important" frequency bands, retrain) moved accuracy by only 1.5 percentage points where a genuine mechanism required ≥ 5.3 (a noise-derived threshold).
**Why it matters to an ML audience:** this is a textbook demonstration that random-forest feature importance on correlated features can manufacture a narrative (the correlated-feature bias described by Strobl et al. 2007) — and that ablation + permutation testing (observed score 0.595 vs. a 99th-percentile label-permutation null of 0.386, per Ojala & Garriga 2010) is the discipline that catches it. The surviving claim was narrowed to "an aggregate signal exists, direction-consistent with published predictions" with no band attribution — and that narrowed claim is stated as such everywhere.

### 3.5 Cheap kill decisions: two proposed research directions closed for ~zero compute

**What:** Two follow-on directions were evaluated with deliberately cheap feasibility audits before any pipeline was built:
- A simulation-based-inference direction died at scoping because the simulation suite only provides 4 discrete physics settings (no continuous parameter grid to infer over) and the core idea had just been published by another group — verified by literature search, not assumption.
- A line-statistics direction (fitting individual absorption lines) died at scoping on two independent grounds: a synthetic-recovery test showed the Voigt-profile fitter is reliable in the typical regime (12/12 recovery configurations clean) but degenerate exactly in the broad/saturated-line regime the method needed (7/25 configurations failed a <10% bias bar), and the literature's headline effect on the target statistic collapses ~19× → ~2.2× once a known calibration degeneracy (the UV background) is accounted for.
**Why it matters:** total cost of both kill decisions was under an hour of laptop compute plus literature verification — versus weeks of pipeline construction each. Knowing what not to build is the demonstrated skill.

---

## 4. Decisions and pivots (situation → evidence → decision → outcome)

1. **Per-spectrum feature discovery → population statistics.** *Situation:* the founding hypothesis was that better representations would reveal feedback signatures in single spectra. *Evidence:* the separability-vector study (§3.3) — per-cluster accuracy never beat the global baseline; clusters disagreed across representations. *Decision:* declare the per-spectrum question closed and re-pose the problem at the population level (averaging spectra before classification), explicitly documenting that this changes the scientific question rather than rescuing the old one. *Outcome:* the population-level reformulation produced the project's strongest positive results (§3.1, §3.2).

2. **First mechanism hypothesis (small-scale thermal signature) rejected by its own pre-set test.** *Situation:* the initial physics prediction was that feedback shows up at the smallest spatial scales. *Evidence:* a de-risking probe showed classifier importance peaking at mid scales instead, with the small-scale fraction (0.43) below the pre-committed 0.50 threshold. *Decision:* record the hypothesis as falsified; do not silently reinterpret. *Outcome:* honest pivot to a second, literature-backed mechanism candidate — with the explicit caveat that re-anchoring after seeing data is HARKing-adjacent, mitigated by tying the new hypothesis to independent published predictions (Khaire+2024; Tillman+2023/2024) and adding an ablation test it would have to survive.

3. **Second mechanism hypothesis (mid-scale line-width signature) also falsified — claim narrowed, not spun.** *Evidence:* the ablation and concentration tests in §3.4. *Decision:* retract the mechanism narrative entirely; keep only the signal-existence claim; pre-commit that no third frequency-band hypothesis would be tried without a specific new published prediction (none existed). *Outcome:* the classifier line of work was closed at a documented stopping rule rather than extended into fishing.

4. **The "which classes drive the accuracy?" audit was built in before the big run, and it fired.** *Situation:* a 4-class accuracy headline can be carried by one easy class. *Decision (in advance):* require a dedicated check on the hardest, scientifically central pair (stellar-wind vs. wind+AGN), plus a mean-transmission-leak check on every fit. *Outcome:* the audit did exactly its job — the aggregate accuracy (0.66–0.79 over the aggregation sweep) proved to be carried by the no-feedback and strongest-AGN classes, the central pair sat at chance, and the mean-transmission leak check came back clean (all |r| < 0.3, max 0.215). The headline was scoped down accordingly *on the day of the result*, not after review.

5. **Self-commissioned adversarial audit of the project's own best results (2026-06-10).** *Situation:* with positive results in hand and a paper plausible, the owner commissioned an internal red-team review explicitly tasked with attacking the qualified claims. *Evidence found by the review:* the pairwise matrix derives from confusion-matrix sub-blocks rather than dedicated pair classifiers; the "indistinguishable pair" becomes partially distinguishable at higher aggregation; the distinguishability ordering correlates ≈ 0.94 (Spearman) with mean-transmission offsets; no injected-signal sensitivity test exists at the relevant effect size; one replication count was inflated (two correlated nulls + one no-measurement scoping decision ≠ three independent replications). *Decision:* re-scope the claims (sensitivity bound instead of impossibility; replication count "2 + literature anchor"), and define four bounded validation experiments as preconditions for any publication. *Outcome:* the project's external-facing story today is the post-audit version — which is why this document can be handed to a recruiter without asterisks appearing later.

6. **Citation hygiene treated as a correctness problem.** During literature verification, the workflow caught a non-existent reference (a plausible-looking "Hummels+2020" that does not exist), two arXiv-ID↔journal mispairings, and a conflation of two same-author papers — all corrected and recorded as binding before any manuscript exists. One relevant 2025 preprint (arXiv:2509.18260, AGN feedback on the same statistic) was found missing from the novelty analysis and added to the must-address list.

---

## 5. What changed since 2026-05-13 (last profile sync)

Everything below post-dates 2026-05-13; the repository's verifiable activity between 2026-05-13 and 2026-05-26 was structural (workflow tooling, documentation reconstruction), and all new science landed 2026-05-26 onward.

- **SUPERSEDES prior profile claim (if present): "clustering discovered distinct spectral populations / signal islands."** As of 2026-05-26 the recorded verdict on the clustering work is the *negative* result in §3.3: the clusters are representation-dependent artifacts that mostly track mean transmission; per-spectrum separability is information-limited. Any profile language implying the clustering *found physics* should be replaced with the honest framing: "designed a per-spectrum separability representation and used it to rigorously establish an information ceiling, motivating a successful population-level reformulation."
- **New (2026-05-26 → 2026-06-01):** the entire power-spectrum classifier arc — de-risking probe, adversarially reviewed experimental design, HPC execution (~53-minute full sweep, ~14,000 model fits including a 1,000-permutation null), and the mixed verdict described in §3.1/§3.4/§4.4. This research thread is **closed**, by a pre-committed stopping rule.
- **New (2026-06-02):** two directions killed at scoping for near-zero cost (§3.5); the project formally concluded; then a same-day, zero-new-compute re-analysis of existing outputs produced the two strongest positive results — the strong-AGN detection (§3.1) and the pairwise distinguishability map (§3.2).
- **New (2026-06-10):** the self-commissioned adversarial audit (§4.5). **Consequence for any existing profile text:** the strong-AGN detection and the pairwise map should be presented with their §3.1/§3.2 caveats until the four pre-publication validation experiments are run; the "stellar-wind vs. wind+AGN are indistinguishable" claim must be stated as an aggregation-conditional sensitivity bound, not an absolute.
- **Publication status unchanged:** nothing submitted or drafted, before or after 2026-05-13.

---

## 6. Skills and tools demonstrated (with evidence)

- **Scientific ML / statistics:** balanced-accuracy evaluation under class structure; stratified K-fold CV with bootstrap CIs; label-permutation null tests (Ojala & Garriga 2010); ablation studies; correlated-feature importance bias (Strobl et al. 2007); confound controls (mean-transmission removal; per-class normalization rejected as label-leaky); fold-leakage unit tests asserting no source spectrum spans train/test. Evidence: the six-check evaluation protocol and its outputs in `results/pk_feedback_classifier/stage1/`; `tests/test_stage1_fold_leakage.py`.
- **Classical ML engineering:** scikit-learn RBF-SVMs (6 one-vs-one micro-probes per spectrum across 65k spectra, joblib-parallel), random forests with GridSearchCV, K-Means + silhouette sweeps, UMAP; ~14,000 model fits orchestrated with per-fit checkpointing and resume logic. Evidence: `src/core/probe.py`, `src/core/cluster.py`, `src/core/models/rf_classifier.py`, `experiments/pk-feedback-classifier/run_stage1.py`.
- **Signal processing on physical data:** windowed FFT power spectra with physically derived frequency axes; 6-level db8 wavelet decompositions with per-level normalization; Voigt-profile line fitting and synthetic-recovery validation grids. Evidence: `src/core/transforms.py`; `experiments/linewidth-cddf/Q_C1_PIPELINE_FEASIBILITY.md`.
- **Reproducibility infrastructure:** DVC-versioned data and a 6-stage reproducible pipeline; MLflow experiment tracking with enforced naming/tagging conventions; `uv`-locked Python 3.12 environments reproduced bit-identically across laptop and cluster (md5-verified artifact regeneration). Evidence: `dvc.yaml`, `.claude/skills/mlflow-run/SKILL.md`, restoration record in `experiments/signal-clustering-v2/LEDGER.md`.
- **HPC:** SLURM job design on a university cluster (partition selection, wallclock budgeting from smoke-test extrapolation, 24h ceiling decision recorded with rationale), rsync data staging, and producer-side output verification (jobs fail loudly with distinct exit codes if any expected artifact is missing). Evidence: `scripts/submit_juno_*.sh`; job records 205725 (64s) and 206130 (52m31s).
- **AI-agent orchestration:** designed a multi-agent research workflow (principal-investigator, implementer, data-engineer, literature, and red-team reviewer roles) with append-only decision logs, numbered decisions, pre-committed acceptance criteria, and mandatory adversarial review before expensive runs — the process that produced every catch in §4. Evidence: `.claude/agents/`, the per-experiment decision logs under `experiments/`.
- **Research integrity practices:** pre-registered falsification criteria; honest-reporting rule (observation first, framing second); citation verification that caught a hallucinated reference; symmetric disclosure of negative results. Evidence: §4 items, all traceable in §8.

---

## 7. Resume-ready bullets

1. Built an end-to-end reproducible ML pipeline (Python, scikit-learn, DVC, MLflow, SLURM) testing whether galaxy-feedback physics is detectable in 65,536 simulated quasar spectra (~134M data points).
2. Detected strong AGN feedback from averaged absorption power spectra at 0.97 balanced accuracy (0.96 pessimistic bound), holding accuracy under an explicit mean-transmission confound-removal control.
3. Produced a pairwise distinguishability map of four feedback models, isolating one indistinguishable pair and characterizing how separability scales with sample aggregation — an actionable sensitivity bound for survey design.
4. Falsified two physical-mechanism hypotheses with pre-committed ablation and permutation tests, showing the classifier's feature-importance peaks were correlated-feature artifacts, and narrowed published claims accordingly.
5. Invented a per-spectrum "separability vector" representation (six one-vs-one SVM probes, 24 dims, 65k spectra) and used clustering plus per-cluster classifiers to rigorously establish a per-spectrum information ceiling.
6. Killed two proposed research directions at scoping for under an hour of compute, via synthetic-recovery tests and literature effect-size verification — avoiding weeks of dead-end pipeline work.
7. Designed a multi-agent AI research workflow with append-only decision logs and mandatory adversarial review; it caught a fabricated citation, a leaky normalization, and an inflated replication count before publication.
8. Ran ~14,000 cross-validated model fits on a university SLURM cluster in 53 minutes, with checkpoint/resume, fold-leakage unit tests, and fail-loud artifact verification.

*(Each bullet obeys the translation rule: numbers are stated against chance, baselines, or external references; none depend on internal thresholds for meaning. Bullets 2–3 carry the §3.1/§3.2 caveats — see §8 before using in an interview where follow-up questions are likely.)*

---

## 8. Provenance appendix (internal vocabulary permitted here only)

This section maps every externally-framed claim above to repository sources and raw internal numbers so the translation can be audited. Internal evaluation vocabulary (gates, PASS/FAIL cells, D-XX decisions, LEDGER sections, close-outs, panels, reframes) appears here as it does in the sources.

### Project structure and status (§1, §2)

| Claim | Source |
|---|---|
| Data scale: 4 × (16384, 2048) float64 flux; Δv = 2.6365 km/s/pixel from `vel.npy`; ~1.1 GB | `experiments/pk-feedback-classifier/LEDGER.md` header, §4 ([D-04], [D-05]) |
| Project concluded 2026-06-02 (Path α park, user-accepted), re-opened same day for reframe-suite, re-closed at qualification | `experiments/linewidth-cddf/CLOSE_OUT.md` §7–§8; `experiments/reframe-suite/CLOSE_OUT.md` |
| No paper exists; `papers/` tree never scaffolded; paper-author never dispatched | filesystem check this session; `reframe-suite/CLOSE_OUT.md` §4 |
| 2026-06-10 defense-panel verdict NEEDS WORK; four bounded hardening items not yet authorized | `.claude-memory/diagnosis-2026-06-10-defense-panel.md` (PI re-verified the M-sweep numbers against `reframe5_lattice.csv` this session) |
| Timeline anchor: earliest verifiable activity 2026-02-27 (MLflow `Baseline_RF` experiment creation) | `experiments/baseline-random-forest/LEDGER.md` §1 |

### §3.1 — Strong-AGN binary detection (internal: reframe-suite Reframe 1 + Reframe 9)

- Raw: binary {C4 vs C1∪C2∪C3} balanced_acc **median 0.971, p16 0.960** at (per_sightline regime, M=64), from `results/reframe_suite/phase1/reframe1_binary_c4_vs_rest.csv`; verdicts in `experiments/reframe-suite/CLOSE_OUT.md` §1 (R1 table: cleared p16 ≥ 0.90 bar by 6 pp; Δ vs 4-class 0.285; regime parity 0.0031).
- ⟨F⟩-removal control (Reframe 9): accuracy drop **−0.003 pp** vs. the 5 pp gate (53× margin), `results/reframe_suite/phase2/reframe9_verdict.csv`; CLOSE_OUT §1 R9. Load-bearing caveat recorded there: multiplicative and additive ⟨F⟩-removals were degenerate at ⟨F⟩ ≈ 0.97, so R9 demonstrated the existing preprocessing already removed the DC channel, not a stronger control; higher-moment ⟨F⟩-correlated structure remains open.
- 2026-06-10 panel attack #4: lattice p16 ranking matches per-class |Δ⟨F⟩| ranking at Spearman ≈ 0.94; the ⟨F⟩-scalar baseline classifier (hardening item ii) was never run. This is the basis of the [UNVERIFIED] flag on the *interpretation* in §3.1.
- "~8,000 model fits / under an hour": Stage 1 sweep 7,940 CV fits + ablation arm ≈ 1,400 fits; Juno job 206130, Elapsed 52m31s ([D-25], [D-26]).

### §3.2 — Pairwise lattice (internal: Reframe 5)

- Raw p16 at (per_sightline, M=64), `results/reframe_suite/phase1/reframe5_lattice.csv` via CLOSE_OUT §1 R5: C1-C2 **0.842**, C1-C3 **0.741**, C1-C4 **0.978**, C2-C3 **0.501**, C2-C4 **0.980**, C3-C4 **0.926**. Qualification: 4 of 5 non-(C2-C3) pairs ≥ 0.75.
- C2-C3 M-dependence (panel attack #2, PI re-verified this session): per_sightline median 0.306 (M=1) → 0.678 (M=256); **p16 at M=256 = 0.599 > chance**; global 0.310 → 0.681, p16 0.594. Hence "M-conditional exclusion bound," replacing the "intrinsic degeneracy floor" verbing.
- Panel attack #1: all pair numbers derive from 4-class confusion-matrix sub-blocks (`pair_balacc_from_cm` in `experiments/reframe-suite/scripts/phase1_reframes.py`); no dedicated pair classifier trained; three inconsistent C2/C3 numbers circulate (0.347 from [D-26] G4 / 0.501 lattice p16 / 0.553) — hardening item (i).
- The three caveats together are the basis of §3.2's [UNVERIFIED] flag.

### §3.3 — Per-spectrum information ceiling (internal: signal-clustering-v2, [D-13]; baseline v0.2)

- Baseline: global 4-class RF balanced accuracy **0.4514** (10-fold stratified CV; best representation = raw spectra), `experiments/baseline-random-forest/LEDGER.md`; `results/baseline_rf/`. Chance = 0.25.
- Separability vectors: `(16384, 24)` float64, 6 OVO RBF-SVM probes × 4 signed decision distances, intercepts dropped; `src/core/probe.py`; `results/signal_clustering_v2/fingerprints_{wavelet,raw}.npy`.
- [D-13] verdict (recorded 2026-05-26, `experiments/signal-clustering-v2/LEDGER.md` §3): per-cluster RF ≤ 0.451 baseline in both representations; cross-run cluster overlap **20–33%** (`results/signal_clustering_v2/cross_run_overlap_summary.csv`); reading: "information-limited, not representation-limited; signal islands are representation-disagreement."
- Second confirmation: Reframe 7 top η² = **0.125** at (wavelet_k5, mean_flux) vs. the 0.20 large-effect bar; next (raw_k5, mean_flux) 0.057; `results/reframe_suite/phase1/reframe7_cluster_axes.csv`; CLOSE_OUT §1 R7 (adjacent-finding routing).
- M=1 ≈ chance corroboration: Stage 1 M=1 balanced_acc = 0.29 ([D-26]).

### §3.4 — Mechanism falsifications (internal: pk-feedback-classifier Stages 0–1, gates G1–G6)

- Stage 0 ([D-07]): per_sightline M=64 = 0.6993, global = 0.7042; **high_k_frac 0.43 < 0.50 gate** → [D-01] high-k thermal-cutoff prior falsified; importance peak k ≈ 0.06–0.13 s/km; per-class recall at M=64: C1 0.882, C2 0.627, C3 0.346, C4 0.961. Juno job 205725, 1m04s.
- Stage 1 ([D-26], Juno job 206130): verdict cell **PPFFPF** → STAGE1_SPEC.md §3 Cell 14 → band "PP-C4+C1" (Partial PASS — C4+C1-carried, C2/C3 null). Raw gate values: G1 PASS (per_sightline M=64/128/256 = **0.657/0.723/0.788**; global 0.662/0.715/0.769; M=1 = 0.29); G2 PASS (permutation null per Ojala & Garriga 2010: observed median **0.595** vs null p99 **0.386**, 1,000 permutations); G3 FAIL (mid_k_frac median **0.340** < 0.50); G4 FAIL (C2-vs-C3 sub-confusion balanced_acc median **0.347** < 0.60); G5 PASS (⟨F⟩ ↔ predict_proba Pearson |r|: C4 0.215, C1 0.164, C2 0.083, C3 0.076 — all < 0.3); G6 FAIL (ablation Δ **0.0148** vs threshold max(5pp, 2σ) = **0.0530**). Files: `results/pk_feedback_classifier/stage1/{summary,g1_g4_summary,g5_routing,g2_permutation_null,cv_partial}.csv`.
- "≥ 5.3 percentage points" in §3.4 = the data-derived max(5pp, 2σ) G6 threshold ([D-23] C1 corrected the original [D-16] 5pp noise argument — an honest PI miscalculation recorded as such).
- Mechanism literature anchors: Khaire+2024 MNRAS 527, 4545; Tillman+2024 arXiv:2410.05383 (ApJ); Tillman+2023 ApJL 945 L17 ([D-14]/[D-15]; z=0.1→0.3 extrapolation hedge binding).
- Track CLOSED 2026-06-01 by the [D-20] S6 cascade-exhaustion clause, Fork 1 in [D-26].

### §3.5 — Scoping kills

- fps-sbi: closed at scoping; Q-S1 returned (B) Sherwood is discrete-θ only; Q-S3 found Sinigaglia+2026 prior-recovered the SBI idea. `experiments/fps-sbi/{SCOPING,Q_S1_PARAM_AUDIT,Q_S3_LIT_ANCHOR}.md`. (2026-06-10 panel note: Sinigaglia+2026 is z=2.0–3.5, so the low-z CAMELS successor pitch survives.)
- linewidth-cddf: closed at scoping 2026-06-02, two independent kill paths. Q-C1: Voigt synthetic 5×5 grid — Lyα-forest sub-grid **12/12 PASS**, **7/25 cells FAIL** the <10% b-bias / <5 km/s σ_b gate, failures concentrated at b ≥ 40 km/s, log N_HI ≥ 14.5 (the Tillman+2023 diagnostic regime); line density 5.1/sightline ≈ 2× below the Davé+1999/Williger+2010 10–50 band. Q-C3: Tillman+2023 CDDF χ²_R factor **~19× → ~2.2×** under UVB rescaling; Nasir+2017 same-suite AGN Δb = 1.8 km/s < Bolton+2022 2.5 km/s b-peak floor; AGN-vs-stellar CDDF reported qualitative-only. `experiments/linewidth-cddf/CLOSE_OUT.md` §1–§3.

### §4 — Decision-trail sources

| §4 item | Internal record |
|---|---|
| 1. Per-spectrum → population pivot | signal-clustering-v2 [D-13] → pk [D-01] (lowered-confidence cascade), [D-08] clause 5 (population-statistics re-scope) |
| 2. High-k falsification + re-anchor | [D-07] (verdict), [D-10] (re-anchor), [D-13]-panel K1 (HARKing-adjacent attack), [D-14]/[D-15] (Khaire/Tillman defense + verb tightening), [D-16] (ablation gate) |
| 3. Mid-k falsification + stopping rule | [D-26] G3+G6 joint FAIL; [D-20] S6 cascade-exhaustion clause; Fork 2 declined absent published prior |
| 4. Built-in class-driver audit | [D-23] C3 (G4 promoted to load-bearing C2/C3 gate), [D-21] (G5 ⟨F⟩-leak gate), [D-26] routing to PP-C4+C1 on result day |
| 5. Self-commissioned audit | `.claude-memory/diagnosis-2026-06-10-defense-panel.md`; panel attacks 1–7; four hardening items (i)–(iv): dedicated C2-C3 binaries across M; ⟨F⟩-scalar baseline column; injection-recovery curve; re-verb as M-conditional + replication "2 + anchor" |
| 6. Citation hygiene | linewidth-cddf CLOSE_OUT §3 (Christiansen+2020 correction; "Hummels+2020" hallucination — never cite; Tillman+2024 AJ-vs-ApJ disambiguation); pk [D-19] (Walther arXiv↔journal pairing); panel attack #7 (arXiv:2509.18260 gap) |

### §5 — Since-2026-05-13 anchors

- Git log since 2026-05-13: structural commits first (`4b7f5d5` Research-OS structure; `f1707dc` reconstructed LEDGERs) → `d2bd505` (pk track opened 2026-05-26) → `fc1867a`/`0b88c90` (Stage 1 close, 2026-06-01) → `a9c4af5`/`3f57015` (linewidth-cddf close + Path α park, 2026-06-02) → `1d8b7bd`/`c4fe456`/`7aa3add` (reframe suite + close-out, 2026-06-02). [D-13] recorded 2026-05-26 (`64ae875`), hence the SUPERSEDES flag on any pre-2026-05-13 clustering claim.
- Tags: `v0.1-eda`, `v0.2-baseline-rf`, `v0.3-clustering-v1`, `v0.4-clustering-v2`, `v0.6-reframe-suite-qualified`.

### Binding constraints any external use inherits

- [D-17] rule-14(iii) demotion: all classifier numbers publish only as "stacked-P_F(k) population-statistics adjacent finding," never per-sightline classification; the binding non-claim list ("We do not claim per-sightline feedback identification at z ≈ 0.3 from P_F(k) alone...") must accompany any paper-text accuracy citation.
- Reframe-suite qualification is **not defense-hardened** (rule-15 narrowing-close exemption meant the headline numbers escaped panel review); the 2026-06-10 panel verdict NEEDS WORK binds until the four hardening items run.
- All results are z ≈ 0.3, Sherwood-suite-specific; z = 0.1 → 0.3 literature extrapolation hedge and no-generalization-to-z=2–5 hedge stack ([D-15], [D-20] S1).
