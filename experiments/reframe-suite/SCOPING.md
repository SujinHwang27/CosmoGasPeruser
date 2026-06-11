# SCOPING — reframe-suite (re-open of parked project under no/minor-compute reframes)

**Proposed track:** reframe-suite (four parallel reframes mining already-closed-track artifacts under reframed questions: Reframe 1 = binary {C4 vs rest} re-aggregation; Reframe 5 = 6-pair distinguishability lattice; Reframe 7 = signal-clustering-v2 K=5 cluster physical-axis interpretation; Reframe 9 = mean-flux-removed P_F(k) observable variant)
**Proposed branch:** `exp/reframe-suite` (off `exp/pk-feedback-classifier`'s closed HEAD; LEDGER deferred to Phase-1 results landing — Q-style audits + SCOPING first)
**Motivating decisions / parent state:** `experiments/linewidth-cddf/CLOSE_OUT.md` §7 (Path α park, 2026-06-02); `experiments/pk-feedback-classifier/LEDGER.md` [D-26] (cascade-exhaustion close); `experiments/signal-clustering-v2/LEDGER.md` [D-13] (per-sightline info ceiling).
**Status:** DESIGN PROPOSAL — no LEDGER, no branch, Phase-2 compute NOT yet authorized.

This document **un-parks** Path α (CLOSE_OUT §7) under a re-open path the user explicitly invoked: *no-new-compute and minor-new-compute reframes of existing data*. This is **NOT** one of the CLOSE_OUT §5 catalogue items (CAMELS-LH / metal lines / transverse correlations / mock survey statistics — all of which required wholesale-different-suite or wholesale-different-observable investments and remain parked under the original park guidance). The park guidance binds those §5 paths; this scoping is a different surface the original park did not anticipate.

---

## §1 — Re-open framing and cascade-inherited confidence

### Acknowledgement of the re-open

The project was parked at honest cascade-close (CLOSE_OUT §7, Path α accepted 2026-06-02). The user re-opened on 2026-06-02 with the standing directive: *"Approve. Run more experiments under new reframes if necessary."* The reframes under consideration here are the no/minor-compute set surfaced in this session's brainstorming pass — they do **not** invoke any of CLOSE_OUT §5's catalogue items, which remain parked.

### Rule-2 compliance argument (binding interpretation, PI-only sign-off, NOT provisional under rule 15(c))

Project-architect rule 2 (falsified-prior cascade) binds **new candidate priors** entering the cascade. The cascade has now closed two consecutive P_F(k)-band candidate priors ([D-01] high-k, [D-10]/[D-15] mid-k) and the [D-20] S6 clause forbids a third basis pivot on the same falsified observable axis absent a specific new published prior.

The four reframes under consideration here do **not** open a third candidate P_F(k) band. Specifically:

- **Reframe 1** (binary C4 vs rest) and **Reframe 5** (6-pair distinguishability lattice) are **re-aggregations of the existing Stage 1 cv_partial.csv 4-class confusion matrices**. No new RF fits. The underlying classifier, basis, k-bins, normalization, M-sweep, fold structure, and seed protocol are bit-identical to the closed [D-26] run; only the post-hoc aggregation of the confusion matrix changes.
- **Reframe 7** (K=5 cluster physical-axis interpretation) **mines labels produced under signal-clustering-v2** ([D-13]) against candidate physical axes (mean flux, integrated τ thermal proxy, detected line count) extracted from already-on-disk data. No new clustering, no new features, no new classifier.
- **Reframe 9** (mean-flux-removed P_F(k) variant) is a **focused re-run of a subset of Stage 1** with one preprocessing step changed (subtract ⟨F⟩ in flux space pre-FT instead of [D-06]'s k=0 + lowest-log-bin drop). Estimated at ~3-5% of Stage 1's wallclock (M ∈ {64, 256} × per_sightline × 5 seed × 5 fold ≈ 250 fits vs Stage 1's 7940). This is the only reframe touching new compute.

**Binding PI interpretation:** rule 2 binds *new candidate priors*. A re-aggregation of an already-closed classifier's outputs is not a new candidate prior; it is reading the existing artifact's information content under a coarser-grained question. A re-interpretation of a closed clustering track's labels against new physical-axis covariates is not a new candidate prior on the closed track's question; it is a discovery-science use of an artifact whose original purpose has been retired. The Reframe-9 minor-compute variant is the only reframe that even arguably touches rule 2's binding surface — and even there, it is **the same observable (P_F(k))** with **one preprocessing knob changed to test the honesty hurdle of Reframe 1**, not a new candidate band. If the user disagrees with this interpretation, override; the discipline-faithful read is the one I have stated.

**Honest counter-pressure (rule 7 self-flagging):** the closest rule-2 risk in this set is Reframe 9 — it could be read as "a fourth attempt to make P_F(k) work" rather than "a control experiment on Reframe 1's honesty hurdle." The mitigation is that Reframe 9 is gated on Phase 1 returning a Reframe-1 binary accuracy that *materially* exceeds 4-class accuracy at the same M (§3 Phase-1→Phase-2 gate). If Reframe 1 returns a binary accuracy that is not materially different from re-reading [D-26], Phase 2 is **NOT authorized** — there is nothing for Reframe 9 to disentangle. This gate is what keeps Reframe 9 from being a rule-2-anti-pattern third P_F(k) band.

### Verb ceiling on each reframe headline (per cascade-inheritance + [D-37]-extension rules 1 + 2)

The cumulative cascade (six entries per CLOSE_OUT §4) means **no** reframe headline can be verbed at higher than:

- **Reframe 1:** "first quantified binary {C4 vs C1∪C2∪C3} detection under stacked P_F(k) at z ≈ 0.3 Sherwood, in the post-cascade-close regime, with shape-vs-⟨F⟩ disentanglement gated on Reframe 9." Never "Lyα detects strong-AGN feedback" unverbed — the verb must always carry the conditional.
- **Reframe 5:** "first quantified pairwise distinguishability lattice across the 4 Sherwood feedback recipes at z ≈ 0.3 on stacked P_F(k), in the post-cascade-close regime." Never "we map feedback distinguishability."
- **Reframe 7:** "first physical-axis interpretation of the signal-clustering-v2 K=5 clusters under reframed discovery-science questions, in the post-[D-13]-falsification regime." Never "we discover physical structure" — the clusters themselves are inherited from a falsified-track artifact; their re-interpretation is hedged accordingly.
- **Reframe 9:** "first shape-vs-mean-flux disentanglement of the Reframe-1 binary signal under explicit ⟨F⟩-removal." Never "we isolate the shape signal."

All four headline-verb ceilings are binding on any future paper-author dispatch (none authorized here; user-triggered-only).

### Falsified-prior ledger line (rule 4)

Cumulative project-wide cascade ledger inherited verbatim from CLOSE_OUT §4 (six entries: signal-clustering-v2 [D-13]; pk-feedback-classifier [D-01], [D-10]/[D-15], [D-26]; fps-sbi SCOPING; linewidth-cddf SCOPING). This scoping adds no new candidate priors; on Phase-1 results landing, any new D-XX entries enter the reframe-suite LEDGER (deferred per workflow).

---

## §2 — Per-reframe gate set

### Reframe 1 — Binary {C4 vs rest} re-aggregation

**(a) Pre-committed PASS condition.** Binary {C4 vs C1∪C2∪C3} balanced_acc at M=64 stacked, per_sightline regime, 5-fold CV + 10-seed bootstrap lower-CI edge ≥ **0.90**, **AND** the binary balanced_acc materially exceeds the 4-class balanced_acc at the same M (Δ ≥ +0.10 percentage points on the median, where +0.10 is the rough margin between binary-chance-0.50-baseline and 4-class-chance-0.25-baseline normalized to the bounded-accuracy scale; a binary task should be substantially easier than the 4-class task if the C4 signal really is the load-bearing channel).

**(b) Pre-committed FAIL / inconclusive routing.** Binary accuracy fails to exceed the 0.90 lower-CI bar → Reframe 1 returns "no material binary lift over multiclass," the entire reframe set collapses to a re-aggregation of the existing null, and Reframe 9 is **NOT authorized** (Phase-1→Phase-2 gate fails). Binary accuracy clears 0.90 but Δ over 4-class < 0.10 pp → binary framing adds no information; same routing.

**(c) Honesty hurdle (rule 3 anti-degeneracy audit).** A binary {C4 vs rest} accuracy near 0.90 is **degenerate** between two distinct physical readings: (i) "C4 has a real P_F(k)-shape feedback signature" vs (ii) "C4 has a 0.5% ⟨F⟩ offset (data-engineer-confirmed; LEDGER §3 [D-02]) that survives the per-sightline normalization through a shape-encoded carry-over channel ([D-07] regime parity reading)." Reframe 1 alone cannot distinguish these. **The ⟨F⟩-vs-shape disentanglement is the load-bearing honesty hurdle and IS Reframe 9's purpose.** Reframes 1 and 9 are paired, not independent: Reframe 1's headline is rule-7-fragile until Reframe 9 returns. Until Phase 2 fires, the Reframe-1 headline is verbed at lower confidence with the disentanglement caveat on the face of every claim. If the user wishes to take Reframe 1 standalone (skip Phase 2), the headline collapses to: "binary {C4 vs rest} is detectable on stacked P_F(k); we do not isolate the contribution of the ⟨F⟩ offset from the shape channel."

### Reframe 5 — Distinguishability lattice (6 pairwise binary)

**(a) Pre-committed PASS condition.** Of the 6 pairwise binaries (C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4), produce all 6 at M=64 (and M=256 if the re-aggregation supports it without re-running) with bootstrap lower-CI edge. PASS = **at least 3 of the 5 non-(C2-C3) pairs** show lower-CI ≥ **0.75** (a "non-trivial" pair being one where one or both classes is C4 OR one class is C1 — both confirmed-separable from [D-26]), producing a non-trivial pattern beyond the trivial "C4 vs everything is easy." Y = 0.75 chosen as: chance = 0.50, [D-26] C4-recall = 0.96 carries to ≈ 0.85-ish binary accuracy for C4-pairs, and 0.75 is the half-way bar between binary chance and the empirical ceiling — clearance of 0.75 means the pair is "substantially better than chance," failure means "indistinguishable in practice."

**(b) Pre-committed FAIL routing.** Fewer than 3 of the 5 non-(C2-C3) pairs clear 0.75 lower-CI → Reframe 5 returns "lattice collapses to {C4 vs rest is easy, everything else null}"; the lattice is dominated by a single axis and adds no information beyond Reframe 1. Routing: log as adjacent finding, no headline.

**(c) Honesty hurdle.** A 6-pair distinguishability table is **rule-7-pretty** but only scientifically informative if it produces a non-trivial pattern. The expected null-feedback-vs-feedback structure (C1 separable from all of C2/C3/C4) is already known from [D-26] confusion patterns; the load-bearing-unknown rows are the *feedback-vs-feedback* pairs (C2-C4, C3-C4, C2-C3 already known null). What does this gate leave unconstrained? *Whether the lattice numerics are reproducible across seeds at sufficient precision to support a quantitative claim* — the 10-seed bootstrap on cv_partial already provides this; the gate is well-posed.

### Reframe 7 — K=5 cluster physical-axis interpretation

**(a) Pre-committed PASS condition.** For each of the two K=5 cluster sets (wavelet, raw) and for each candidate physical axis (mean flux ⟨F⟩_los; integrated optical depth ∫τ dv as a thermal-state proxy; detected line count from any available line catalog or a τ-threshold proxy), compute one-way ANOVA F-statistic and η² effect size for cluster membership predicting the axis. PASS = **at least one (cluster_set × physical_axis) combination shows η² ≥ 0.20** (Cohen 1988 "large effect" cutoff). Z = 0.20 chosen for large-effect-only — anything smaller is "yeah, clusters technically segregate weakly on this axis" and not headline-worthy.

**(b) Pre-committed FAIL routing.** No (cluster_set × physical_axis) combination clears η² ≥ 0.20 → the K=5 clusters do not track the candidate physical axes; the [D-13] "signal islands are representation-disagreement, not physics" reading holds and Reframe 7 is recorded as a *second independent confirmation* of [D-13]'s null reading. Routing: adjacent finding, no headline.

**(c) Honesty hurdle.** η²-based ANOVA is unconstrained on whether the cluster-axis correspondence is **physically meaningful** vs **a trivial artifact of mean-flux ordering** (e.g., if clusters trivially track ⟨F⟩-deciles, that is informative — it tells you what the clusters actually encode — but it is NOT the same as "the clusters discovered novel physical structure"). The honest framing of any η² ≥ 0.20 PASS is "clusters track [axis], i.e., the 24-dim separability vector encodes [axis]," not "we discovered structure." Discovery-science framing, never hypothesis-confirmation framing.

### Reframe 9 — Mean-flux-removed P_F(k) variant (PHASE 2 — minor new compute)

**(a) Pre-committed PASS condition.** Re-run M ∈ {64, 256} × per_sightline regime × 5 seed × 5 fold with one preprocessing change: explicitly subtract ⟨F⟩_los in flux space before FT (stronger than the current [D-06] k=0 + lowest-log-bin drop). PASS = binary {C4 vs rest} balanced_acc at M=64 drops by **less than W = 0.05** (5 percentage points) relative to Reframe-1 baseline at the same M. W = 0.05 chosen as: if the shape signal is real and dominant, the explicit ⟨F⟩-removal in flux space should not move binary accuracy more than the typical 5-fold-CV bootstrap SD (~3-4 pp from [D-23] C1 empirical SD computation); 5 pp is ~1.3-1.7 σ — modest tolerance, but pre-committed.

**(b) Pre-committed FAIL routing.** Binary accuracy drops by ≥ 0.05 after ⟨F⟩-removal → the C4 binary signal was substantially ⟨F⟩-carried, not shape-carried, and Reframe 1's headline collapses to "binary {C4 vs rest} is detectable through the 0.5% ⟨F⟩ offset, not through a P_F(k)-shape feedback signature." This is itself an honest result — it explains the [D-07] regime-parity puzzle (per_sightline ≈ global because the shape-encoded ⟨F⟩-carry-over channel survives both normalizations) — but it is NOT a "Lyα detects feedback" headline; it is a "Lyα measures mean flux which correlates with strong feedback" finding, which is qualitatively weaker science. Routing: record as the surviving honest interpretation, demote Reframe-1 headline accordingly, no rule-14(iii)-paper-worthy lift beyond what [D-26] already supports.

**(c) Honesty hurdle.** Phase 2 itself is the honesty hurdle for Phase 1. What does Reframe 9 leave unconstrained? *Whether the ⟨F⟩-removal preprocessing is itself complete* — subtracting ⟨F⟩_los in flux space removes a constant per sightline but does not remove higher-order mean-flux-correlated structure (e.g., the shape of F's distribution around its mean could still carry the C4 ⟨F⟩-rank as a feature). The pre-committed reading: a clean PASS (< 5pp drop) is "shape signal survives a strong ⟨F⟩-removal control"; a FAIL is "binary signal was carried by ⟨F⟩-correlated structure that survives this normalization." The gate is well-posed within its operational scope; outside that scope, the binding non-claim is "we do not claim total ⟨F⟩-independence; we claim survival of the specific preprocessing control specified here."

---

## §3 — Compute envelope and dispatch authorization

### Phase 1 — no new compute (Reframes 1 + 5 + 7)

All three Phase-1 reframes operate on already-on-disk artifacts:

- **Reframe 1 + 5 inputs:** `results/pk_feedback_classifier/stage1/cv_partial.csv` (per-fold confusion-matrix flatten column; 1400 rows already enumerating regime × M × seed × fold). Re-aggregation is a Python script reading the CSV, parsing the `confusion_flat` column back to 4×4 matrices, computing binary and pairwise balanced-accuracies per fold, and bootstrapping across seeds. Wallclock: minutes on the laptop.
- **Reframe 7 inputs:** `data/feature_discovery/labels_{wavelet,raw}_k5.npy` (16384,) + the separability vectors used to produce them (already on disk per signal-clustering-v2 LEDGER §3 [D-13] / §4 data lineage table) + `data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,tau}.npy` for the candidate physical axes (⟨F⟩_los is one mean per sightline; ∫τ dv is one integral per sightline; line counts via a τ-threshold proxy is one count per sightline — all O(16384 × 2048) reductions, seconds on laptop).

**Compute plan (Phase 1).** (a) instance = local laptop CPU; (b) hours = < 1h wallclock total across all three reframes; (c) cost = $0 marginal; (d) ceiling = none required; (e) auto-stop = N/A (process completes naturally); (f) lifecycle = outputs to `results/reframe_suite/phase1/` (Git-tracked CSVs; no DVC stage; no MLflow run — same as Stage-0 probe per [D-01] discipline). This compute plan satisfies the "Economic compute" minimum.

**Dispatch authorization (Phase 1).** I authorize **support-researcher** dispatch for all three Phase-1 reframes as a single brief (Reframes 1 + 5 + 7 are a natural unit: re-aggregation + correlation analyses on existing artifacts, all CSV/NumPy-level work). Deliverables: (i) `results/reframe_suite/phase1/reframe1_binary_c4_vs_rest.csv` (per regime × M, binary balanced_acc median + 16/84 percentile + Δ vs 4-class); (ii) `results/reframe_suite/phase1/reframe5_lattice.csv` (6 pairwise rows × regime × M with bootstrap CI); (iii) `results/reframe_suite/phase1/reframe7_cluster_axes.csv` (cluster_set × physical_axis × η² × F-stat × p-value). One brief, one return, one PI review.

**Phase-1 dispatch is authorized immediately** — no caveats to surface beforehand. (The honesty hurdles named in §2(c) are downstream-of-Phase-1; they don't gate Phase-1 execution, they gate Phase-2 + qualification reads.)

### Phase 2 — minor new compute (Reframe 9)

**GATED on Phase-1 returning a Reframe-1 binary accuracy that materially exceeds 4-class accuracy at the same M.** If Reframe-1 fails its PASS condition (§2 Reframe 1 (a)), Reframe 9 is **NOT authorized** — there is nothing for Reframe 9 to disentangle, and dispatching it would be a rule-2-anti-pattern third P_F(k) attempt.

**If gate passes:** Phase 2 dispatches to **core-implementer + infrastructure-manager** with a focused Stage-1-subset brief (M ∈ {64, 256} × per_sightline × 5 seed × 5 fold = 250 fits, vs Stage 1's 7940). The single methodology change: replace [D-06]'s k=0 + lowest-log-bin drop with explicit per-sightline ⟨F⟩ subtraction in flux space before the FT. Expected wallclock per Stage-1 ceiling extrapolation (CLOSE_OUT §2 cited 3-5% of Stage 1; Stage 1 ran 52m31s for 7940 fits, so 250 fits ≈ 100s of compute + I/O + setup ≈ 30-60 minutes wallclock realistic). Local laptop sufficient; Juno not required.

**Compute plan (Phase 2, prospective).** (a) laptop CPU; (b) ≤ 1h wallclock; (c) $0; (d) no ceiling required; (e) auto-stop on completion; (f) outputs to `results/reframe_suite/phase2/`. Dispatch authorization for Phase 2 is **DEFERRED until Phase-1 returns**.

---

## §4 — What "success" looks like (qualification gates)

Per the freshly-sharpened paper-trigger rule, paper-author is user-triggered-only IFF solid/successful results. The user has NOT triggered paper authoring with this re-open; they have authorized experiments to determine whether the reframed results *would qualify*. Define "qualify" per reframe:

- **Reframe 1 qualifies as "solid/successful"** if: binary {C4 vs rest} balanced_acc at M=64 stacked, per_sightline regime, 5-fold CV + 10-seed bootstrap lower-CI ≥ **X = 0.90** AND Δ vs 4-class ≥ 0.10 pp on median AND the per_sightline/global regime parity holds at the binary level (a tight Δ ≤ 0.005 between per_sightline and global binary accuracies, mirroring [D-07]) AND Reframe 9 returns < 5pp drop on ⟨F⟩-removal (Phase 2 PASS). All four conjuncts required. Failing any single conjunct → does not qualify; the verdict is honest-adjacent-finding only.

- **Reframe 5 qualifies as "solid/successful"** if: at least **3 of the 5 non-(C2-C3) pairs** show M=64 lower-CI ≥ **Y = 0.75** AND the resulting lattice produces a non-trivial pattern (i.e., NOT just "everything-vs-C4 is easy; everything else null"). Concretely: at least one of C1-C2 or C1-C3 (NoFB-vs-stellar/AGN, both feedback recipes) clears 0.75; OR at least one of C2-C4 / C3-C4 (stellar-vs-AGN-class pairs) clears 0.75. This is a "structural" gate on lattice informativeness, not just a count.

- **Reframe 7 qualifies as "solid/successful"** if: at least **one (cluster_set × physical_axis) combination** shows η² ≥ **Z = 0.20** with clean interpretation (i.e., the axis the clusters track is physically meaningful, not trivially "the clusters re-discovered mean flux deciles" — though the latter is itself a publishable adjacent finding, just not a "solid/successful" headline). The qualification is honest-physical-axis-discovery, not pretty-η². If clusters track ⟨F⟩ trivially, that returns as "Reframe 7 explains what the K=5 clusters encode (mean-flux structure), confirming [D-13]'s representation-disagreement reading" — informative but adjacent.

- **Reframe 9 qualifies as "solid/successful" (as Phase-2 control on Reframe 1)** if: binary {C4 vs rest} balanced_acc at M=64 drops by < **W = 0.05** under explicit ⟨F⟩-removal. This is the *enabling* condition for Reframe 1 to qualify; it does not itself produce a standalone headline.

**X = 0.90 / Y = 0.75 / Z = 0.20 / W = 0.05 chosen as pre-committed numeric thresholds.** *(2026-06-11 amendment per HARDENING_SPEC §6 E5: more precisely, these are post-hoc qualification thresholds — authored after the Stage-1 confusion matrices were on disk; pre-committed only relative to the Phase-1 re-aggregation run, commit `e090b52`.)* Rationale logged per gate in §2 (a)/(c). All four are project-internal bars (rule-14-fragile by construction; no external published anchor exists for any of them — the lit-search closures in [D-12] / [D-14] / linewidth-cddf Q-C3 establish this). Per rule 14, the rescue path is rule-14(ii): pre-committed process-failure paths producing **no qualification claim** under specified failure conditions. The "no headline if FAIL" routing in §2 (b) of each reframe is the explicit rule-14(ii) invocation.

**I do NOT recommend paper-author dispatch.** Qualification per the above is the user's decision surface — these are the numbers the user will read against to decide whether to trigger paper authoring. The user's standing reasoning ("paper-author is needed if and only if we have solid and successful results") binds; the gates here are the operational definitions of "solid/successful" for this re-open's deliverable shape.

---

## §5 — Honest [D-37] assessment of expected outcome shape

The most-likely outcome shape per reframe, called honestly going in:

- **Reframe 1 (most likely):** high binary accuracy (≥ 0.90 at M=64) is expected from Stage 0's C4 recall = 0.96 alone (a binary aggregation of a confusion where one class has 0.96 recall and the other three each have ~0.4-0.6 recall lands near 0.90+ on balanced accuracy as a matter of arithmetic). The PASS-of-PASS-condition is high probability. **The decisive question is whether Reframe 9 survives the honesty hurdle** — i.e., whether the binary signal is shape-encoded (Reframe-1 + Reframe-9 jointly qualify) or ⟨F⟩-carry-over (Reframe-9 fails and Reframe-1 collapses to a measurement-of-mean-flux-offset finding). Most-likely outcome shape: Reframe 1 returns "binary PASS, Δ vs 4-class moderate not huge," Reframe 9 returns "binary accuracy drops materially but not to chance after ⟨F⟩-removal." The qualifying conjunction is a coin-flip-or-worse.

- **Reframe 5 (most likely):** the 4 unknown pairs (C1-C2, C1-C3, C2-C4, C3-C4) cluster into two groups — NoFB-vs-feedback pairs (C1-C2, C1-C3) are clean (high accuracy because C1's separability from feedback recipes is well-established from [D-26] confusion); feedback-vs-feedback pairs (C2-C4, C3-C4) are intermediate-to-null (the [D-26] G4 C2-C3 null at chance suggests stellar-wind-vs-wind+AGN distinguishability is broadly weak). The lattice headline most likely reads "Lyα at z = 0.3 discriminates feedback-vs-no-feedback strongly; feedback-recipe-vs-feedback-recipe distinctions partially or fully null." This is informative — it sharpens the [D-26] cascade-close reading — but the "lattice" framing may collapse to a 2-bucket finding rather than a rich 6-pair table. Qualification probability: moderate; the count-3-of-5 gate is achievable but the structural informativeness gate (non-trivial pattern beyond "C4 + C1 are easy") is the real bar.

- **Reframe 7 (most likely):** the K=5 clusters in either representation track mean flux + thermal-state percentiles strongly (η² ≥ 0.20 likely on at least one axis × representation). This would be **honest discovery science explaining what the K=5 clusters actually encode**, but it is NOT a "feedback discrimination" finding — the clusters did not track feedback labels (that was [D-13]'s null); finding that they track ⟨F⟩ instead is a coherent re-interpretation but does not lift the cascade. Qualification probability: high on the numeric gate (η² ≥ 0.20 likely passes); the "physical meaningfulness" interpretation question is where the honesty hurdle lives — and the honest read is most likely "clusters encode mean-flux ordering, which is what the 24-dim separability basis surfaces when the class-discrimination signal is weak."

- **Reframe 9 (most likely, Phase 2):** binary C4 accuracy drops materially but not to chance after ⟨F⟩-removal — i.e., there IS a shape component but it's smaller than the headline-from-Reframe-1 suggested. Most-likely drop: 0.05-0.15 pp range. This sits **on or just outside the W = 0.05 PASS bar**. Honest framing of either side: PASS-by-the-margin = "shape signal survives strong ⟨F⟩-removal but is reduced; not robust to all preprocessing variants"; FAIL = "binary C4 signal is substantially but not exclusively ⟨F⟩-carried." Both are scientifically informative; only the PASS lets Reframe 1's headline qualify.

### What the user should know going in

The most-likely deliverable surface across all four reframes is a **quantified, externally-corroborated "what Lyα at z ≈ 0.3 can and cannot resolve" lattice** — sharpening the cascade-close reading from [D-26]/CLOSE_OUT with finer-grained numeric structure (binary detection bar, distinguishability lattice, cluster physical-axis interpretation, shape-vs-mean-flux disentanglement). This is **NOT** a strong positive feedback-classifier result; it is a sharpened, quantified version of the same honest cascade-close that landed at park, plus the cluster re-interpretation as adjacent discovery science.

**Whether this clears the user's solid/successful bar** is a judgment call the user owns. The PI's honest read: **a clean joint Reframe-1 + Reframe-9 PASS (binary headline + shape-survives-control) IS a publishable adjacent finding** under rule-14(iii) demotion (per [D-17]'s deliverable contract), but probably **NOT** a "headline classification" result; **anything short of that joint PASS** collapses to the existing CLOSE_OUT story with extra quantification. The experiments are the only way to find out which shape lands.

### Decision-quality discipline (rule 7)

The same discipline that produced CLOSE_OUT §6's rule-7 sound-decision-quality framing applies here. The spec is hedged per rule 2 (verb ceiling on every headline; explicit cascade-inheritance acknowledgment). The anti-degeneracy audit per rule 3 named the failure spaces honestly per reframe (the ⟨F⟩-vs-shape degeneracy on Reframe 1; the lattice-collapses-to-1-bucket failure on Reframe 5; the trivially-tracks-mean-flux failure on Reframe 7; the preprocessing-incompleteness scope on Reframe 9). Falsification criteria are pre-committed per rule 5. The outcome may be mixed; the discipline is sound. If the reframes return null, that is a valid end state, and the project re-parks at a slightly-sharper cascade-close.

---

**Authored by:** project-architect (PI), 2026-06-02.
**Branch:** `exp/pk-feedback-classifier` (track-internal scoping; reframe-suite LEDGER deferred to Phase-1 results landing).
**Sign-off provenance:** PI-only, NOT provisional per rule 15(c).
**Downstream dispatch (Phase 1):** support-researcher — single brief covering Reframes 1+5+7, deliverables per §3 above.
**Downstream dispatch (Phase 2):** DEFERRED until Phase-1 returns. On Reframe-1 PASS, core-implementer + infrastructure-manager for the 250-fit Stage-1-subset re-run.
**Paper-author:** NOT dispatched. Will remain not-dispatched through any outcome of this scoping per user-triggered-only rule; the qualification gates in §4 are the surface the user uses to decide whether to trigger.
