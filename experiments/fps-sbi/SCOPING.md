# SCOPING — fps-sbi (methodology-axis pivot from pk-feedback-classifier)

**Proposed track:** fps-sbi (flux power spectrum + simulation-based inference)
**Proposed branch:** `exp/fps-sbi` (off `main`; sibling to closed `exp/pk-feedback-classifier`)
**Motivating decision:** `experiments/pk-feedback-classifier/LEDGER.md` [D-26] (cascade-exhaustion close per [D-20] S6 Fork 1)
**Status:** DESIGN PROPOSAL — no LEDGER, no branch, no compute committed by this document.

This is a scoping pass authored at user request to evaluate a methodology-axis re-entry (classification → regression/SBI) on the same P_F(k) observable that the pk-feedback-classifier track closed against. Per [D-26]'s "On user re-open" clause path (b), this is the rule-2-clean methodology pivot, NOT a third candidate band on the classification axis.

---

## §1 — Cascade-inherited confidence framing (rule 2)

The inheriting cascade, recorded honestly per [D-37]:

1. `signal-clustering-v2` [D-13] — per-sightline feedback separability is information-limited (per-cluster RF ≤ v0.2 baseline ≈ 0.451 across both representations); falsified the "better per-sightline features recover separability" prior at high confidence.
2. `pk-feedback-classifier` [D-01] (high-k thermal-cutoff prior) — falsified by Stage 0 [D-07] (importance peak at mid-k, not high-k).
3. `pk-feedback-classifier` [D-10]/[D-15] (mid-k line-width-regime prior, with Khaire+2024 / Tillman+2024 partial defense) — falsified by Stage 1 [D-26] (G3 FAIL: mid_k_frac 0.34 < 0.50; G6 FAIL: ablation Δ 1.5pp vs threshold 5.3pp).

Two consecutive same-confidence mechanism-attribution priors on the **same observable** (P_F(k)) have been falsified. Per project-architect rule 2 (falsified-prior cascade), the next prior in this lineage inherits **two levels** of confidence downgrade. The methodology-axis pivot (classification → regression/SBI) does NOT reset the cascade, because the underlying observable is unchanged; only the inference framing changes.

**Confidence ceiling on the fps-sbi headline claim, verbed:**

> "First SBI-on-FPS posterior-width measurement of Sherwood-feedback-class distinguishability at z ≈ 0.3, in the stacked regime where a scale-dependent signal exists (per [D-26] G1+G2 PASS direction-confirmation of Khaire+2024 / Tillman+2024)."

**NOT admissible** as headline:
- "We measure feedback parameters from Lyα forest at z ≈ 0.3."
- "SBI recovers feedback recipes from P_F(k)."
- "We constrain AGN feedback efficiency from low-z flux statistics."

The verb "measure" / "recover" / "constrain" is reserved until a posterior with quantified width against an external anchor lands. Until then: "posterior-width measurement," "distinguishability quantification," "first test of."

**Falsified-prior ledger line (rule 4):**
- pk-feedback-classifier [D-01] (high-k mechanism) — FALSIFIED, [D-07]
- pk-feedback-classifier [D-10]/[D-15] (mid-k mechanism) — FALSIFIED, [D-26]
- signal-clustering-v2 [D-13] (per-sightline information ceiling) — falsified the broader "richer features rescue separability" prior; expected to RE-MANIFEST in this track as wide per-sightline SBI posteriors.

**Anti-degeneracy audit (rule 3):** what does an SBI posterior-width metric leave unconstrained when the supervision signal is weak? Answer: posterior width is **bounded below by the prior width** — a wide posterior is degenerate with "prior recovered, no information learned," and reporting only "posterior width" without a "prior-to-posterior KL or posterior-shrinkage statistic" allows an uninformative null to read as a number. Q-S3 must produce a posterior-width bar that has the right scale to distinguish "prior-recovered" from "prior-narrowed."

---

## §2 — Three open questions, scoped (NOT decided)

### Q-S1 — Sherwood parameter space (is it continuous, or 4 discrete recipes?)

**Question.** Are the 4 feedback classes parameterized as continuous knobs (e.g., AGN energy-injection efficiency, stellar-wind velocity, mass-loading factor) somewhere in the Sherwood metadata, or are they 4 discrete simulation runs at fixed feedback recipes with NO continuous parameter grid available locally?

**What we know.**
- The 4 classes are named in `CLAUDE.md` and the LEDGERs as `1=NoFeedback, 2=StellarWind, 3=WindAGN, 4=WindStrongAGN` — discrete-recipe naming, no continuous knob exposed in the on-disk layout (`data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,vel,tau,axis,wave}.npy`).
- The Sherwood reference paper (Bolton+2017) and Sherwood-Relics (Puchwein+2023) suite documents whether the four runs share initial conditions and vary only feedback prescription, vs being independent realizations.
- `pk-feedback-classifier` [D-12] noted that the Lyα/IGM ML community frames feedback inference as **parameter regression / SBI / emulation, not discrete classification** (Maitra 2026 CAMELS-SBI; Tillman 2024 P_F(k) parameter sweep; Pirecki 2025 CAMELS-SIMBA). Those works rely on **continuous-parameter training suites** (CAMELS Latin hypercube, ~1000 sims). Sherwood likely does NOT have that.

**Why it is load-bearing.**
- **If continuous-parameter:** SBI on (P_F(k) | θ_feedback) is the standard CAMELS-style problem; training samples are interpolation in θ-space; posterior-width measurement against θ is a real measurement. This is the "regression/SBI as it is done in the literature" outcome.
- **If discrete-only (4 recipes, no grid):** SBI degenerates to a **discrete likelihood-ratio test over 4 recipes**, which is structurally a 4-class classification problem in fancier clothing. The pk-feedback-classifier verdicts inherit nearly unchanged — Bayes factors on discrete feedback recipes are mathematically a reparameterization of `predict_proba` for an RF classifier. The "methodology pivot" claim weakens substantially in this regime.

**What we'd need before kickoff.**
1. Inventory the Sherwood metadata on disk (any `params.txt`, `README`, simulation log) for continuous-parameter exposure.
2. If on-disk is silent: a literature pass on the Sherwood suite documentation (Bolton+2017, Puchwein+2022 Sherwood-Relics) to confirm whether these 4 are sibling runs of a parameter scan or independent recipes.
3. Honest answer of the form: "discrete-only" OR "continuous-on-θ for {param1, param2, ...} with sample density N at z=0.3."

**Owner.** `data-engineer` (on-disk metadata audit). May escalate to `support-researcher` for literature confirmation if metadata is silent. User may know offhand — quickest path.

### Q-S2 — Observable definition (per-sightline / stacked / joint richer basis?)

**Question.** What is the SBI training observable? Three candidates:

- **(i) Stacked P_F(k) at chosen M.** Reuses Stage 1 stacked-P_F(k) numerics computed for M ∈ {1, 4, 16, 32, 64, 128, 256} per regime — NO new compute for training data. The stacking already showed G1 PASS at M ≥ 64. But: stacking conflates per-sightline information loss with population statistic — the SBI "measurement" is a measurement of a *population summary*, not a per-sightline observable. Maitra-2026 CAMELS-SBI works in this regime (per-snapshot statistics).
- **(ii) Per-sightline P_F(k).** SBI implicitly does the population averaging via the likelihood model; keeps single-sightline information theoretically. BUT: [D-26] G1 M=1 ≈ chance (0.29) re-confirms [D-13]'s per-sightline information ceiling AT THE P_F(k) BASIS. A correct SBI would report this as a **posterior width ≈ prior width** (uninformative). That is the [D-37] honest expected outcome — but it produces a publishable "quantified null" not a "measurement."
- **(iii) Richer joint observable.** P_F(k) + line-width statistics + CDDF (per `pk-feedback-classifier` [D-13] panel P5). Most information-rich, most engineering cost, AND introduces a new rule-14(i) self-anchored-bar fragility — no published prior for SBI on a richer-than-P_F(k) basis at z = 0.3 was identified in [D-12]/[D-14]. Without a published prior, (iii) is the rule-2 anti-pattern S6 was written to prevent in disguise.

**PI recommendation (preliminary, subject to Q-S1/Q-S3 returns).**

> **Recommend (i) Stacked P_F(k) at M ∈ {16, 64} as primary training observable, with (ii) per-sightline P_F(k) added as a paired diagnostic arm to publish the per-sightline posterior-width null cleanly.**

Rationale:
- (i) is the **literature-anchored** observable — CAMELS-SBI and Maitra+2026 work in stacked / per-snapshot regimes. The external anchor for the bar (Q-S3) is most likely defined against this same observable family.
- (ii) is a near-zero-marginal-cost addition (SBI runs on the same P_F(k) features; only the stacking step differs). The (i)+(ii) pair lets the headline be "SBI gives [width X] posterior on stacked P_F(k) vs [width ≈ prior] on per-sightline P_F(k)" — a *quantified* statement of the cascade-exhaustion outcome, not a denial of it.
- (iii) is **REJECTED** at scoping for cascade-rule-2 reasons; can be revisited only if Q-S3 returns a published prior that specifically predicts richer-basis-recovery of feedback parameters at low z, AND a defense-panel APPROVE on that pivot is obtained.

**What would change the recommendation.**
- If Q-S1 returns "discrete-only," the SBI-on-stacked-P_F(k) recommendation weakens — the "measurement" becomes a 4-recipe likelihood-ratio test and we should consider whether the deliverable is worth opening a new track for at all (vs folding into a paper on the existing pk-feedback-classifier verdict).
- If Q-S1 returns "continuous-on-θ with dense sample grid," the recommendation stays at (i) but the posterior surface becomes interpretable as θ-recovery rather than recipe-distinguishability — larger deliverable surface, more rule-14 discipline required.

**Owner.** PI + defense-panel (this is a methodology decision, not a data or literature question). Decision deferred until Q-S1 and Q-S3 return.

### Q-S3 — External anchor for posterior-width bar (rule 14 critical)

**Question.** Does the cascade-inherited literature ([D-14] Khaire+2024 / Tillman+2024 verified citations, plus the Maitra+2026 CAMELS-SBI work cited in [D-12] Deliverable-1) provide an external **posterior-width** bar — a specific numeric the SBI posterior must clear (or NOT clear) to declare "consistent with published"?

**Two candidate anchors.**

- **Anchor A — Khaire+2024 / Tillman+2024 ΔP_F(k) magnitudes translated to posterior width.** Khaire+2024 says "≥ 5% difference at k > 0.04 s/km" between Illustris and IllustrisTNG; Tillman+2024 says "AGN feedback has 2% effect on P1D for k < 5×10⁻² s/km, 8% effect for k > 5×10⁻² s/km." These are ΔP_F(k) magnitudes between known recipes, NOT posterior-width predictions. To make this a posterior-width bar, we would need to **forward-propagate the ΔP_F(k) magnitude into an expected posterior-width via Fisher analysis** on the same M-sweep — which itself is project-internal computation and re-introduces rule-14 self-anchoring.
- **Anchor B — Maitra+2026 CAMELS-SBI posterior-width benchmarks.** If Maitra+2026 (or Tillman 2024 or Pirecki 2025) reports posterior widths on AGN feedback parameters in a comparable SBI setup, those are **external numerics** we can self-test against without forward-propagating anything. This is the rule-14(i)-clean anchor IF the numbers exist.

**What we need before kickoff.**
1. `support-researcher` literature dispatch (web-permitted, per [D-14] precedent that web tools surface useful posterior-width numerics): verify whether Maitra+2026 / Tillman+2024 / Pirecki+2025 report posterior widths in σ-units on AGN feedback parameters, at z comparable to 0.3 (or extrapolatable per the [D-15] hedging discipline).
2. If found: extract the numeric posterior width and cite — that becomes the pre-committed bar. PASS = our posterior width comparable within 2× / NOT-PASS = our posterior width ≥ 2× wider (consistent with "stacked Sherwood under our SBI setup is uninformative compared to CAMELS-trained SBI").
3. If NOT found: the bar reverts to a rule-14(ii) pre-committed-process-failure construction (we pre-commit a specific posterior-width threshold below which the deliverable is published and above which it is NOT), AND the deliverable is demoted at scoping to **rule-14(iii) adjacent-finding only** — same surface as `pk-feedback-classifier` [D-17].

**PI recommendation (preliminary).**

> **Recommend Anchor B (Maitra+2026 CAMELS-SBI posterior-width benchmarks) as primary external anchor, with Anchor A held as supporting cross-check on the signal-existence direction only.**

Rationale: Anchor A is structurally rule-14-fragile (requires Fisher propagation through our setup); Anchor B is rule-14(i)-clean if the numbers exist. Anchor A still has value as a **signal-existence cross-check** — if our SBI posterior says "consistent with NoFeedback" but the ΔP_F(k) magnitude at our k-range is 5–8% per Khaire/Tillman, that is a sharp inconsistency worth surfacing — but as a *direction check*, not a posterior-width bar.

**Owner.** `support-researcher` with WebSearch/WebFetch permitted (per [D-14] precedent that the support-researcher tool list does not include web tools by default — the dispatch must be configured to allow them, or fall back to general-purpose). Dispatch is BLOCKING per §4 gate sequence.

---

## §3 — Branch + LEDGER structure proposal

### Branch
- **Name:** `exp/fps-sbi`
- **Off:** `main` (NOT off `exp/pk-feedback-classifier`; this is a new methodology track, not a continuation — per CLAUDE.md experiment-isolation rule).
- **Author:** user, on user acceptance of this SCOPING.md + defense-panel APPROVE. PI does NOT create the branch in this dispatch.

### LEDGER skeleton (proposed; NOT authored here)

Title: `LEDGER — fps-sbi`

§1 Pulse row outline (Stage 2 = first stage of this track):

| Stage | Focus Area | Status | Pass Condition | Outputs |
|:---|:---|:---|:---|:---|
| **Stage 2 — SBI on stacked P_F(k)** | Build amortized neural likelihood / posterior estimator on stacked P_F(k) at M ∈ {16, 64}, both norm regimes, against 4-feedback-recipe labels. | 🟡 PROPOSED — pending Q-S1, Q-S3, defense-panel APPROVE | (deferred to spec — depends on Anchor B return) | (deferred to spec) |

**Stage numbering rationale.** Local D-XX numbering starts at **[D-01]** per experiment-isolation rule (precedent: `pk-feedback-classifier/LEDGER.md` notes `[D-12]` is local to that track despite an unrelated [D-12] existing in `signal-clustering-v2`). Stage numbering: keep "Stage 2" naming continuity with the project's Stage 0 (pk-feedback-classifier de-risking probe) / Stage 1 (pk-feedback-classifier stacked-pipeline) lineage to make the cascade reading-order obvious in the project narrative — i.e., Stage 2 = "what comes after Stage 1's cascade-exhaustion close." Alternative: re-number to Stage 0 (fresh track) — defensible but loses the cascade-inheritance reading. PI's preference: **Stage 2** with the LEDGER §1 row title explicitly naming the inheritance ("first stage of fps-sbi; inherits pk-feedback-classifier Stage 1 [D-26] cascade-exhaustion close").

§3 starts at [D-01]; [D-01] is the local re-statement of "this track inherits pk-feedback-classifier [D-26]'s methodology-pivot fork; cascade-inheritance and verb-ceiling per §1 of this SCOPING.md."

### Existing artifact re-use

`results/pk_feedback_classifier/stage1/` contains the Juno-produced numerics: `cv_partial.csv`, `g1_g4_summary.csv`, `g2_permutation_null.csv`, `g5_routing.csv`, `g6_ablation.csv`, `summary.csv`. These were computed under the pk-feedback-classifier methodology (RF classifier with 5-fold CV / 10-seed bootstrap / M-sweep), and they are NOT SBI training data per se — but the underlying **stacked P_F(k) feature vectors** that fed those CSVs are the same training observable Stage 2 would consume.

**Recommendation:** rather than re-running the stacking + P_F(k) transform (engineering cost, recompute risk), **re-use the Stage 1 P_F(k) feature arrays as SBI training data** under the following artifact-versioning contract:

1. **DVC-track the Stage 1 P_F(k) feature arrays as a frozen artifact.** If they are not already DVC-tracked (current Stage 1 outputs are summary CSVs + the cv_partial table, the underlying feature matrices may be in-memory-only inside the Juno run script), then **first task on the new track is for `data-engineer` to recompute-and-freeze them under DVC** — bit-identical to the Stage 1 production run (deterministic seed-controlled transforms per `pk-feedback-classifier/LEDGER.md` Methodology §2). This is the rule-15 honest path: do not assume the on-disk artifact equals the Stage 1 production artifact without an md5 re-verification.
2. **Symlink or copy, NOT in-place edit.** Stage 1 results CSVs stay frozen on `exp/pk-feedback-classifier` branch; the new track reads them via either a DVC pull (preferred) or a one-time copy into `data/feature_analysis/fps_sbi_training/` with a `provenance.md5` file recording the source-artifact hash.
3. **Versioning gate.** Any SBI claim citing a Stage 2 result must trace, via the LEDGER, back through the feature-artifact md5 → the Stage 1 production run_id (`pkstage1-20260601-184636-76c1de`) → the pk-feedback-classifier [D-26] verdict. This is the rule-15 inherited-claim re-verification path for any downstream paper-author dispatch.

**Net compute saving estimate.** Stage 1 wallclock was 52m31s on Juno for the full M-sweep including 50-fold-bootstrap RF ensemble. The P_F(k) feature transform alone (no RF) is ~5–10% of that, so ~3–5 minutes — small absolute savings, but the value of NOT re-running is the **provenance continuity** with [D-26], not the wallclock.

---

## §4 — Pre-kickoff gate sequence

Ordered gates before any compute commits. Each gate is BLOCKING on the prior.

1. **User acceptance of this SCOPING.md.** No downstream dispatch without explicit user GO. PI does not assume acceptance.

2. **`data-engineer` dispatch on Q-S1 (Sherwood parameter-space audit).** BLOCKING. Self-contained brief: "Inventory `data/preprocessed/Sherwood_z0.3_inf/` and `data/raw.dvc`-pointed contents for any continuous-parameter metadata across the 4 feedback classes. If silent on-disk, identify the Sherwood suite reference paper(s) (Bolton+2017 / Puchwein+2022 Sherwood-Relics) and report whether the 4 classes are sibling runs of a continuous-parameter scan or independent feedback prescriptions at fixed values. Return: 'continuous on {params}' OR 'discrete-only, 4 fixed recipes' OR 'metadata-inconclusive, recommended escalation: support-researcher lit pass.'" Web tools permitted if local metadata silent.

3. **`support-researcher` dispatch on Q-S3 (external posterior-width anchor).** BLOCKING. **Parallel with #2** — same session. Self-contained brief: "Identify the most recent / authoritative SBI-on-flux-statistics work that reports posterior widths on AGN-feedback-related parameters at z near 0.3, or extrapolatable to 0.3 per [D-15] hedging discipline. Priority candidates: Maitra+2026 (CAMELS-SBI), Tillman+2024 (2410.05383), Pirecki+2025 (CAMELS-SIMBA), Wadekar+2023 (LyαNNA). Return: posterior-width numeric in σ-units on a named feedback parameter at named z, with the citation; OR 'NULL — no published posterior-width benchmark exists for this regime' (rule-14(ii)/rule-14(iii) rescue path mandatory at scoping). Web tools required."

4. **PI authors Stage 2 spec ([D-02]+ in fps-sbi LEDGER).** Uses Q-S1 return (parameter space → discrete-likelihood-ratio vs continuous-θ-recovery framing); Q-S3 return (posterior-width bar); Q-S2 PI recommendation (stacked + per-sightline paired). The spec must include: (a) gate set (with falsification paths per rule 5), (b) cascade-inheritance verb-ceiling per §1 of this SCOPING, (c) anti-degeneracy audit per rule 3, (d) compute envelope per "Economic compute" discipline, (e) deliverable-surface verb whitelist + binding non-claim list per `pk-feedback-classifier` [D-17] precedent, (f) rule-14 rescue path explicit (Anchor A or B from Q-S3, plus pre-committed-failure path).

5. **defense-panel adversarial review of the spec.** BLOCKING. The pk-feedback-classifier [D-13] panel review precedent applies — adversarial role-play against rules 2/3/4/5/6/7/14/15. Panel APPROVE WITH CAVEATS path expected (per [D-13] precedent); PI amendment cycle until APPROVE clean.

6. **On panel APPROVE: `core-implementer` + `infrastructure-manager` dispatch.** Implementer authors SBI training script + spec doc; infrastructure-manager scopes Juno re-use (the [D-05] juno-hpc skill applies unchanged; .env JUNO_* block already provisioned). Smoke-timing run before full sweep per `pk-feedback-classifier` [D-25] precedent. **NOT before this gate.**

**`paper-author` NOT dispatched at any point in this gate sequence.** User-triggered-only per CLAUDE.md / PI binding rule. Even when fps-sbi Stage 2 lands a verdict (PASS, NULL, or Ambiguous), the verdict is captured in fps-sbi/LEDGER.md §3 + §7 ONLY — no paper-author dispatch absent explicit user trigger.

**No compute is committed by this SCOPING.md.** Acceptance of this document is a methodology-pivot scoping authorization only. Gates 2–6 each have their own authorization step downstream.

---

## §5 — Honest assessment per [D-37] (most likely failure mode)

The most likely outcome shape for this track, given the cascade, is: **the SBI posterior on per-sightline P_F(k) is approximately the prior** (wide; uninformative), **and the SBI posterior on stacked P_F(k) at M=64 is narrower than the prior on the C4-vs-others axis but approximately the prior on the C2-vs-C3 axis** — i.e., the quantitative SBI re-confirmation of `pk-feedback-classifier` [D-26]'s "C4+C1-carried, C2/C3 null" reading at the same observable, now translated into posterior-width units.

That outcome is a *quantified statement* of the cascade-exhaustion result, not a refutation of it. Under rule-14(iii) adjacent-finding terms (the same demotion `pk-feedback-classifier` [D-17] established), it is publishable as: "Posterior width on Sherwood feedback-recipe distinguishability from stacked P_F(k) at z ≈ 0.3 is [X] on the C4-vs-others axis and [≈ prior] on the C2-vs-C3 axis; per-sightline P_F(k) posteriors recover the prior, consistent with the [signal-clustering-v2 D-13] information ceiling at this basis."

That is **not** the "we built an SBI classifier on Lyα feedback" outcome the user originally framed when re-opening the track. It is a narrower, more honest deliverable that the cascade evidence supports. The user should accept this SCOPING.md only if that outcome shape is the deliverable they want — not in the hope of a sharper "measurement" result, which the cascade gives moderate-to-strong reason to expect will NOT land. If the user prefers a different deliverable surface, the alternative re-open paths are: (a) explicit paper-author trigger on the pk-feedback-classifier [D-26] verdict as-is (no new compute), or (b) a fundamentally different observable family (e.g., 2D flux maps, transverse correlations, metal lines) with its own scoping pass and its own published-prior literature review — neither of which is what this SCOPING.md covers.
