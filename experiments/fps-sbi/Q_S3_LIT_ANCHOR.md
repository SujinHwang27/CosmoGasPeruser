# Q-S3 — External posterior-width anchor literature audit

**Track:** fps-sbi (scoping pass; LEDGER not yet open)
**Branch:** `exp/pk-feedback-classifier` (audit authored here per SCOPING.md gate 3)
**Owner:** support-researcher
**Web tools used:** WebSearch + WebFetch (20+ calls), per SCOPING.md §2 Q-S3 mandate
**Audit date:** 2026-06-02
**Linked decisions:** `experiments/pk-feedback-classifier/LEDGER.md` [D-12], [D-14], [D-15], [D-17] / `experiments/fps-sbi/SCOPING.md` §2 Q-S3

---

## §1 — Anchor B search (5 priority candidates + cross-cut)

The "Maitra+2026 CAMELS-SBI" reference inherited from pk-feedback-classifier [D-12] Sub-deliverable A is a **citation-attribution error** in the prior LEDGER. The actual paper is:

> **Sinigaglia, Iglesias-Navarro, Viel (2026)**, *"Simulation-based inference from the Lyman-alpha forest 1D power spectrum with CAMELS"*, arXiv:2603.13011 (PRD submitted; v1 March 2026, v2 May 2026).

This is the only published SBI-on-P1D-with-explicit-AGN-feedback-parameters paper in the search space. The remaining four candidates are NOT SBI work. Detail per candidate:

### Candidate 1 (Anchor B principal): Sinigaglia, Iglesias-Navarro, Viel — arXiv:2603.13011

- **Methodology:** Normalizing flow → neural posterior estimation on Lyα forest P1D from CAMELS Latin Hypercube (LH) suite, IllustrisTNG **and** SIMBA models, 1000 simulations per model.
- **Parameters inferred:** 2 cosmological (Ω_m, σ_8) + 4 astrophysical feedback (**A_SN1, A_SN2, A_AGN1, A_AGN2** — the CAMELS standard set).
- **Redshifts:** z ∈ {2.00, 2.15, 2.30, 2.46, 2.63, 2.80, 3.01, 3.49}.
- **Quantitative posterior-width result on feedback parameters (verbatim):**
  - Abstract: *"the astrophysical parameters are generally unconstrained due to the limited probed volume."*
  - Body / Appendix C, Fig. 15: *"the four astrophysical parameters have broad posteriors, spanning the whole ranges covered by the priors."*
  - Appendix C / Fig. 16 (k_max = 9.0 h Mpc⁻¹, most aggressive configuration): predicted values "scattered randomly across the full prior ranges with no correlation to true values"; *"the networks seem to be insensitive to the values of the astrophysical parameters."*
  - Root cause stated by the authors: *"cosmic variance effects, which dominate over the specific impact of any variations in the astrophysical parameters"* given the (25 h⁻¹ Mpc)³ CAMELS box.
- **Cosmological constraints (for scale calibration):** Ω_m and σ_8 within 10% of true in ≥75% and ≥90% of test cases respectively.
- **Anchor-B applicability:** YES — directly comparable methodology (NPE-on-P1D), directly comparable parameter family (AGN feedback knobs). The numeric "posterior-width" is the **prior-recovered (degenerate) result** — i.e., the *expected null* in the [D-37] honest-reading framing.

### Candidate 2: Tillman, Burkhart, Tonnesen, Bird, Bryan (2024) — arXiv:2410.05383

- **Methodology:** Direct simulation-comparison study (Simba variants), NOT SBI / not parameter regression.
- **Posterior widths on AGN feedback parameters:** None reported.
- **What it reports (Anchor A territory):** ΔP_F(k) magnitudes: 2% effect for k < 5×10⁻² s/km, 8% for k > 5×10⁻² s/km at z > 2.0; effects span 1.5×10⁻³ < k < 10⁻¹ s/km; "most dramatic at z < 1." Status: ApJ accepted; v2 Feb 2025.
- **Anchor-B applicability:** NO. Already confirmed in pk-feedback-classifier [D-14] Sub-C as Anchor-A material.

### Candidate 3: Pirecki, Tillman, Burkhart, Tonnesen, Bird (2025) — arXiv:2509.18260

- **Methodology:** CAMELS-SIMBA AGN feedback **sensitivity study** (5-parameter sweep of A_AGN1, A_AGN2, BHRadiativeEff, BHJetTvirVel, BHJetMassThr). Qualitative ΔP1D-per-parameter, **NOT SBI / not posterior estimation.**
- **Redshifts:** z ∈ {0.1, 0.4, 0.7, 1.0, 1.25, 2.0} — relevant low-z range, but the analysis is sensitivity-only.
- **Quantitative results (sensitivity, not posterior width):** A_AGN2 decrease → 50–100% P1D enhancement by z = 0.1; A_AGN1 decrease → up to 80% P1D enhancement at z = 0.1; BHJetMassThr increase → 50–100% changes at z = 0.1; BHRadiativeEff and BHJetTvirVel cross 10% only at z ≤ 0.4. **No posterior widths, no Fisher matrices, no σ-constraints.**
- **Status:** Accepted ApJ; v2 May 2026.
- **Anchor-B applicability:** NO (sensitivity, not SBI posterior). Useful as a supporting low-z signal-existence cross-check (sibling to Anchor A) — at z = 0.1 the parameter knobs produce 10–100% ΔP1D, which is a strong signal-existence direction confirmation at low z extrapolatable toward z = 0.3.

### Candidate 4: "Wadekar 2023 LyαNNA" — citation error; actual = Nayak, Walther, Gruen, Adiraju (2024) — arXiv:2311.02167

- **Methodology:** ResNet field-level regression on Lyα forest spectra (Nyx sims) for **thermal IGM parameters (T_0, γ) only**.
- **Redshift:** z = 2.2.
- **Posterior widths on AGN feedback parameters:** Not applicable — AGN feedback parameters are NOT in the inferred set. Constrains thermal-evolution parameters only.
- **Anchor-B applicability:** NO. (LEDGER [D-12] / SCOPING.md candidate-list attribution to "Wadekar 2023" is a citation error; the LyαNNA author is Parth Nayak. The "Wadekar" name in pk-feedback-classifier [D-12] / [D-14] should be corrected if it propagates into the fps-sbi LEDGER, per the [D-14] Walther-precedent metadata-correction discipline.)
- **Sequel note:** A LyαNNA-II paper exists (arXiv 2510.19899, "field-level inference with noisy spectra"). Spot-check: same thermal-parameter scope; not feedback-parameter.

### Candidate 5: Cabayol-Garcia, Chaves-Montero, Font-Ribera, Pedersen (2023) — arXiv:2305.19064

- **Methodology:** Neural network **emulator** for P1D (not SBI). Parameterized in (A_p, n_p) linear-matter-power amplitude/slope, not in feedback parameters.
- **Redshift:** z = 2 to 4.5.
- **Posterior widths on AGN feedback parameters:** None — feedback parameters are NOT in the emulation basis. The emulator does not constrain AGN feedback at all.
- **Anchor-B applicability:** NO.

### Cross-cut search (broader queries, 5+ additional WebSearch calls)

Additional searches for: "simulation-based inference Lyα forest AGN feedback posterior σ", "neural posterior estimation P1D Lyman alpha", "normalizing flow OR neural posterior Lyman alpha forest AGN feedback Fisher constraint 2024 2025 2026", "low-z z=0.3 Lyα forest AGN feedback inference posterior σ 2025 2026", "SBI feedback parameter σ posterior IGM gas low redshift power spectrum 2026", '"Maitra" CAMELS Lyα forest SBI'.

Cross-cut returns surfaced these additional adjacent papers — none of which provide a posterior-width-on-feedback-parameter anchor:

- **Ho, Qezlou, Bird, Yang, Avestruz, Fernandez, Iršič (2025)** — *"Small-scale Lyα forest cosmology with PRIYA"*, arXiv:2509.18271. Inference target is (A_P, n_P) primordial-power-spectrum + IGM thermal-history; **no AGN feedback parameter inference.** z = 2 to 5.
- **Khaire+2024** (arXiv:2306.05466) — direct simulation-comparison; no SBI.
- **Tillman+2023** (arXiv:2210.02467) — direct simulation-comparison; no SBI.
- **Burkhart+2022** (arXiv:2204.09712) — direct simulation-comparison + multi-statistic exploration; no SBI.
- **No "Maitra" CAMELS-SBI-on-Lyα paper exists.** The citation in pk-feedback-classifier [D-12] is the only-such-paper-in-the-field but mis-attributed in name.

---

## §2 — Anchor B verdict

**ANCHOR B FOUND — but as a NULL-style upper-bound bar, NOT a target σ-width.**

Single-paper anchor: **Sinigaglia, Iglesias-Navarro, Viel (2026)**, arXiv:2603.13011 (PRD submitted), Lyα forest P1D + CAMELS LH + NPE → AGN feedback posteriors "broad, spanning the whole ranges covered by the priors" (= **prior-recovered, posterior width ≈ prior width**) for all four CAMELS astrophysical knobs (A_SN1, A_SN2, A_AGN1, A_AGN2) at z = 2.0–3.5.

### The bar (rule-14(i)-clean construction):

> **External-anchor bar (Anchor B):** A successor SBI-on-P_F(k) measurement of AGN-feedback-related parameters from a small-box (≲ 25 h⁻¹ Mpc)³ training suite should **NOT report posterior widths < 50% of the prior range on any AGN feedback parameter**, because Sinigaglia+2026's CAMELS-LH (same box scale, larger N_sim = 1000) explicitly demonstrated that cosmic-variance noise on the (25 h⁻¹ Mpc)³ box dominates parameter sensitivity and produces prior-width posteriors. A narrower posterior in our setup (Sherwood-trained, fewer effective samples, smaller-effective-volume-per-recipe, z = 0.3 NOT z = 2–3.5) would be **a-priori suspicious** — most likely a normalizing-flow overconfidence pathology (the literature default failure mode for NPE), not a genuine information gain.

This is a **rule-14(i)-clean external bar in the inverted direction** the SCOPING.md §2 Q-S3 contemplated. It does NOT give us a numeric width-to-clear. It gives us a numeric width-NOT-to-clear-without-additional-evidence (overconfidence flag). Per SCOPING.md §5's expected-outcome shape, this is operationally exactly what is needed: the cascade-inherited expected result is "posterior ≈ prior on per-sightline P_F(k); posterior narrower than prior only on the C4-vs-others axis at large M." Sinigaglia+2026 gives us a **pre-committed expectation** that even with 1000 LH training sims at the canonical CAMELS scale, the feedback posterior is prior-recovered — so our (smaller-effective-suite, 4-discrete-recipe) Sherwood result that recovers the prior on the C2/C3 axis is **consistent with published**, and a sub-prior-width posterior on the C2/C3 axis would be the result that demands external verification before any "measurement" verb.

### Caveats / honest limits on Anchor B (per [D-37], [D-15] hedging discipline):

1. **Redshift mismatch.** Sinigaglia+2026 is z = 2.0–3.5; our target is z ≈ 0.3. Per pk-feedback-classifier [D-15], z-extrapolation from published z-ranges to z = 0.3 must be hedged on the face of every claim ("extrapolating from CAMELS-LH z = 2–3.5 to z = 0.3"). Pirecki+2025 (z = 0.1–2.0, sensitivity-only) plus Tillman+2024 ("most dramatic at z < 1") give *direction-consistent* evidence that the feedback signal grows at low z, which means our z = 0.3 posterior **could plausibly be narrower than CAMELS-LH at z = 2–3.5**. The Anchor B bar therefore should be read as a *moderate* not *strict* overconfidence flag.
2. **Suite-architecture mismatch.** Sinigaglia+2026 has continuous-parameter LH (interpolation in θ-space, 1000 sims). Sherwood at z = 0.3 is discrete-recipe (Q-S1 owner). If Q-S1 returns "discrete-only", the rule-14(i) bar mechanics partly degrade — discrete-likelihood-ratio over 4 recipes has different "posterior width" semantics than continuous-θ NPE. The Anchor B bar still applies in the "prior recovered = uninformative" reading direction, but a posterior-width-in-θ-units comparison stops being a direct apples-to-apples.
3. **No σ-numeric.** Sinigaglia+2026's report is qualitative ("broad, spans the prior") not numeric (e.g., "σ(A_AGN1)/σ_prior(A_AGN1) ≈ 0.95 at 68%"). The bar therefore lives at the *prior-recovered vs prior-narrowed* binary, not at a numeric posterior-width threshold. A more numeric bar would require Sinigaglia+2026 appendix-figure-extraction by reading their published rank statistic / posterior-shrinkage statistic, which their abstract does not promise and our literature pass did not surface verbatim. The most defensible numeric statement: *"posterior 68%-width / prior 68%-width ≥ 0.8 is consistent with Sinigaglia+2026; a value < 0.5 is inconsistent and demands a normalizing-flow-overconfidence diagnostic."*

### Rule-14 routing per SCOPING.md §2 Q-S3:

- Anchor B status: **FOUND (qualitative, NULL-style upper-bound).** This is sufficient for rule-14(i)-clean external-anchor framing — the bar is published, externally derived, applicable to the same observable family + same parameter family.
- **NOT triggered:** rule-14(ii) pre-committed-process-failure construction (would have been triggered by a literal NULL on Anchor B literature).
- **NOT triggered:** rule-14(iii) deliverable demotion to adjacent-finding-only at scoping (the pk-feedback-classifier [D-17] precedent path). The bar exists, so the deliverable can carry a "measurement" verb of the form *"posterior-width measurement consistent / inconsistent with published"* — the SCOPING.md §1 verbed-ceiling allows this verb.
- **Recommended verb-ceiling for Stage 2 headline** (refining SCOPING.md §1):
  > "First SBI-on-FPS posterior-width measurement of Sherwood-feedback-class distinguishability at z ≈ 0.3, with posterior 68%-width / prior 68%-width compared against the Sinigaglia+2026 CAMELS-LH null benchmark of ≈ 1 (prior-recovered) on the analogous AGN feedback parameters."

---

## §3 — Anchor A confirmation (Khaire+2024, Tillman+2024)

Per pk-feedback-classifier [D-14] Sub-deliverable C / [D-15] verb amendment, the Anchor A numerics were already web-verified. This audit independently re-confirms (re-fetching arXiv abstracts + the MNRAS Oxford-Academic article page):

### Khaire, Hu, Hennawi, Walther, Davies (2024)

- **arXiv:** 2306.05466 (submitted June 2023; revised November 2023)
- **Journal:** MNRAS, Volume 527, Issue 3 (January 2024), pages 4545–4562, DOI stad3374 — **re-confirmed via Oxford Academic**.
- **Title:** *"Can the Low Redshift Lyman Alpha Forest Constrain AGN Feedback Models?"*
- **Redshift:** z = 0.1 (Illustris vs IllustrisTNG comparison; shared initial conditions).
- **Verbatim ΔP_F(k) numeric (Oxford Academic / MNRAS body):**
  > *"The difference between the two simulations is approximately 5 per cent for k < 0.02 s km⁻¹ and increases monotonically at higher k-values."*
  > *"only at k > 0.04 s km⁻¹ is the difference in the simulations more than the measurement uncertainties."*
  > *"they begin to diverge at smaller scales (k > 0.02 s km⁻¹) where Illustris exhibits higher power than IllustrisTNG."*

  **Refinement vs the pk-feedback-classifier [D-14] / SCOPING.md transcription:** SCOPING.md §2 Q-S3 quotes "≥ 5% difference at k > 0.04 s/km" and "5% at k > 0.02 s/km" — the MNRAS-verified text actually says "approximately 5% for k < 0.02 s km⁻¹" (i.e., the floor of the divergence at large scales) and "increases monotonically at higher k-values," with detectability above measurement uncertainty only above k > 0.04 s km⁻¹. The "5%" is therefore the **floor at low-k**, not a number at k > 0.04 s/km. **This is a refinement of the pk-feedback-classifier [D-14] transcription** — recommend the fps-sbi LEDGER §3 [D-01] (when authored) re-quote the verified text directly rather than the SCOPING.md paraphrase. The directional claim (5–8% at our empirical k = 0.06–0.13 s/km peak) is *strengthened*, not weakened, by the verified text (the actual divergence is bigger than 5% at our peak).

- **SBI / posterior widths:** **No**, this paper does NOT perform SBI; it is a forward simulation-comparison study. Anchor A is signal-existence-direction only, NOT a posterior-width anchor — as SCOPING.md §2 Q-S3 already noted.

### Tillman, Burkhart, Tonnesen, Bird, Bryan (2024)

- **arXiv:** 2410.05383 (submitted October 2024; revised February 2025).
- **Journal:** ApJ accepted.
- **Title:** *"The Effects of AGN Feedback on the Lyman-α Forest Flux Power Spectrum"*.
- **Redshift range:** z = 0.1–2.0 examined, with explicit z > 2.0 and z < 1.0 contrasts.
- **Verbatim ΔP_F(k) numeric (re-confirmed via arXiv abstract):**
  > *"At higher redshifts (z > 2.0), AGN feedback has a 2% effect on the P1D for k < 5×10⁻² s/km and an 8% effect for k > 5×10⁻² s/km."*
  > Effects span the wavenumber range 1.5×10⁻³ < k < 10⁻¹ s/km. "Most dramatic at z < 1."
- **SBI / posterior widths:** **No**, this paper does NOT perform SBI. Anchor A only.

### Anchor A status:

**CONFIRMED, with a minor [D-14] transcription refinement on Khaire.** Anchor A remains the signal-existence cross-check (NOT a posterior-width bar), per SCOPING.md §2 Q-S3's design.

---

## §4 — Implications for Stage 2 SBI framing (honest [D-37] reading)

### What the audit returned:

- **Anchor B: FOUND (single paper, qualitative).** Sinigaglia, Iglesias-Navarro, Viel (2026), arXiv:2603.13011, gives a published-prior posterior-width benchmark on AGN-feedback parameters (broad/prior-recovered) at z = 2–3.5 on the analogous methodology (NPE on P1D from CAMELS-LH). It is rule-14(i)-clean.
- **Anchor A: CONFIRMED.** Khaire+2024 (MNRAS 527, 4545) and Tillman+2024 (ApJ accepted) ΔP_F(k) numbers stand as signal-existence direction cross-check.

### Framing implication for Stage 2 spec ([D-02]+ when the fps-sbi LEDGER opens):

**Rule-14(i)-clean, deliverable NOT demoted to adjacent-finding at scoping.** The Stage 2 deliverable surface admits a "posterior-width measurement" verb (not a "constrain" / "recover" / "measure feedback recipes" verb), constructed as:

> "Posterior 68%-width / prior 68%-width on Sherwood-feedback-class distinguishability from stacked P_F(k) at z ≈ 0.3, on the C4-vs-others axis (where the [D-26] cascade gives signal-existence direction) and on the C2-vs-C3 axis (where the cascade gives null direction), compared against the Sinigaglia+2026 CAMELS-LH benchmark of ≈ 1 (prior-recovered) at z = 2–3.5."

### Cascade-inherited verb ceiling (rule 2 + SCOPING.md §1) holds:

- **Admissible:** "first SBI-on-P_F(k) posterior-width measurement at z ≈ 0.3", "distinguishability quantification", "first test of", "posterior consistent / inconsistent with Sinigaglia+2026 prior-recovered benchmark".
- **NOT admissible:** "measure feedback parameters from Lyα forest", "constrain AGN feedback efficiency", "SBI recovers feedback recipes from P_F(k)".

### Expected-outcome shape per [D-37] (refined from SCOPING.md §5):

The most-likely Stage 2 result given the cascade and the Sinigaglia+2026 prior is:

- Per-sightline P_F(k): posterior-width / prior-width ≈ 1 on all axes (= Sinigaglia consistent + [D-13] / [D-26] M=1 chance-level result re-confirmed in posterior-width units).
- Stacked P_F(k) at M = 64: posterior-width / prior-width substantially < 1 on C4-vs-others axis (= signal-existence direction confirmed); ≈ 1 on C2-vs-C3 axis (= Sinigaglia consistent + [D-26] G4 FAIL re-confirmed in posterior-width units).

This is publishable as a *quantified statement of the cascade-exhaustion outcome* — narrower-and-honest than the user's original re-open framing, exactly as SCOPING.md §5 warned.

### Honest-reading flag on Anchor B's qualitative-only character:

The bar lives at *prior-recovered (≈ 1) vs prior-narrowed (< 1)* — it is binary-ish, not a sharp numeric σ-threshold. A Stage 2 spec should pre-commit a specific numeric cut (e.g., posterior-68%-width / prior-68%-width < 0.5 = "narrowed beyond Sinigaglia null"; ≥ 0.8 = "consistent with Sinigaglia null"; 0.5–0.8 = "intermediate, ambiguous"). This pre-commit is a rule-14(ii) self-anchored *threshold* on an externally-anchored *direction* — defensible because the direction (prior-recovered) is externally fixed by Sinigaglia+2026, and only the numeric threshold is project-internal. PI should adjudicate the exact threshold at spec authorship.

### What this audit does NOT do:

- Does NOT change the SCOPING.md §1 cascade-inherited verb ceiling.
- Does NOT lift the Q-S1 BLOCKING gate (Sherwood parameter-space audit — `data-engineer` dispatch).
- Does NOT make a Q-S2 decision (observable choice; defense-panel + PI on Q-S1/Q-S3 returns).
- Does NOT trigger paper-author (user-triggered-only, per CLAUDE.md / PI binding rule).
- Does NOT open the fps-sbi LEDGER (`exp/fps-sbi` branch not created by this audit).

---

## §5 — Citations (arXiv IDs verified against arXiv directly + Oxford Academic for the MNRAS Khaire paper)

| Bibkey | Authors | Title (short) | arXiv | Journal | Year | Role in audit |
|:---|:---|:---|:---|:---|:---|:---|
| Sinigaglia+2026 | F. Sinigaglia, P. Iglesias-Navarro, M. Viel | SBI from Lyα P1D with CAMELS | 2603.13011 | PRD submitted | 2026 | **Anchor B principal** |
| Tillman+2024 | M. T. Tillman, B. Burkhart, S. Tonnesen, S. Bird, G. L. Bryan | Effects of AGN feedback on Lyα forest flux power spectrum | 2410.05383 | ApJ accepted | 2024 (v2 2025) | Anchor A (Δ-magnitude direction) |
| Khaire+2024 | V. Khaire, T. Hu, J. F. Hennawi, M. Walther, F. Davies | Can low-z Lyα forest constrain AGN feedback? | 2306.05466 | MNRAS 527, 4545 | 2024 | Anchor A (Δ-magnitude direction) |
| Pirecki+2025 | M. Pirecki, M. T. Tillman, B. Burkhart, S. Tonnesen, S. Bird | CAMELS-SIMBA AGN feedback variations on Lyα P1D | 2509.18260 | ApJ accepted | 2025 (v2 2026) | Low-z signal-existence supporting evidence; NOT Anchor B (sensitivity-only) |
| Nayak+2024 (NOT Wadekar) | P. Nayak, M. Walther, D. Gruen, S. Adiraju | LyαNNA: deep-learning field-level inference for Lyα forest | 2311.02167 | A&A 689, A153 | 2024 | NOT Anchor B (thermal-parameter inference, not feedback) |
| Cabayol-Garcia+2023 | L. Cabayol-Garcia, J. Chaves-Montero, A. Font-Ribera, C. Pedersen | NN emulator for Lyα P1D | 2305.19064 | MNRAS (stad2512) | 2023 | NOT Anchor B (cosmological-parameter emulator, not feedback inference) |
| Ho+2025 | M.-F. Ho, M. Qezlou, S. Bird, Y. Yang, C. Avestruz, M. A. Fernandez, V. Iršič | Small-scale Lyα cosmology with PRIYA | 2509.18271 | (not yet) | 2025 | NOT Anchor B (primordial-power + thermal inference; no AGN feedback) |
| Tillman+2023 | M. T. Tillman, B. Burkhart, et al. | Efficient long-range AGN feedback affects low-z Lyα forest | 2210.02467 | ApJL 945, L17 | 2023 | Anchor-A supporting (low-z line-broadening / CDDF mechanism, per [D-14]) |
| Burkhart+2022 | B. Burkhart et al. | Low-z Lyα forest as a constraint for models of AGN feedback | 2204.09712 | (MNRAS) | 2022 | Background (P1D-as-best-feedback-probe identification, per Cabayol-Garcia citation chain) |

### Citation-error corrections to propagate forward (per [D-14] Walther-metadata-correction discipline):

1. **"Maitra+2026 CAMELS-SBI"** — does not exist as cited in pk-feedback-classifier [D-12] Sub-deliverable A and propagated into SCOPING.md §2 Q-S3 priority candidate list. The actual paper is **Sinigaglia, Iglesias-Navarro, Viel (2026)** arXiv:2603.13011. When the fps-sbi LEDGER opens, [D-01] should record this correction explicitly per the [D-14] precedent (Walther arXiv↔ApJ metadata correction).
2. **"Wadekar 2023 LyαNNA"** — Wadekar is not the lead author of LyαNNA. The paper is **Nayak, Walther, Gruen, Adiraju (2024)** A&A 689, A153 / arXiv:2311.02167. Spot-check vs pk-feedback-classifier [D-12]: same correction should propagate.

---

**Audit sign-off provenance:** support-researcher dispatch, 2026-06-02, on `exp/pk-feedback-classifier` branch (per SCOPING.md gate 3, parallel with `data-engineer` Q-S1 dispatch). Web tools: 20+ WebSearch + WebFetch calls, both arXiv abstracts and one MNRAS Oxford-Academic article page fetched and direct-quoted. Anchor B verdict: **FOUND (qualitative null-style bar)**, single paper, rule-14(i)-clean. Anchor A: **CONFIRMED** with a minor transcription refinement on the Khaire ΔP_F(k) percentage location (5% is the LOW-k floor, not the high-k value; the actual divergence at our k = 0.06–0.13 s/km peak band is above the 5% floor and above the measurement-uncertainty threshold). No paper-author dispatch; no fps-sbi LEDGER modification (track not yet open); no `data/` modification. This audit hands off to the PI for Stage 2 spec authorship (SCOPING.md gate 4), pending Q-S1 return and user acceptance of the SCOPING.md as a whole.
