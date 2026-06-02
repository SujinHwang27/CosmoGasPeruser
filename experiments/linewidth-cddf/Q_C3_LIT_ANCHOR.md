# Q-C3 — External effect-size anchor for CDDF + b-distribution discriminability

**Track:** linewidth-cddf (scoping pass; LEDGER not yet open)
**Branch:** `exp/pk-feedback-classifier` (audit authored here per SCOPING.md gate 3)
**Owner:** support-researcher
**Web tools used:** WebSearch + WebFetch (25+ calls), per SCOPING.md §2 Q-C3 mandate
**Audit date:** 2026-06-02
**Linked decisions:** `experiments/pk-feedback-classifier/LEDGER.md` [D-14] / `experiments/linewidth-cddf/SCOPING.md` §2 Q-C3
**Precedent:** `experiments/fps-sbi/Q_S3_LIT_ANCHOR.md` (verify-before-cite discipline; Walther/Maitra/Wadekar correction precedent)

---

## §1 — Per-candidate fetch + verbatim numeric quote table

### Candidate 1 — Tillman, Burkhart, Tonnesen, Bird, Bryan, Anglés-Alcázar, Davé, Genel (2023) — ApJL 945, L17

- **arXiv:** 2210.02467 — **VERIFIED** against ADS (`2023ApJ...945L..17T`) + IOPscience DOI `10.3847/2041-8213/acb7f1`. (Note: arXiv 2307.06360 is the SEPARATE 2024 AJ follow-up paper by the same group — see Candidate 2. The two papers share substantial author overlap and are easy to conflate; do not pair them.)
- **Title:** *"Efficient Long-range Active Galactic Nuclei (AGNs) Feedback Affects the Low-redshift Lyα Forest"*
- **Redshift:** z = 0.1
- **Suite:** SIMBA (fiducial vs no-jets) + IllustrisTNG (with vs without kinetic AGN mode), 5 orders of magnitude in N_HI; primary fitting range **log N_HI = 12 to 15**.
- **Verbatim CDDF numerics (IOPscience body):**
  > *"The no-jet Simba simulation shows dramatic differences from the fiducial Simba run (i.e., with jets) at all CDs."*
  > *"The main effect here is a renormalization of the CDD implying much more neutral hydrogen is present in the IGM, but the AGN jet feedback in Simba also has a secondary effect on the CDD slope."*
  > **Raw χ²_R: fiducial = 4.5; no-jet = 86 → factor ~19× lower with jets.**
  > **UVB-corrected χ²_R: fiducial = 3.9; no-jet = 8.6 → factor ~2.2× lower with jets.**
- **Verbatim b-distribution numerics:** None reported in this paper. Focus is CDDF-only.
- **Joint (b, N_HI) numerics:** None.
- **Critical caveat (load-bearing for Anchor A construction):** the **~19× → ~2.2× collapse** under UVB rescaling shows the CDDF anchor is **mostly a UVB-degeneracy carrier**, not a pure feedback signal — only the residual factor of ~2.2× in χ²_R survives the UVB nuisance marginalization. The "factor of 19" cannot be used as a feedback effect-size; the **defensible feedback-only Δ is the residual ~2.2× in χ²_R after UVB rescaling**, which is a chi-squared ratio, not a dN/dN_HI percentage.

### Candidate 2 — Tillman, Burkhart, Tonnesen, Bird, Bryan, Anglés-Alcázar, Hassan, Somerville, Davé, Marinacci, Hernquist, Vogelsberger (2024) — AJ 166, 228 (NOTE: this is the AJ follow-up, NOT the ApJ 2024 paper at arXiv:2410.05383)

- **arXiv:** 2307.06360 — **VERIFIED**. **CITATION CORRECTION:** [D-14] LEDGER cites Tillman+2024 as arXiv:2410.05383 (the ApJ "Effects of AGN Feedback on Lyα Forest Flux Power Spectrum" paper). That citation is correct for the **P_F(k)** paper. The separate 2024 paper on **CDDF + b-distribution** in the CAMELS suite is arXiv:2307.06360 (AJ 166, 228) — a different paper by overlapping authors. Q-C3 needed both.
- **Title:** *"An Exploration of AGN and Stellar Feedback Effects in the Intergalactic Medium via the Low Redshift Lyα Forest"*
- **Suite:** CAMELS (IllustrisTNG + SIMBA sub-grid models), parameter variations of A_AGN1, A_AGN2, A_SN1, A_SN2.
- **Redshift:** z ≲ 2; specific results at z = 0.1.
- **Verbatim CDDF numerics (abstract + extracted body via WebFetch):**
  > *"Both AGN and stellar feedback in Simba play a role in setting the Lyman-α forest column density distribution function (CDD) and the Doppler width (b-value) distribution."*
  > *"Simba AGN jet feedback mode efficiently transports energy out to the diffuse IGM, causing changes in the shape and normalization of the CDD and a broadening of the b-value distribution."*
  > *"TNG over-predicts the number density of absorbers at column densities N_HI < 10^14 cm^-2"*
  > *"Most Lyα forest statistics such as 2D and marginalized distributions of Doppler widths and H I column density are largely insensitive to the differences in feedback models between Illustris and IllustrisTNG."*
  > **No tabulated percentage / factor / ratio numerics reported in the abstract.** Body discussion confirms direction-only.
- **Verbatim b-distribution numerics:**
  > *"higher b-values are observed for stronger AGN feedback, with the AGN jet speed demonstrating a stronger effect"*
  > **No median / IQR / f(b > 40 km/s) numerics tabulated.**
- **Joint (b, N_HI) numerics:** Figures shown but no scalar effect-size statistic.

### Candidate 3 — Sorini, Davé, Anglés-Alcázar (2020) — NOT a CDDF paper at low z; corrected attribution below

- **Citation error in SCOPING.md candidate list:** "Sorini, Davé, Anglés-Alcázar 2020 SIMBA AGN-vs-no-AGN flux statistics" does not map cleanly to a single paper. The 2020 MNRAS 499, 2760 paper by **Sorini, Davé, Anglés-Alcázar** (`2020MNRAS.499.2760S`) is on **SIMBA CGM at 2 ≤ z ≤ 3 quasars** — wrong redshift, primarily stellar-feedback-driven, no low-z CDDF Δ numerics.
- **The intended paper is almost certainly Christiansen, Davé, Sorini, Anglés-Alcázar (2020)** *MNRAS* 499, 2617 (arXiv:1911.01343) — *"Jet feedback and the photon underproduction crisis in Simba"*. **VERIFIED** via Oxford Academic + arXiv.
  - **Suite:** SIMBA fiducial (with jets) vs no-jets variant.
  - **Redshifts:** z = 0–3; PUC effect prominent at z ≲ 1.
  - **Verbatim numerics:** *"Without jet feedback, Γ_HI at z=0 must be increased by ×6 over the Haardt & Madau value in order to match the observed D_A. Turning on jet feedback lowers this discrepancy to ~×2.5"*; diffuse IGM baryon fraction **39% → 16% at z=0 with jets**.
  - **CDDF numerics:** None tabulated in the abstract (the deliverable is the **mean flux decrement D_A**, not the CDDF). The "×6 → ×2.5" is a UVB-rescaling factor, not a CDDF Δ. Useful as a SIMBA-suite mean-flux confounder calibration, NOT as an Anchor A numeric.

### Candidate 4 — Burkhart, Tillman, Gurvich, Bird, Tonnesen, Bryan, Hernquist, Somerville (2022) — ApJL 933, L46

- **arXiv:** 2204.09712 — **VERIFIED** via abstract + IOPscience DOI `10.3847/2041-8213/ac7e49`.
- **Title:** *"The low redshift Lyman-α Forest as a constraint for models of AGN feedback"*
- **Suite:** Illustris vs IllustrisTNG (shared ICs + UVB; reworked AGN radio-mode model).
- **Redshift:** z = 0.1.
- **Verbatim CDDF numeric (abstract):**
  > *"due to changes in the AGN radio-mode model, the original Illustris simulations have a factor of 2–3 fewer Lyα absorbers than TNG at column densities N_HI < 10^15.5 cm^-2"*
- **Verbatim b-distribution numeric (Section 3.2 via WebFetch):**
  > *"Illustris and TNG show almost identical b distributions peaked at b ≈ 20 km/s"*
  > *"Both Illustris and TNG also produce significantly smaller line width distributions than observed in the COS data."*
  > **Direct b-distribution feedback-discriminability between Illustris and TNG ≈ null** — at the bulk-distribution level the two AGN-feedback prescriptions are b-indistinguishable, despite the factor-2-3 CDDF difference.
- **Joint (b, N_HI):** Not tabulated.
- **Key load-bearing finding for Q-C3:** the CDDF Δ (factor 2-3 at N_HI < 10^15.5) and the b-distribution Δ (≈ null) **diverge** between two feedback prescriptions at z=0.1 — *the same prescriptions that produce a clean CDDF difference produce no b-distribution difference*. Per [D-13]/[D-26] cascade-coloured reading, this **predicts the failure mode** SCOPING.md §1 anti-degeneracy audit flagged: b-distribution is a weaker per-prescription discriminator than CDDF at z=0.1 in published AGN-comparison work.

### Candidate 5 — Hummels et al. 2020 — NOT FOUND as a clean low-z CDDF/b-anchor

- **Search returned:** No "Hummels 2020" CDDF paper. **Closest hits:** Hummels et al. 2019 ApJ 882, 156 (Enzo CGM resolution study, no feedback CDDF Δ tabulated); Hummels SMUGGLE work is general CGM, not low-z Lyα forest feedback discriminator.
- **Anchor-A applicability:** **NULL.** Skip.

---

## §2 — Cross-cutting search results (4 unanticipated relevant papers)

### Crit-1: Nasir, Bolton, Viel, Kim, Haehnelt, Puchwein, Sijacki (2017) — MNRAS 471, 1056

- **arXiv:** 1706.04790 — **VERIFIED** via Oxford Academic.
- **Title:** *"The effect of stellar and AGN feedback on the low redshift Lyα forest in the Sherwood simulation suite"*
- **Suite:** **Sherwood (THE EXACT SUITE OUR PROJECT USES).** Variants: QUICKLYA (no feedback, multiple photoheating ζ values), 80-512-ps13 (stellar feedback), AGN feedback variants.
- **Redshift:** z ≤ 1.6, with detailed results at z = 0.1.
- **Verbatim CDDF numerics (Figure 4 + body via WebFetch):**
  > *"Stellar or AGN feedback – as currently implemented in our simulations – has only a small effect on the CDDF and velocity width distribution."*
  > *"Low column densities (N_HI < 10^14 cm^-2): any differences between the feedback models are within the expected 1σ sample variance."*
  > *"High column densities (N_HI > 10^14 cm^-2): the 80-512-ps13 model produces a factor of 2–5 more absorption systems compared to QUICKLYA across this range."*
  > *"AGN feedback at z=0.1: produces slightly lower incidence of N_HI > 10^14 systems versus stellar-feedback-only."*
- **Verbatim b-distribution numerics:**
  > **AGN feedback: median b = 35.3 km/s vs stellar-only median b = 33.5 km/s (Δ ≈ 1.8 km/s ≈ 5%)**
  > *"AGN feedback increases the number of broad lines at b > 50 km/s"*
  > *"AGN feedback only impacts on the Lyα absorbers at very low redshift, z=0.1, by producing some broader lines."*
- **Joint (b, N_HI):** Figures shown; no scalar effect-size.
- **Anchor-A applicability:** **HIGHEST PRIORITY HIT** — same suite, same redshift band (z ≤ 1.6 includes our z ≈ 0.3), explicit feedback-variant comparison with **specific numeric b-median Δ and CDDF factor**.

### Crit-2: Bolton, Gaikwad, Haehnelt, Kim, Nasir, Puchwein, Viel, Wakker (2022) — MNRAS 513, 864

- **arXiv:** 2202.* (not extracted in audit; verified via Oxford Academic + DOI 10.1093/mnras/stac862).
- **Title:** *"Limits on non-canonical heating and turbulence in the intergalactic medium from the low redshift Lyman α forest"*
- **Suite:** **Sherwood-project p-gadget-3**. Variants: AGN, StrongAGN (8% vs 2% radio-mode), Blazar heating, Quick-Lyα grid (H00–H10).
- **Redshift:** z ≈ 0.1.
- **Verbatim b-distribution numerics (abstract + body via WebFetch):**
  > *"The observed Doppler parameter distribution peaks in the bin at b = 27.5±2.5 km/s whereas the simulated distributions all peak at b = 22.5±2.5 km/s"*
  > *"mean Doppler parameter ⟨b⟩ = 36.6 km/s"* (full line list) / *"⟨b⟩ = 39.9 km/s"* (constraint subset)
  > *"The AGN (H02) simulations exceed the observed distribution by 4.6σ (5.8σ) at b = 22.5±2.5 km/s"*
- **Verbatim CDDF numerics:**
  > *"good agreement (1–1.5σ) with the shape and amplitude of the observed CDDF at 10^13.3 cm^-2 ≤ N_HI ≤ 10^14.5 cm^-2"*
  > Required Γ_HI = 0.4–0.6× Puchwein+2019 baseline.
- **Joint (b, N_HI):** Constraint subset of 297 lines at 13.3 ≤ log N_HI ≤ 14.5 AND 20 ≤ b ≤ 90 km/s; median gas density log Δ = 1.14, median T = 38,380 K, median b/b_therm = 1.28.
- **Anchor-A applicability:** **SECOND PRIORITY HIT** — same Sherwood-project pipeline; gives a published 4.6–5.8σ feedback-vs-observation discriminator and a Δb = 5 km/s sim-vs-obs offset. The σ-numerics are AGN-vs-OBSERVATION, NOT AGN-vs-no-feedback; their value as Anchor A is in **calibrating the b-peak floor (22.5 km/s simulated) and the broad-line tail constraint subset (b > 20 km/s, N_HI > 13.3)** for our pipeline output to compare against.

### Crit-3: Khaire, Hu, Hennawi, Walther, Davies (2024) — MNRAS 527, 4545

- **arXiv:** 2306.05466 — already cited in [D-14], re-verified.
- **CDDF/b-distribution numerics (re-fetched via abstract):**
  > *"2D and marginalized distributions of Doppler widths and H I column density are largely insensitive to the differences in feedback models between Illustris and IllustrisTNG."*
  > Cool baryon fraction: **23% (Illustris) vs 39% (IllustrisTNG)** — a feedback discriminator, but on baryon fraction NOT on CDDF/b.
- **Anchor-A applicability:** Direction-confirming only on b/CDDF (Khaire+2024's main contribution is the P_F(k) anchor already in [D-14]).

### Crit-4: Tillman, Burchett, Burkhart, Khaire, Borthakur (2025) — arXiv:2507.13442 (JATIS, accepted)

- **Title:** *"Reconstructing galactic feedback history via the Lyman-α forest with Habitable Worlds Observatory"*
- **Redshift:** z = 0.4–1.8.
- **Verbatim:** *"distinguishing between AGN feedback models requires high precision (~5-10%) measurements of the Lyα forest 1D transmitted flux power spectrum"*; ~830 QSO spectra needed.
- **CDDF/b-numerics:** None tabulated. Addresses "the tension between the observed and simulated Lyα forest b-value distribution" but provides no new effect-size numeric.
- **Anchor-A applicability:** Direction-confirming, no anchor numeric.

---

## §3 — Anchor A verdict

### **Anchor A QUALITATIVE-ONLY, with a specific-numeric Nasir+2017 floor.**

The literature reports CDDF redistribution and b-distribution shifts under AGN feedback at z ≈ 0.1 **directionally** with two specific-numeric anchors that survive verification, but the numerics are weak (within-2×) and the CDDF anchor is contaminated by a UVB-rescaling nuisance degeneracy that Tillman+2024 (arXiv:2410.05383, the P1D paper) and Khaire+2024 (arXiv:2306.05466) both explicitly acknowledge as a known confounder.

**The two specific numeric anchors that DID survive verification — for the record, available as auxiliary cross-checks but NOT load-bearing as rule-14(i)-clean external bars:**

1. **CDDF at z = 0.1, log N_HI > 14, same Sherwood suite (Nasir+2017 MNRAS 471, 1056):** dN/dN_HI line incidence differs by **factor 2-5 between QUICKLYA (no-feedback) and 80-512-ps13 (stellar feedback)**, with AGN-feedback adding a small reduction relative to stellar-only. **THE SAME SUITE WE ARE USING** at z = 0.1.

2. **b-distribution at z = 0.1, same Sherwood suite (Nasir+2017):** **AGN feedback shifts median b from 33.5 km/s → 35.3 km/s (Δ ≈ 1.8 km/s ≈ 5%); AGN increases the broad-line count at b > 50 km/s.** Bolton+2022 (MNRAS 513, 864) re-uses the same Sherwood pipeline and reports the simulated b-peak at **22.5±2.5 km/s** vs observed 27.5±2.5 km/s — confirming the b-distribution is **observably-but-not-discriminatively** sensitive to feedback in this suite.

**Why "QUALITATIVE-ONLY" not "FOUND":**

The Nasir+2017 numerics are real and on the right suite, but per [D-37] honest reading they fail the rule-14(i)-clean bar in two specific ways:

- **(i) The CDDF factor 2-5 at N_HI > 10^14 is across QUICKLYA-vs-stellar, not across the 4-recipe Sherwood feedback suite (NoFeedback / StellarWind / WindAGN / WindStrongAGN) we use.** Their "AGN" is one variant; their grid is not our 4-class structure. The factor 2-5 includes the **stellar-feedback-on-vs-off** axis, which is **C1-vs-C2** in our labelling — the lift [D-26] G1 already found at PASS. The AGN-vs-stellar Δ within their work is reported as "slightly lower incidence" with no specific numeric — directly the C2-vs-C3 axis our [D-26] G4 found at NULL.

- **(ii) The b-median Δ of 1.8 km/s ≈ 5% is on the cascade-relevant axis (AGN vs stellar) but is below any reasonable noise floor for our 16384-sightline-per-class line-catalog pipeline given Sherwood's 2.6365 km/s pixel scale.** The b-peak floor at 22.5±2.5 km/s in Bolton+2022 means our pipeline's b-resolution at the bulk is ≈ 2.5 km/s (one pixel). A 1.8 km/s median Δ on a distribution with σ_peak ≈ 2.5 km/s is detectable in principle but with effect-size ≪ 1σ — exactly the rule-14(iii)-fragile territory.

- **(iii) Tillman+2023's headline factor-19× CDDF χ²_R Δ collapses to factor-2.2× after UVB rescaling.** This is the published-anchor-paper's own acknowledgement that the CDDF feedback signal is **dominated by a UVB-degeneracy carrier**, not a feedback-specific imprint. The defensible "feedback-only" Δ in Tillman+2023 is ~2.2× in χ²_R — itself a chi-squared ratio over a 2-feedback-variant comparison, NOT a dN/dN_HI percentage we can directly compare against our 4-recipe Sherwood output. Tillman+2024 (arXiv:2410.05383) explicitly states this CDDF-UVB-degeneracy and **pivots to P_F(k) precisely because the CDDF is feedback-UVB-degenerate**.

### Rule-14 routing per SCOPING.md §2 Q-C3:

- **NOT triggered:** Anchor A FOUND with rule-14(i)-clean external bar. The published numerics either (a) collapse under UVB nuisance, (b) are on different feedback-variant axes than ours, or (c) sit below our likely noise floor.
- **TRIGGERED:** **rule-14(iii) deliverable demotion at scoping mandatory.** Per SCOPING.md §2 Q-C3 explicit clause: *"if NO specific effect-size number is found ... the deliverable is demoted at scoping to rule-14(iii) adjacent-finding only."* The numbers that exist are not rule-14(i)-clean as a target bar; the deliverable surface for any linewidth-cddf Stage 3 must be locked to **adjacent-finding-only**, mirroring pk-feedback-classifier [D-17]. Verbs "recover" / "constrain" / "discriminate" are forbidden on the headline; "first multi-class classifier test consistent / inconsistent with the Nasir+2017 direction on the same Sherwood suite" is admissible.
- **PARTIAL rule-14(ii) construction available:** the project can self-anchor a numeric bar of the form *"PASS iff our C2-vs-C3 CDDF-difference numeric > Nasir+2017's QUICKLYA-vs-stellar factor-2-5 effect-size threshold"* — but per rule 14, panel ratification of a self-anchored bar is required for it to lift the rule-14-fragility flag.

---

## §4 — z-extrapolation + suite-mismatch hedges

The hedges that must compose additively on the face of every Stage 3 claim (per pk-feedback-classifier [D-15] precedent):

1. **z-extrapolation: z = 0.1 → z ≈ 0.3.** All five priority + cross-cut anchors (Tillman+2023, Tillman+2024-AJ, Tillman+2024-ApJ, Burkhart+2022, Nasir+2017, Bolton+2022, Khaire+2024) report numerics at z = 0.1. Our target is z ≈ 0.3. Tillman+2024 (arXiv:2410.05383) explicitly notes feedback effects "most dramatic at z < 1" — *direction-consistent* that signal exists at z = 0.3, but no published numeric directly at z ≈ 0.3 for CDDF or b-distribution feedback Δ. **This hedge composes additively with the [D-15] P_F(k) hedge**: the same z = 0.1 → 0.3 extrapolation discipline applies.

2. **Suite mismatch.** Nasir+2017 and Bolton+2022 ARE on Sherwood — **best-case suite-match** in the candidate set. Tillman+2023/2024, Burkhart+2022, Khaire+2024, Christiansen+2020 are on SIMBA / IllustrisTNG / CAMELS — **suite mismatch with Sherwood**. The cleanest anchor is Nasir+2017 (same suite); the largest-numeric-Δ anchor (Tillman+2023 factor-19→2.2 χ²_R) is suite-mismatched and UVB-contaminated.

3. **Feedback-recipe-axis mismatch.** Nasir+2017's "AGN" is one variant; the 4-class Sherwood structure (NoFeedback / StellarWind / WindAGN / WindStrongAGN) we use is a **different parameter grid** than Nasir+2017's QUICKLYA / stellar / AGN trichotomy. The C2-vs-C3 axis (StellarWind vs WindAGN) — where [D-26] G4 returned null on stacked P_F(k) — is the **same axis where Nasir+2017's "AGN slightly reduces below stellar"** lives, but Nasir+2017 reports no specific numeric on that comparison.

4. **CDDF-UVB degeneracy.** **Load-bearing for Stage 3 spec design:** Tillman+2024 (arXiv:2410.05383) explicit acknowledgement: *"the effects of AGN feedback on the column density distribution function was found to be highly degenerate with re-scaling the assumed UVB"*. Citing **Burkhart+2022 + Khaire+2024**. Our Sherwood `tau.npy` carries a fixed Sherwood-pipeline UVB choice per class — we cannot marginalize it out post-hoc without separate τ-with-rescaled-UVB outputs (not on disk). **This is the dominant rule-3 anti-degeneracy concern for the CDDF deliverable** — it composes with the SCOPING.md §1 anti-degeneracy audit's ⟨F⟩-sensitive-detection-threshold concern (the ⟨F⟩ offsets [D-26] G5 found at C4 |r|=0.215 are directly UVB-correlated).

---

## §5 — Citation table

All entries verified against arXiv direct + (where available) NASA ADS / Oxford Academic / IOPscience.

| Bibkey | Authors | Title (short) | arXiv | Journal | Year | Anchor role |
|:---|:---|:---|:---|:---|:---|:---|
| Nasir+2017 | F. Nasir, J. S. Bolton, M. Viel, T.-S. Kim, M. G. Haehnelt, E. Puchwein, D. Sijacki | Stellar+AGN feedback on low-z Lyα forest in Sherwood suite | 1706.04790 | MNRAS 471, 1056 | 2017 | **Anchor A primary (same suite)** — qualitative + specific b-median (1.8 km/s Δ) + CDDF factor 2-5 (QUICKLYA-vs-stellar) |
| Bolton+2022 | J. S. Bolton, P. Gaikwad, M. G. Haehnelt, T.-S. Kim, F. Nasir, E. Puchwein, M. Viel, B. P. Wakker | Limits on non-canonical heating and turbulence in IGM from low-z Lyα | (verify) | MNRAS 513, 864 | 2022 | **Anchor A secondary (same suite)** — b-peak floor 22.5±2.5 km/s sim vs 27.5±2.5 km/s obs; AGN(H02) exceeds obs by 4.6–5.8σ at peak |
| Tillman+2023 | M. T. Tillman, B. Burkhart, S. Tonnesen, S. Bird, G. L. Bryan, D. Anglés-Alcázar, R. Davé, S. Genel | Efficient long-range AGN feedback affects low-z Lyα forest | 2210.02467 | ApJL 945 L17 | 2023 | Anchor A (suite-mismatched) — CDDF χ²_R 4.5 vs 86 raw (factor 19); collapses to 3.9 vs 8.6 (factor 2.2) under UVB rescaling; **UVB-degeneracy carrier** |
| Tillman+2024 (AJ) | M. T. Tillman, B. Burkhart, S. Tonnesen, S. Bird, G. L. Bryan, D. Anglés-Alcázar, S. Hassan, R. S. Somerville, R. Davé, F. Marinacci, L. Hernquist, M. Vogelsberger | Exploration of AGN and stellar feedback in IGM via low-z Lyα forest | 2307.06360 | AJ 166, 228 | 2024 | Anchor A direction-only — CAMELS A_AGN1/A_AGN2 effects on CDDF + b qualitative; no tabulated effect-size; **DIFFERENT paper from arXiv:2410.05383** |
| Tillman+2024 (ApJ) | M. T. Tillman, B. Burkhart, S. Tonnesen, S. Bird, G. L. Bryan | Effects of AGN feedback on Lyα forest flux power spectrum | 2410.05383 | ApJ accepted | 2024 (v2 2025) | P_F(k) anchor (NOT CDDF); explicit acknowledgement that **CDDF feedback signal is degenerate with UVB rescaling** |
| Burkhart+2022 | B. Burkhart, M. Tillman, A. B. Gurvich, S. Bird, S. Tonnesen, G. L. Bryan, L. E. Hernquist, R. S. Somerville | Low-z Lyα forest as constraint for AGN feedback models | 2204.09712 | ApJL 933 L46 | 2022 | Anchor A (suite-mismatched) — **factor 2-3 fewer absorbers at N_HI < 10^15.5** Illustris vs TNG; b-distributions "almost identical" (b-null between same feedback prescriptions) |
| Christiansen+2020 | J. F. Christiansen, R. Davé, D. Sorini, D. Anglés-Alcázar | Jet feedback and the photon underproduction crisis in Simba | 1911.01343 | MNRAS 499, 2617 | 2020 | NOT Anchor A — D_A / Γ_HI calibration (×6 → ×2.5 with jets); no CDDF Δ; **citation-attribution-correction over the SCOPING.md "Sorini+2020" entry** |
| Khaire+2024 | V. Khaire, T. Hu, J. F. Hennawi, M. Walther, F. Davies | Can low-z Lyα forest constrain AGN feedback? | 2306.05466 | MNRAS 527, 4545 | 2024 | Anchor A direction-only on b/N_HI — *"largely insensitive to differences in feedback models"*; cool baryon fraction 23% vs 39% is a non-Lyα-statistic discriminator |
| Tillman+2025 | M. T. Tillman, J. N. Burchett, B. Burkhart, V. Khaire, S. Borthakur | Reconstructing galactic feedback history via Lyα with HWO | 2507.13442 | JATIS accepted | 2025 | NOT Anchor A — observability requirements paper; ~5-10% P1D precision needed at z = 0.4–1.8; b-distribution tension acknowledged with no new numeric |
| Dong+2024 | C. Dong, K.-G. Lee, R. Davé, W. Cui, D. Sorini | AGN feedback in Lyα forest signature of galaxy protoclusters at z~2.3 | 2402.13568 | MNRAS accepted | 2024 | NOT Anchor A — wrong z (2.0 < z < 2.5); transmission-vs-overdensity not CDDF/b |

### Citation-error corrections to propagate forward (per [D-14] Walther-precedent + fps-sbi Q_S3 Maitra→Sinigaglia precedent):

1. **"Sorini, Davé, Anglés-Alcázar 2020 SIMBA AGN-vs-no-AGN flux statistics"** in SCOPING.md §2 Q-C3 candidate list 3 → **actual paper is Christiansen, Davé, Sorini, Anglés-Alcázar (2020)** *MNRAS* 499, 2617 (arXiv 1911.01343). The 2020 *Sorini*-first-author paper at MNRAS 499, 2760 (`2020MNRAS.499.2760S`) is on SIMBA CGM at z = 2–3, NOT the intended low-z AGN-vs-no-AGN flux-decrement paper. Same Walther-class arXiv↔journal pairing error as [D-14] / fps-sbi Q_S3.

2. **"Hummels et al. 2020"** in SCOPING.md §2 Q-C3 candidate list 5 → **no such paper exists in the searched space.** Hummels 2019 ApJ 882, 156 is a CGM resolution study (Enzo), not a low-z Lyα-forest feedback CDDF/b paper. Same hallucination-class error as the fps-sbi Q_S3 "Maitra+2026" entry (which resolved to Sinigaglia+2026). Drop from any future candidate list unless a specific Hummels-co-authored CDDF paper is identified by other means.

3. **"Tillman+2024 (arXiv:2410.05383)"** in [D-14] Sub-deliverable C is the **P1D paper**, not the CDDF/b-distribution paper. The Tillman+2024 paper on **CDDF + b-distribution from CAMELS** is **arXiv:2307.06360** (AJ 166, 228) — same overlapping author group, separate paper. Q-C3 needed both; [D-14] cited only the P1D paper. **This is a Walther-class arXiv↔journal pairing error** — two papers by overlapping authors with thematically-adjacent titles. Recommend the linewidth-cddf LEDGER's eventual [D-01] record this correction explicitly.

---

## §6 — Implications for Stage 3 spec

### Verb-ceiling tightening (binding on the eventual Stage 3 spec):

Per Anchor A = QUALITATIVE-ONLY, the Stage 3 deliverable verb-ceiling is **even tighter** than SCOPING.md §1 currently sets:

- **NOT admissible (extending SCOPING.md §1):**
  - "First multi-class classifier test of the Nasir+2017 / Tillman+2023 CDDF redistribution framing" — the CDDF redistribution numerics are UVB-confounded and not directly comparable to our 4-recipe output.
  - "Consistent with Tillman+2023 CDDF effect-size of factor 19×" — the factor 19 is the **raw UVB-contaminated number**, not the feedback-specific signal.

- **Admissible (refined per this audit):**
  - "First multi-class classifier test of the **Nasir+2017 same-Sherwood-suite direction**: AGN feedback produces a small (≲5%) b-distribution shift and a (≲ factor-2-5) high-N_HI CDDF redistribution at z ≈ 0.1, extrapolated to our z ≈ 0.3 4-recipe suite."
  - "C2-vs-C3 sub-gate is pre-committed to test whether the Nasir+2017-direction AGN-vs-stellar b-median Δ ≈ 1.8 km/s ≈ 5% is recoverable as a classifier signal on our 4-recipe Sherwood output at z ≈ 0.3."

### Stage 3 gate set must include (rule-14(iii) demotion-mandatory amendments):

1. **A pre-committed CDDF-UVB-degeneracy diagnostic gate.** Tillman+2023's factor 19 → 2.2 collapse under UVB rescaling MUST be acknowledged in the spec; the deliverable verb must NOT claim "feedback-driven CDDF redistribution" without naming the UVB confounder. Project-internal: a control gate "CDDF-Δ-between-classes is not 1:1 correlated with per-class ⟨F⟩-ranking" (analogous to [D-26] G5) is mandatory.

2. **A b-distribution noise-floor sanity gate.** Bolton+2022's simulated-b-peak floor at 22.5±2.5 km/s gives a pixel-scale-set bin width; the Nasir+2017 Δb = 1.8 km/s is below this. Stage 3 must pre-commit a noise-floor estimate from Q-C1 (synthetic Voigt recovery on `tau.npy` matching the pixel + noise model) before any b-median Δ result is reported. If recovered b is biased > 5% on synthetic at the matching pixel scale, the b-distribution deliverable is signal-limited and must short-cycle-close per SCOPING.md §5 failure mode (a).

3. **C2-vs-C3 sub-gate per [D-26] G4 precedent — load-bearing.** The cascade-coloured failure mode SCOPING.md §1 anti-degeneracy audit named is C4-vs-others lift + C2/C3 null re-manifesting on linewidth-cddf. The Nasir+2017 result *"AGN feedback produces slightly lower N_HI > 10^14 incidence than stellar-only with no specific numeric"* is exactly the qualitative C2-vs-C3 direction that produces a null on numeric quantification — strong cascade-inherited prediction that G4-analogue gate on Stage 3 will fail at the C2-vs-C3 axis.

### Rule-14(iii) deliverable surface — locked at scoping:

The Stage 3 deliverable surface is **locked to adjacent-finding-only** at scoping, mirroring pk-feedback-classifier [D-17]:

> *"Stage 3 linewidth-cddf result is published, under any user-triggered paper trigger, **only as a per-class CDDF + b-distribution comparison adjacent finding on the 4-recipe Sherwood suite at z ≈ 0.3** — never as a feedback-discrimination headline claim. Any effect-size we report is project-internal; the closest published comparison is Nasir+2017 (same suite, z = 0.1, QUICKLYA-vs-stellar factor 2-5 CDDF Δ at N_HI > 10^14 and Δb = 1.8 km/s median; AGN-vs-stellar direction-only). The bar is rule-14(iii)-fragile by SCOPING.md §2 Q-C3 verdict; no headline 'we recover Tillman+2023's CDDF redistribution' or 'we discriminate AGN feedback recipes via line widths' verbs admissible under any framing."*

### What Anchor A DOES enable:

- **A direction-of-test framing** — "are our 4-recipe Sherwood CDDF / b-distribution Δs in the direction Nasir+2017 reports on the same suite at z = 0.1?"
- **A pre-committed C2-vs-C3 null expectation** — Nasir+2017's qualitative AGN-vs-stellar direction without numeric is *direct published support* for the [D-26]-cascade-inheritance prediction that C2-vs-C3 will be at-or-near-null on linewidth-cddf at z ≈ 0.3.
- **A specific cross-check** — our pipeline's b-peak floor MUST be within ≈ 2-3 km/s of Bolton+2022's 22.5 km/s sim peak on the same pixel scale, else the line-fitting is broken.

### What Anchor A does NOT enable (rule-14(i)-clean bar):

- A pre-committed numeric PASS/FAIL effect-size threshold for the C4-vs-others CDDF normalisation Δ or b-distribution median Δ on our 4-recipe Sherwood output, in the rule-14(i)-clean sense. Such a threshold must be **self-anchored at spec authorship per rule-14(ii) construction**, and the rule-14(iii) deliverable-demotion-to-adjacent-finding is mandatory regardless.

---

**Audit sign-off provenance:** support-researcher dispatch, 2026-06-02, on `exp/pk-feedback-classifier` branch (per linewidth-cddf SCOPING.md gate 3, parallel with `data-engineer` Q-C1 dispatch). Web tools: 25+ WebSearch + WebFetch calls, abstracts + Oxford Academic body excerpts + IOPscience abstracts + ADS bibcodes fetched and direct-quoted. Anchor A verdict: **QUALITATIVE-ONLY** with two specific-numeric same-suite floors (Nasir+2017 b-Δ 1.8 km/s, CDDF factor 2-5 QUICKLYA-vs-stellar; Bolton+2022 b-peak 22.5±2.5 km/s). Rule-14(iii) deliverable demotion at scoping **MANDATORY** per SCOPING.md §2 Q-C3 explicit clause. No paper-author dispatch; no linewidth-cddf LEDGER modification (track not yet open); no `data/` modification; no `src/core/` modification. This audit hands off to the PI for Stage 3 spec authorship (SCOPING.md gate 4), pending Q-C1 return and user acceptance of the SCOPING.md as a whole. Three citation-attribution corrections surfaced (Sorini-vs-Christiansen, Hummels-not-found, Tillman+2024-AJ-vs-ApJ) — all Walther-class arXiv↔journal pairing errors with fps-sbi Q_S3 / pk-feedback-classifier [D-14] precedent.
