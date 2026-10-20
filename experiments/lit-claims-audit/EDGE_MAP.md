# EDGE_MAP v2 — lit-claims-audit

**PI synthesis, 2026-10-20. Replaces v1.** Numbers come from `LEDGER.md` §3 [D-01]–[D-15] (not recomputed). **Review status:** PI-only, PROVISIONAL (rule 15). It incorporates the panel's NEEDS WORK corrections ([D-12]), and a panel re-review of v2 is required before any of it is load-bearing. **Scope:** Sherwood z = 0.3, noiseless, except where forward-modelled.

> **What changed from v1 and why**
> - **Khaire+2024a:** "partially contradicts" is withdrawn → CONSISTENT / EXTENDS by quantification ([D-12]). Their data (Δz = 1.79) would see only ≈ 1.0σ (post-hoc estimator) or ≈ 0.5σ (pre-registered) of our 3−2 residual.
> - **Mechanism ([D-15]):** in every N bin, the fixed-N width difference equals the difference in thermal width from absorber gas temperature. It is the thermal state. **This makes the edge weaker as a feedback discriminator** (see §1).
> - **E1 v1 (fixed-N Δb):** degenerate with a b_x ≈ 9 km/s broadening at the forecast sample size ([D-14]). The forecast rises to ~1,700 / ~1,400 sightlines once that nuisance is free.
> - **Per-sightline 2-vs-3:** fails under the combined realistic instrument (BA 0.524 [0.520, 0.527]) → G1 is closed for 2–3.
> - **E2 v1 (localisation):** retracted. The jitter null scores 0.996, above the real pair's 0.951 ([D-13]).
> - **E3 v1 (class-4 T₀ +38%):** this was warm gas leaking under the 10^5 K cut. At a 10^4.5 K cut it is +2.7%, so "contradicts Bolton" is withdrawn ([D-13]).
> - **E4 v1 (P_F(k) shape):** survives forward modelling but is demoted to a replication.
> - **T4a:** CONFIRMS is downgraded to "consistent, borderline" ([D-12]).
> - **Verb ceiling:** the 13 ≤ log N < 13.5 location was chosen post hoc and stays at "observed".

## 1. Bottom line

Nulls first:
- **Winds do not broaden lines.** Δb(2−1) at log N ≥ 13 is −0.46 / +0.07 km/s.
- **Per-sightline 2-vs-3 discrimination fails under realistic combined conditions.** BA 0.524, lower bound 0.520 against the > 0.52 bar.
- **Feedback leaves the cool T–ρ normalisation within ≈ 3% of no-feedback.** At a 10^4.5 K cut every class is within +2.7%, consistent with Bolton+2022.
- **The b-envelope is unchanged within a borderline ±1 km/s.**
- **The "localised minority of absorbers" statistic was uninformative** and is retracted.

What survives: after dN/dz matching over Khaire's range, Sherwood feedback variants leave a fixed-N width residual that is **distinguishable between two simulations under a COS-like model with known thermal nuisances**. Δb(3−2) is +3.9 and Δb(4−1) is +4.8 km/s at 13 ≤ log N < 13.5 (observed). It equals the thermal-width difference bin by bin (+2.8 / +4.9 at S/N 30). Feedback heats part of the moderately overdense absorbing gas: at fixed N, the median T_τ goes from 34,300 to 43,400 K.

**Plainly, this weakens the edge as a feedback discriminator.** The signal is the thermal state, which is exactly what observers marginalise when they fit [T₀, γ] and b_turb. A 9 km/s broadening already makes the profile degenerate at the forecast sample size. The attribution is real physics, but it also puts the signal where it is most likely to be absorbed. What remains of the edge is one testable hypothesis: feedback adds a warm, N-localised tail that is not a T–ρ power law. Whether a [T₀, γ] fit absorbs that tail is untested.

## 2. Gap map

| Gap | Literature position | Our result | Verdict | Why |
|---|---|---|---|---|
| G1 Per-sightline discrimination | None found (LIT_SURVEY §3.9) | D-15: 2v3 0.524 [0.520, 0.527] realistic; 1v4 0.566 | CLOSED (2–3) / PARTIAL (1–4) | 1–4 is not a clean AGN contrast: NoFeedback's star-formation prescription is unverified (D-12) |
| G2 Do winds broaden b? (V17 vs N17) | C-09 Viel+2017 vs C-03 Nasir+2017 | D-03: Δb(2−1) −0.46 / +0.07 | PARTIAL | Favours N17, but noiseless, z = 0.3, run identity unknown |
| G3 Envelope vs bulk | C-13 Bolton+2022; C-39 Hu+2024 | D-04: Δb_env(4−1) −0.54 fiducial / −1.03 ⟨F⟩-matched | CLOSED | Consistent, borderline; not an edge |
| G4 Paired-absorber response | No analogue | D-13: \|Δb\| > 5 in 16% vs 2.7% for jitter; two-sided. D-15: P(ΔT > 0 \| Δb > 5) = 0.80 vs base rate 0.64 | PARTIAL (narrowed) | Localisation retracted. Chaotic divergence is not yet separated from the response |
| G5 Degeneracy breaking after matching | C-26 Khaire+2024a; C-27 Burkhart+2022 | D-15: Δb = Δb_th. D-13: P_F(k) −2.4 / −4.4 / −6.8% at S/N 30 | PARTIAL | The b side is thermal and nuisance-degenerate at the forecast sample; the P_F(k) side is a replication |
| G6 COS forecast at z ≈ 0.3 | C-47 Tillman+2025b; C-26 | D-09 / D-12 / D-14: n₃σ 650–700 (post-hoc) or 2,800–7,000 (pre-registered) for 3−2; ~1,700 / ~1,400 with the nuisance free | PARTIAL | SEs may be low by up to √3 (3 axes × 64 cells); the factor 2 was dropped (with it, ≈ 1,300 for 3−2) |
| G7 Sign structure across N | Pooled or windowed statistics (C-12, C-26) | D-14: at b_x = 9 the residual is still +2.1 / +3.0 at 13–13.5. D-15: the sign change is thermal (class-4 gas at 14–14.5 is 0.08 dex cooler) | PARTIAL (observed) | The only part a uniform broadening cannot absorb. Its location was chosen post hoc |
| G8 Class-4 thermal state | C-13 Bolton: ΔT₀ ≤ 2.4% | D-13: +2.7% [2.0, 3.4] at 10^4.5 K; hot fraction 64.8% | CLOSED | No contradiction; the warm tail moves to G9 |
| **G9 (new)** Warm tail at fixed N, not a power law | C-36 Hu+2026 (hot IGM or v_tur ≈ 15 km/s); C-17 Bolton (extra heat or turbulence); C-39 Hu+2024 | D-15: +0.10 to +0.12 dex (3 vs 2), +0.15 to +0.29 dex (4 vs 1) at 12.5–13.5; cool T₀ moves ≤ 3% | OPEN (untested decider) | Whether a [T₀, γ] fit absorbs it is the open question. Its size (≤ 5 km/s) is below the 8–10 km/s tension, so feedback does not close the b tension |

## 3. Claim scorecard

| Claim | Verdict | D-xx |
|---|---|---|
| C-01 N17: small effect on CDDF and b | CONSISTENT | D-03 |
| C-02 N17: Δb +1.8 km/s (AGN vs none) | CONSISTENT (ours +1.3 to +1.7) | D-03 |
| C-03 N17: winds change b little | CONSISTENT | D-03 |
| C-04 N17: winds give 2–5× more log N ≥ 14 absorbers | INCONCLUSIVE (ratio 1.84 / 1.47) | D-03 |
| C-05 N17: AGN broadening seen only at z = 0.1 | EXTENDS (present at z = 0.3) | D-03 |
| C-06 AGN adds b > 50 km/s lines | CONSISTENT (ratio 1.31–1.34) | D-03 |
| C-08 Forward model shifts median b by ~8 km/s | INCONCLUSIVE (model-dependent, −0.5 to +8.8) | D-09 |
| C-09 V17: winds +3, AGN +2 km/s | CONTRADICTS (winds) / CONSISTENT (AGN) | D-03 |
| C-12 Bolton: narrow-line tension | INCONCLUSIVE (realistic-instrument medians 29.9–31.1; peak statistic not computed) | D-09, D-15 |
| C-13 Bolton: T₀ unchanged by AGN | CONSISTENT (all classes within +2.7% at 10^4.5 K) | D-13 |
| C-14 Bolton: Γ ratios | INCONCLUSIVE (A₄/A₃ = 1.54, A₃/A₁ = 0.89) | D-07 |
| C-16 StrongAGN changes CDDF shape | CONSISTENT | D-07 |
| C-21 AGN broadens b (direction) | CONSISTENT; attributed to heating | D-15 |
| C-24 b stays elevated after T₀ is restored | CONSISTENT (pattern) | D-13, D-15 |
| C-26 Khaire: insensitive after dN/dz matching | CONSISTENT; EXTENDS by quantification | D-12 |
| C-26 / C-27 P_F(k) shape residual | CONSISTENT (survives forward model) | D-13 |
| C-33 Danforth: β = 1.65 | INCONCLUSIVE | D-07 |
| C-39 Hu+2024: b–N insensitive to WHIM | CONSISTENT, borderline | D-04, D-12 |

## 4. Ranked edges (hedged one verb level: two v1 edges were falsified)

**E1. Candidate: feedback leaves a warm tail at fixed N that is not a T–ρ power law (G9).**
- **Claim (candidate).** In Sherwood z = 0.3, feedback heats part of the moderately overdense absorbing gas. The fixed-N median T_τ rises +0.10 to +0.12 dex (3 vs 2), while the cool T–ρ normalisation moves ≤ 3% and the b-envelope stays put (borderline).
- **Evidence (D-15, D-13).**
  - Δb = Δb_th in every N bin, noiseless and at S/N 30.
  - The fraction of absorbing gas above 10^4.5 K rises from 0.54 to 0.62.
  - The G7 sign change is also thermal.
  - The thermal-only τ synthesis reproduces the stored τ shape with a log-correlation of 1.000 / 0.998.
- **Strongest objection.** This is the nuisance. A [T₀, γ] fit (with a steeper γ) plus b_turb may absorb it fully, and at the forecast sample even a one-parameter broadening does ([D-14]).
- **Experiment.** τ re-synthesis with swapped temperatures, followed by a [T₀, γ, A] fit (runnable on disk; criteria in §7).

**E2. The N-shape of Δb is the part a uniform broadening cannot absorb (G7).**
- **Evidence (D-14).** At b_x = 9 km/s the residual stays at +2.1 (3−2) and +3.0 (4−1) at 13–13.5, while the neighbouring bins go negative. The profile χ² stays non-degenerate at full sample (178 / 213 over 5 bins).
- **Strongest objection.**
  - The 13–13.5 location was chosen post hoc (observed only).
  - It is untested against an N-dependent T₀/γ nuisance.
  - With the nuisance free it needs ~1,700 (Δz ≈ 39) / ~1,400 (Δz ≈ 33) sightlines, still optimistic (one-sided nuisance, 3 km/s grid).
  - SEs may be up to √3 too small.
- **Experiment.** A 3D-subcube jackknife to get honest SEs and bin covariance, then rescore the [D-14] χ² and forecasts. If they hold, apply the E1 [T₀, γ, A] fit to the profile.

**E3. Broadened absorbers in the feedback run are the heated ones (G4, narrowed).**
- **Evidence (D-13, D-15).**
  - The paired response is ~6× refit jitter (16% vs 2.7%) and two-sided, with net mean +0.7 km/s.
  - Broadened absorbers have ΔT > 0 in 80% of cases against a 64% base rate (1→4: 0.91).
  - Narrowed absorbers are not cooler (0.37 vs 0.36): they are re-decomposition or kinematics.
- **Strongest objection.**
  - Per-absorber Δb barely tracks Δb_th (r = 0.25; 5.6% of the variance).
  - Chaotic divergence (Genel+2019) is not separated from the response.
  - It is a simulation-only mechanism statement and cannot be observed directly.
- **Experiment.** A phase-randomised Δτ null: perturb class-2 τ with the power of (τ₃ − τ₂) but random phases, refit, and compare the paired tail fractions and the ΔT conditionals (runnable on disk).

## 5. What is NOT an edge

- **2-vs-3 per-sightline BA.** It fails the realistic instrument. The 1v4 BA of 0.566 rests on an unclean contrast.
- **The T6 localisation share.** The jitter null scores higher (0.996). The pre-registered rule could not fail.
- **The class-4 "+38%" T₀.** Cut leakage: +2.7% at 10^4.5 K.
- **The P_F(k) shape residual.** It survives forward modelling (−2.4 / −4.4 / −6.8%) but replicates Khaire+2024a and Burkhart+2022. Significance against COS error bars is not assessed.
- **The C-08 b-shift.** Model-dependent (−0.5 to +8.8 km/s); a systematics finding.
- **T3 A-ratios.** Just outside the confirm bands; INCONCLUSIVE.
- **"Contradicts Khaire."** Withdrawn: their Δz = 1.79 would see ≲ 1σ.
- **Any single-estimator forecast.** Quote the post-hoc and pre-registered estimators together, every time.
- **Feedback as the fix for the b tension.** Δb ≤ 5 km/s against an 8–10 km/s offset, and realistic medians are 29.9–31.1 km/s.

## 6. Open caveats

1. **Run identity.** The 4-class set matches no published run table. NoFeedback's star-formation prescription is unverified. Shared ICs are confirmed: void-density correlation with class 1 is r = 0.991 / 0.891 / 0.501.
2. **Redshift.** z = 0.3 vs the literature's 0.1.
3. **Fitter.** Patched VoigtFit with AICc vs VPFIT.
4. **LSF.** G130M at 1327 Å; z = 0.3 Lyα needs G160M.
5. **Literature numbers.** [body] numbers, including Khaire's data description, still need PDF verification.
6. **Resolution.** Single resolution, and b depends on resolution (C-18).
7. **Temperature.** T is HI-/τ-weighted.
8. **Forecasts.** Statistical only. The factor 2 was dropped (a deviation), SEs may be up to √3 low, and bin covariance was ignored.
9. **Selection.** At S/N 30 the noise stage moves the signal in N (3−2 at 13–13.5 goes from 2.9 to 3.9). For 3−2 this selection is class-blind; for 4−1 it is not (class 4 is detected 1–10 points less often).
10. **Novelty.** Novelty for G4/G9 rests on searches that found nothing.

## 7. Next-step authorisation gate

**Remaining open tests:**

| Test | Decides | Runnable on disk? |
|---|---|---|
| τ re-synthesis with swapped temperatures (class-2 ρ, v, f_HI with class-3 T, and vice versa) | E1: is the thermal attribution causal? | **Yes.** LOS fields are on disk and the mapping is validated (D-15) |
| [T₀, γ, A] fit: parametric b_th(N) power law plus b_turb, fitted to class 2 to reproduce class 3's forward-modelled profile | E1/E2: is the tail absorbed by the nuisance? | **Yes**, as a parametric fit. A Hu-style SBI would need a Nyx training set, which we do not have |
| Forward-space dN/dz matching (to Danforth+2016) | Matching realism | **Yes** (Danforth numbers need PDF verification) |
| Phase-randomised Δτ null | E3 | **Yes** |
| 3D-subcube jackknife + nuisance-marginalised Fisher forecast | E2, G6 | **Yes**, if the skewer coordinates are in the LOS headers (verify first) |
| G160M LSF | Instrument realism at z = 0.3 | **No.** Public STScI file; external fetch needed |
| z = 0.1 snapshot | Direct comparison with the literature | **No.** Data owners needed |

**Proposed pre-commitment for the E1 decider, for the panel to ratify before it runs:**
- **CAUSAL:** the swapped-T synthesis reproduces ≥ 80% of the class 3−2 Δb at 12.5–13.5.
- **DEGENERATE (E1 and E2 close):** the best-fit [T₀, γ, A, b_turb] applied to class 2 reproduces class 3's binned profile with χ² p > 0.05 at full sample.
- **OPEN-EDGE:** p ≤ 0.05 at full sample *and* at the sample size the jackknife-corrected forecast gives.

A DEGENERATE outcome is a valid end state: feedback in this suite would then be indistinguishable from a thermal-state shift at fixed N, which is CONSISTENT with Khaire.

**PI authorises (probe-class, on-disk data, run each in full):**
1. The panel re-review of v2.
2. The swap synthesis and the [T₀, γ, A] fit, first.
3. The jackknife.
4. The phase-randomised null.
5. Forward-space dN/dz matching.
6. PDF verification of the [body] numbers.
7. A LEDGER [D-16] recording v2, plus the §6 figures and the §7 history.

**User-owned:**
- Fetching the G160M LSF and other external data.
- Requesting a z = 0.1 snapshot or the run tables.
- Opening any follow-on track.
- Any paper. This map does not recommend one.
