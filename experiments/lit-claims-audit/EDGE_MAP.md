# EDGE_MAP — lit-claims-audit

**PI synthesis, 2026-10-20.** Numbers come from `LEDGER.md` §3 [D-01]–[D-10] (not recomputed). **Review status:** PI-only, PROVISIONAL (rule 15). The defense-panel review of results (LEDGER §7 Next 2) is pending, so nothing below is load-bearing. **Scope:** Sherwood z = 0.3, noiseless, except T2 (COS-like forward model).

## 1. Bottom line

The nulls come first:
- **Winds do not broaden lines.** Δmedian b(2−1) at log N ≥ 13 is −0.46 km/s fiducial and +0.07 ⟨F⟩-matched, so Viel+2017's +3 km/s is excluded.
- **Feedback does not move the thermal b-envelope.** Δb_env(4−1) = −0.54 km/s.
- **The typical absorber is unchanged between paired runs.** Median paired Δb(3−2) = +0.01 km/s.
- **Per-sightline discrimination is weak.** 2-vs-3 BA is 0.541 at S/N 30 against nulls of 0.50, and it fails the 0.52 bar at S/N 10 (0.519).
- **The Γ-degeneracy A-ratio test is inconclusive.**

What survives is this. After dN/dz matching over Khaire+2024a's range, two Sherwood feedback variants stay **distinguishable between two simulations under a COS-like model with known thermal nuisances**. The difference is in line widths at fixed N: Δb(3−2) = +3.9 [3.5, 4.4] and Δb(4−1) = +4.8 [4.2, 5.7] km/s at 13 ≤ log N < 13.5. It does not restate line density: N/count-only BA is 0.509. In noiseless paired skewers the broadening is carried by a minority of absorbers, with the top 20% of lines giving 95% of it. For class 4 it goes with a volume-filling hot phase.

The signal is smaller than the 8–10 km/s offset between observed and simulated b. Detecting it would need ~650–700 (3−2) or 180–290 (4−1) box-equivalent sightlines at S/N 30, from a post-hoc estimator. Our edge is a mechanistic characterisation of where the feedback b-signal lives: resolved in N, built on shared ICs, and checked under forward modelling. It is not a claim of observational detectability.

## 2. Gap map

| Gap | Literature position | Our result | Verdict | Why |
|---|---|---|---|---|
| G1 Per-sightline discrimination | None found (LIT_SURVEY §3.9, §4). Per-sightline ML exists only for thermal parameters (C-42 Nasir+2024, z = 4–5) | D-08: BA 0.579 noiseless (null 0.516). D-09: 0.541 at S/N 30 (nulls 0.497 / 0.499); 1-vs-4 0.602 | PARTIAL | Holm-significant, but ~4 points above chance is not a usable classifier. "Nothing published" rests on a search that found nothing |
| G2 Do winds broaden b? (V17 vs N17) | C-09 Viel+2017: +3 km/s. C-03 Nasir+2017: "very little difference" | D-03: Δb(2−1) −0.46 / +0.07; Δb(3−2) +1.5 to +1.8 | PARTIAL | Settles the disagreement within the code family toward N17, but noiseless, at z = 0.3 vs 0.1, and with unconfirmed run identity |
| G3 Envelope vs bulk | C-13 Bolton+2022; C-39 Hu+2024 | D-04: b_env 19.6–20.2 km/s for every class | CLOSED | A confirmation, not an edge |
| G4 Paired-absorber response | No published analogue | D-06: median +0.01, mean +0.68 [+0.60, +0.76], top-20% share 95%; 1→4 match rate 38–39% | OPEN-EDGE (provisional) | Only shared ICs make this possible. It still lacks a fitter-jitter null |
| G5 Degeneracy breaking after matching | C-26 Khaire+2024a (~4–5% at k > 0.04); C-27 Burkhart+2022; C-14 Bolton+2022 | D-07: P_F(k) shape −3.3 / −5.5 / −9.4%; A-ratios INCONCLUSIVE. D-08: Δb(4−1) +8.0 [+7.4, +8.6] at 12.5–13 (dndz12) | PARTIAL | The P_F(k) side repeats a known degeneracy breaker in a third suite. The b side is new, but only for 4−1: the 2–3 pair barely loads the degeneracy (D-02) |
| G6 COS forecast at z ≈ 0.3 | C-47 Tillman+2025b (z ~ 0.4–1.8 gap); C-26 | D-09: n₃σ 650–700 / 180–290 (post-hoc); pre-registered estimator 2,800–7,000 / 640–1,140, unstable across noise seeds | PARTIAL | Statistical errors only; the 8–10 km/s offset is not included; G130M LSF |
| **G7 (new)** Sign structure across N | Pooled or windowed statistics (C-12 uses 13.3–14.5; C-26) | D-08 / D-09: Δb > 0 at 12.5–13.5, ≈ 0 at 13.5–14, < 0 at 14–14.5 (forward-modelled 3−2 −1.0, 4−1 −1.5). The pre-registered pooled estimator cancels across bins | PARTIAL (post-hoc) | A candidate reason pooled statistics look "insensitive". Exploratory, and untested on Illustris/TNG |
| **G8 (new)** Class-4 thermal state | C-13: Bolton ΔT₀ ≤ 2.4%. C-15: StrongAGN WHIM fraction 68.4% | D-05: cool-phase T₀ +38%; hot fraction at mean density 64.8% vs 11.0%; envelope unchanged | PARTIAL | A coherent picture inside our suite. Whether it contradicts Bolton depends on run identity and on how T₀ is defined |

## 3. Claim scorecard

| Claim | Verdict | D-xx | Note |
|---|---|---|---|
| C-01 N17: small effect on CDDF and b | CONSISTENT | D-03 | Classes 1–3 only |
| C-02 N17: Δb +1.8 km/s (AGN vs none) | CONSISTENT | D-03 | Our Δb(3−1) is +1.3 to +1.7 |
| C-03 N17: winds change b little | CONSISTENT | D-03 | |
| C-04 N17: winds give 2–5× more log N ≥ 14 absorbers | INCONCLUSIVE | D-03 | Ratio 1.84 / 1.47, at or below the low end |
| C-05 N17: AGN broadening seen only at z = 0.1 | EXTENDS | D-03 | Present at z = 0.3 (N17 had no z = 0.3 output) |
| C-06 AGN adds b > 50 km/s lines | CONSISTENT | D-03 | Ratio 1.31–1.34 |
| C-08 Forward model shifts median b by ~8 km/s | INCONCLUSIVE | D-09 | Shift ranges from −0.5 to +8.8 depending on the model |
| C-09 V17: winds +3, AGN +2 km/s | CONTRADICTS (winds) / CONSISTENT (AGN) | D-03 | |
| C-12 Bolton: narrow-line tension | INCONCLUSIVE | D-09 | Medians 30.9–34.0 km/s; the peak statistic was not computed |
| C-13 Bolton: T₀ unchanged by AGN | CONSISTENT (class 2) / mild tension (class 3, +6.9%) / CONTRADICTS (class 4, provisional) | D-04, D-05 | |
| C-14 Bolton: Γ ratios | INCONCLUSIVE | D-07 | A₄/A₃ = 1.54 and A₃/A₁ = 0.89, both just outside the confirm bands |
| C-16 StrongAGN changes CDDF shape | CONSISTENT | D-07 | χ²/bin 36 vs 5–9 |
| C-21 AGN broadens b (direction) | CONSISTENT | D-03, D-08 | |
| C-24 b stays elevated after T₀ is restored | CONSISTENT (pattern) | D-04 | Different mechanism (Simba jets) |
| C-26 Khaire: insensitive after dN/dz matching | EXTENDS / partially CONTRADICTS (information content) / CONSISTENT (practical) | D-09 | Different simulation suites |
| C-26 / C-27 P_F(k) shape residual | CONSISTENT | D-07 | |
| C-33 Danforth: β = 1.65 | INCONCLUSIVE | D-07 | Ours 1.94–2.08, noiseless; not interpreted |
| C-39 Hu+2024: b–N insensitive to WHIM | CONFIRMS (envelope) | D-04 | |

## 4. Ranked edges

**E1. The fixed-N width difference survives dN/dz matching and the COS-like forward model (G5, G6, G7).**
- **Claim.** On Sherwood z = 0.3, after matching dN/dz over 12 ≤ log N < 14.5, pairs of feedback variants are distinguishable between two simulations under a COS-like model with known thermal nuisances. The difference is ≤ 5 km/s and concentrated at 13 ≤ log N < 13.5.
- **Evidence (D-08, D-09, D-10).**
  - Δb +3.9 (3−2) and +4.8 (4−1) km/s at S/N 30.
  - BA 0.541 (worst seed 0.538) vs a null + margin of 0.519; 1-vs-4 BA 0.602.
  - It is not line density or ⟨F⟩: N/count-only BA 0.509; residualising on [⟨F⟩, n13, ⟨τ⟩] leaves BA at 0.580.
  - Sensitivities stay within 0.535–0.547.
- **Strongest objection.** The 8–10 km/s offset between data and model is larger than every Δb. A thermal/turbulent nuisance that varies with N could mimic or erase the 3−2 difference. The only b_turb sensitivity run so far was one Gaussian broadening per sightline.
- **Experiment.** A nuisance-degeneracy test on the existing T2 machinery. Apply an N-dependent b_turb plus a T₀/γ envelope shift to class 2 before forward modelling, fit it to class 3's binned b–N, and ask whether any nuisance setting reproduces class 3 within the block CIs at S/N 30. If one does, the edge reduces to "degenerate with thermal nuisance." The G160M LSF is not on disk and is needed for the z = 0.3 instrument.

**E2. Feedback acts on a localised minority of absorbers (G4).**
- **Claim.** In noiseless paired skewers the typical absorber is unchanged. The mean broadening comes from roughly 20% of lines, and strong AGN reshuffles absorbers rather than shifting them.
- **Evidence (D-06).** Median +0.01, mean +0.68 km/s, top-20% share 95%; 1→4 match rate 38–39%; the 5 km/s matching window agrees. It also fits the signed-response track's finding that the response is localised to the responding regions.
- **Strongest objection.** No fitter-jitter null. Re-splitting blends produces heavy symmetric tails (9% of lines broaden by > 5 km/s, 7% narrow by > 5 km/s), and those tails alone could give a high top-20% share. It is also unobservable directly and noiseless only.
- **Experiment.** Run the same paired statistics on class 2 against its own refits (`null2_dndz12`, `tol005`, `tol002` catalogs already exist) to get the jitter baseline for the localisation share and the > 5 km/s tails. The edge holds only if 2→3 clearly exceeds that baseline.

**E3. Class 4 is a two-phase IGM with the thermal floor intact (G8, G3).**
- **Claim.** In this suite, the strong-AGN variant leaves its 10th-percentile temperature and b-envelope unchanged, while 61–65% of mean-density gas sits above 10⁵ K. That single picture accounts for its higher ⟨F⟩, fewer lines and extra broad weak lines.
- **Evidence (D-04, D-05, D-08).** Hot fraction 64.8% vs 11.0–16.5%; f_HI about 10× lower; the 12–12.5 refit excess of +1.9 km/s. The 64.8% resembles Bolton's StrongAGN WHIM fraction of 68.4%, though the two are defined differently.
- **Strongest objection.** Largely an explanation of the ⟨F⟩ offset, not an observable orthogonal to it. The +38% cool-phase T₀ may be an artefact of the T < 10⁵ K cut: the hot-phase tail can leak below the cut and pull T₀ up. Run identity is unresolved.
- **Experiment.** Refit T₀/γ from the LOS fields with alternative cuts (10^4.5 K; HI- vs density-weighted). The contradiction stands only if ΔT₀ stays well above 2.4% under every cut. Settling run identity needs the Sherwood run parameter files, which we do not hold.

**E4. A P_F(k) shape residual survives CDDF matching (G5). Weak.**
- **Evidence (D-07).** −3.3 / −5.5 / −9.4% at k > 0.04 s/km; Holm reject.
- **Strongest objection.** It replicates Khaire+2024a and Burkhart+2022 in a third suite, so it is not new. It sits beyond the observed COS cutoff at k > 0.03 s/km (C-35), and it is noiseless.
- **Experiment.** Forward-model P_F(k) with the T2 LSF and noise, and check whether the residual's CI still excludes 0 at the k values COS resolves.

## 5. What is NOT an edge

- **C-08 b-shift.** +4.4 to +4.9 km/s under the pre-registered model, but −0.5 to +8.8 across models. It comes from selection and continuum placement, so it is a systematics finding.
- **T6 median.** Δb +0.010 [0.007, 0.012] km/s is statistically ≠ 0 and practically zero. The pre-registered EXTENDS rule passed on this median, so the rule was badly designed. The real content is the mean and the localisation, and both still need the jitter null.
- **T3 A-ratios.** 1.54 and 0.89, just outside the confirm bands. INCONCLUSIVE, not a contradiction of Bolton.
- **S/N 10.** BA 0.519 fails the bar; completeness at 12.5–13 is 17–24%.
- **Per-sightline BA as a classifier.** 0.54 is a detection statistic, not a tool.
- **2–3 "robust to ⟨F⟩".** A weak test (D-02): A₃/A₂ = 1.047.
- **T4a CONFIRMS label.** The pre-registered bar also required Δmedian b ≥ 4 km/s. The bulk median at log N = 13.5 is +2.3 / +2.5, so the bar is met only by the pooled or low-N medians. The ⟨F⟩-matched Δb_env of −1.03 sits on the 1 km/s bar.
- **Narrow-line tension "not confirmed".** Our statistic is the median and Bolton's is the peak, at a different z. This is not evidence against Bolton+2022.
- **CDDF slope.** 1.94–2.08 against Danforth's 1.65, but noiseless; not interpreted.

## 6. Open caveats (all results)

1. **Run identity.** Bolton+2022 has no winds-only run, and our 4-class set matches no published run table.
2. **Redshift.** Every literature anchor is at z = 0.1, and we have one snapshot at z = 0.3.
3. **Fitter.** We use patched VoigtFit with AICc; the literature uses VPFIT. A FLAME/VIPER cross-check has not been done.
4. **LSF.** We used the G130M LSF at 1327 Å; z = 0.3 Lyα needs G160M.
5. **Literature numbers.** [body] numbers are WebFetch paraphrases, including Khaire's "largely insensitive". They must be checked against the PDFs.
6. **Resolution.** Single resolution: b falls as resolution increases, and the CDDF needs 10–30% corrections at log N ≥ 14 (C-18).
7. **Temperature weighting.** T is HI-weighted, not mass-weighted.
8. **Forecast.** The 650–700 / 180–290 figures come from a post-hoc estimator (a stated deviation). Systematics are excluded, and 1 sightline is one 60 h⁻¹ cMpc box (Δz = 0.0234).
9. **Bootstrap resolution.** p = 0.002 (500 resamples).
10. **Novelty.** The G1 and G4 novelty claims rest on searches that found nothing. Re-search before any novelty claim.
11. **Prior claims falsified in this line of work.** The eigen-voigt "robust to ⟨F⟩" claim was downgraded by D-02. Under the falsified-prior cascade rule, E1–E4 are hedged one verb level.

## 7. Next-step authorisation gate

**PI can authorise now (probe-class, data on disk, run each in full):**
1. The defense-panel review of this EDGE_MAP. It is required before any edge becomes load-bearing.
2. The E2 fitter-jitter null on the existing refit catalogs.
3. The E1 nuisance-degeneracy test, forward-modelled at S/N 30.
4. The E3 T₀ cut-robustness check on the LOS fields.
5. The E4 forward-modelled P_F(k) check.
6. PDF verification of every [body] number in §3 (support-researcher).
7. A LEDGER [D-11] recording this map, and the §6 figures.

**User-owned:**
- Getting data we lack: the G160M LSF, a z = 0.1 snapshot, Sherwood run parameter files, and other suites (Illustris/TNG catalogs for testing G7).
- Opening any follow-on track.
- Any paper. This document does not recommend one, and the paper trigger is the user's alone.