# CLAIM_TEST_PLAN — lit-claims-audit

**Source:** project-architect (PI), 2026-10-19, from `LIT_SURVEY.md` (36 verified papers, 49 claims).
**Status:** PI-only, PROVISIONAL (rule 15). A defense-panel review of the T1/T2 protocol runs in parallel;
it may amend metrics but may not cancel a test. **Binding user rule:** every test runs in full; a smoke run
checks engineering only ([[feedback-smoke-not-a-gate]]).
**User goal:** "explore all the related studies, run their claims to find gaps, so that we can find our own edge."

## Tests, ranked by decisiveness

### T1 — line-density matching vs the 2–3 width lead (C-26, C-27 b-part, C-03, C-09)
- **Variants.** For each class, A_c is chosen so that dN/dz(log N ≥ X) equals the pooled mean, for
  X ∈ {12.5, 13.0, 13.5}. A_c is solved analytically on the fiducial catalog (log N → log N + log10 A_c),
  then Voigt is re-fitted on roll(exp(−A_c τ_c), 1). The matched dN/dz must land within 2%; iterate once
  if not.
- **Classifier.** Dedicated binary 2-vs-3: RF and logistic regression on the 33 Voigt features
  (fold-internal imputation), 5 spatial-block folds × 10 seeds. Run in four variants: fiducial,
  ⟨F⟩-matched (existing τ-rescaled), dN/dz-matched (13.0 primary; 12.5 and 13.5 sensitivity), and
  dN/dz13 + fold-internal spline residualiser on [⟨F⟩, n_lines(≥13), ⟨τ⟩_clip].
- **Ensemble.** Δmedian b(3−2) and Δmedian b(4−1) within log N bins [13, 13.5), [13.5, 14), [14, 14.5),
  with 192-block bootstrap CIs.
- **Primary.** RF BA, dN/dz13-matched, with block-bootstrap CI.
- **Outcomes.**
  - CONTRADICTS Khaire (lead survives): BA ≥ 0.55, CI lower bound > 0.52 at all three X, and ≥ 0.53
    after residualisation.
  - CONFIRMS Khaire: BA < 0.53, or CI includes 0.52.
  - INCONCLUSIVE: anything between.
  - EXTENDS: Δb(4−1) ≥ 3 km/s at fixed N, CI excluding 0 after matching.

### T2 — forward-modelled observability (C-08, C-02, C-12, C-33, C-34, C-47)
- **Forward model.** Gaussian LSF with FWHM 17 km/s, periodic convolution (an approximation to COS,
  which has wings). Rebin ×3 (2046 px → 682 px, Δv = 7.91 km/s; the last 2 px are dropped). Add
  flux-independent Gaussian noise at S/N per rebinned pixel ∈ {∞, 10, 20, 30, 50}.
- **Fitting.** VoigtFit with the LSF. The stopping rule is AIC: a component is kept only if ΔAIC < −2.
  Seeds and regions use noise-significance thresholds.
- **Runs.** Variants ⟨F⟩-matched and dN/dz13-matched, at every S/N, with noise seed 0. Noise seed 1 is
  added at S/N 30 (primary) for both variants. The T1 binary CV is then rerun on each.
- **Forecast.** n_3σ = (3 σ_b / Δb̄)² sightlines for an ensemble 2-vs-3 detection.
- **Primary.** RF BA 2-vs-3 at S/N 30, dN/dz13-matched.
- **Outcomes.**
  - EXTENDS (observable per sightline): CI lower bound > 0.52.
  - Otherwise INCONCLUSIVE per sightline; report n_3σ.
  - CONFIRMS the narrow-line tension at z = 0.3 if every class's forward-modelled median b < 30 km/s.
  - Report the median-b shift between noiseless and forward-modelled fits; a shift ≥ 4 km/s is
    "consistent with C-08".

### T4 — thermal envelope vs bulk (C-13, C-24, C-39, C-11, C-36)
- **T4a.** Lower envelope b_env(N) = 10th percentile of b in 0.25-dex bins over log N ∈ [12.75, 14.5].
  Fit a power law and evaluate at log N = 13.5. Run in the fiducial and dN/dz13 variants, with block
  bootstrap.
- **Outcomes.**
  - CONFIRMS Bolton / Hu+2024: |Δb_env(4−1)| < 1 km/s (CI) while Δmedian b ≥ 4 km/s.
  - CONTRADICTS: Δb_env ≥ 2 km/s with CI excluding 0.
  - INCONCLUSIVE: otherwise.
- **T4b** (fit T0, γ from temperature–density, and look for a 10^5 K secondary peak) runs only if
  `data/raw` carries temperature and density.

### T6 — paired-absorber response (shared ICs; C-05, C-21 direction)
- **Matching.** On the same skewer, match class-2 to class-3 components (and 1 to 4) by mutual nearest
  v, with |Δv| < max(0.5 b, 5 km/s). Components need log N ≥ 12.5.
- **Statistics.** Per-line Δb vs log N; fraction of lines with |Δb| > 5 km/s; localisation (the share of
  total positive broadening carried by the top 20% of lines); match rate. Also a per-line classifier on
  [b, log N, Δv_nn] and on matched pairs (exploratory).
- **Outcomes.** EXTENDS: median paired Δb(3−2) CI excludes 0 and ≥ 70% of the broadening comes from
  ≤ 20% of lines. INCONCLUSIVE: CI includes 0.

### T3 — Γ/UVB degeneracy and CDDF normalisation (C-14, C-16, C-27, C-20, C-28, C-33, C-25)
- **Matching.** CDDF-matched A_c minimises χ² against class 1 over 13.3 ≤ log N ≤ 14.5; then re-fit.
- **Statistics.** CDDF slope β (dN/dlogN ∝ N^(1−β), fit over 12.5–14.5), flux-PDF KS statistic vs
  class 1, and P_c(k)/P_1(k) with block CIs.
- **Outcomes.**
  - CONFIRMS: A4/A3 ∈ [1.6, 2.3] and A3/A1 ∈ [0.9, 1.1].
  - CONTRADICTS: A3/A1 outside [0.85, 1.15], or A4/A3 < 1.4.
  - EXTENDS: flux-PDF / P_F(k) shape differences survive CDDF matching (CI excludes 0 at k > 0.04 s/km).

### T5 — winds vs AGN on b and the CDDF; Viel+2017 vs Nasir+2017 (C-01, C-03, C-04, C-06, C-09)
- **Statistics.** Δmedian b for 2−1, 3−2 and 4−3 (log N ≥ 13 primary, ≥ 12 secondary); the
  f(b > 50 km/s) ratio class 3 / class 2; the count ratio class 2 / class 1 at log N ≥ 14. Run in three
  variants: fiducial, ⟨F⟩-matched, dN/dz13.
- **Literature values.** V17: Δb(2−1) ≈ +3 km/s, Δb(3−2) ≈ +2 km/s. N17: Δb(2−1) < 1 km/s.
- **Outcomes.** "Consistent with V17 / N17" when the CI covers one and excludes the other; INCONCLUSIVE
  otherwise.

## Controls, multiplicity, ceilings
- **Controls.** 192 spatial blocks (CV and bootstrap); ⟨F⟩ and ⟨τ⟩_clip scalars; detection-threshold
  sweep; per-feature AUC across rescaling variants (panel #3 artefact check).
- **Multiplicity.** One primary per test, with Holm correction across the six primaries (family α = 0.05):
  - T1: BA > 0.5
  - T2: BA > 0.5
  - T3: P_F(k) shape residual at k > 0.04 ≠ 0
  - T4: Δb_env(4−1) ≠ 0
  - T5: Δb(2−1) ≠ 0
  - T6: paired Δb(3−2) ≠ 0

  Everything else is exploratory, verb ceiling "observed".
- **Verb ceilings.** Results are stated "on this Sherwood suite at z = 0.3, noiseless" (or
  "forward-modelled", T2). Literature comparisons are worded "consistent with", noting z = 0.3 vs 0.1
  and run identity. No "invariant" wording. "Observable" only from T2 at S/N 30. NULLs are reported
  first (D-37).
- **Anti-degeneracy (rule 3).** Report the components-per-region distribution and line spacing alongside
  b (fitter under-splitting of blends). Compare b within N bins, never pooled.
- **Paper trigger.** None. Every outcome is recorded as a LEDGER D-XX; the paper call is the user's.
- **Literature numbers.** Body-sourced numbers ([body] tags in `LIT_SURVEY.md`) must be checked against
  the PDFs before they are quoted outside the LEDGER.

## Gap hypotheses (candidate edges)
- **G1.** Per-sightline feedback discrimination (T1, T2, T6): nothing published at low z.
- **G2.** The 2–3 axis at z = 0.3 (T1, T5): the literature splits (V17 vs N17).
- **G3.** b-envelope vs bulk (T4).
- **G4.** Paired-absorber response from shared ICs (T6).
- **G5.** Degeneracy breaking after CDDF/dN/dz matching (T1, T3).
- **G6.** COS observability forecast in the z ≈ 0.3–0.4 gap (T2).

## Pre-checks
- Do `data/raw` (34.4 GB, DVC) LOS files carry temperature/density? This gates T4b.
- `HARDENING_OUTCOME.md` is absent on this branch, so its state is inherited from memory, not re-verified.

---

## Amendments after defense-panel review (2026-10-19, NEEDS WORK) — binding, applied before T1/T2 run

1. **T1 cannot CONTRADICT Khaire+2024a.** Khaire's insensitivity statement is about *forward-modelled*
   spectra (COS LSF, VPFIT); perfect-skewer differences "wash out" there. T1 is re-scoped as a
   **noiseless matched characterisation**. Only T2 can test Khaire, and it uses Khaire's matching range,
   12 < log N < 14.5. (Quote to be verified against the PDF; it was read from the arXiv HTML.)
2. **The 2–3 pair barely loads the degeneracy.** The ⟨F⟩-matched A₃/A₂ is 1.047 (a 0.02-dex shift), so
   the earlier "robust to equalising ⟨F⟩" was a weak test for 2–3. A₃/A₂ and A₄/A₁ are reported for
   every matching, and Δb(4−1) is promoted to **co-primary** (the real Khaire analogue).
3. **T1 primaries are ensemble statistics:** paired per-skewer Δb within log N bins
   [12,12.5), [12.5,13), [13,13.5), [13.5,14), [14,14.5), for 3−2 and 4−1, with 192-block bootstrap.
   **Prediction:** τ scaling leaves an isolated line's b unchanged, so Δb_matched(N) ≈
   Δb_fid(N − log10 A), computed in advance from the fiducial catalog; a refit departure from it is
   attributed to fitter behaviour.
4. **Per-sightline BA is secondary**, with these fixes:
   - logN_detect is fixed at 12.0 (sensitivity at 12.5 and 13.0).
   - The null is a **rescaling-only refit**: class 2 refitted with τ × (A₃/A₂), classified against
     class-2 fiducial.
   - Outcomes are disjoint: SURVIVES (noiseless) if the CI lower bound > null BA + 0.02; VANISHES if the
     CI includes null BA + 0.01; PARTIAL otherwise.
   - Ablations: b-only features vs N/count-only features, and per-line b residuals from a pooled
     fold-internal b(N) spline aggregated per sightline.
   - Fitter-depth sensitivity: tol ∈ {0.005, 0.002} refits for classes 2 and 3 (fiducial), plus
     components per region by class.
5. **T2 forward model:**
   - LSF: the tabulated COS LSF in `data/raw/Sherwood/COS_LSF1_cw1327_75_z0.100.dat`, used both to
     forward-model and to fit. It is G130M at 1327 Å; z = 0.3 Lyα at 1580 Å would be G160M, which is not
     on disk, so this mismatch is stated. A Gaussian with FWHM 17 km/s is a sensitivity.
   - Continuum error: ±1–2% low-order (Legendre ≤ 2) perturbations.
   - Noise: σ ∝ √F as a sensitivity.
   - Model selection: AICc (primary) and ΔAIC < −10 (sensitivity).
   - **Injection–recovery** at every S/N with each class's own (b, N) before classification: report b
     bias and completeness vs (b, N) per class.
   - **Label null:** class 2 vs class 2 with different noise seeds (expected BA = 0.5).
   - ≥ 5 noise seeds at the primary S/N (30).
   - Roll each sightline so that no absorption region straddles the trimmed edge.
   - b_turb nuisance sensitivity: an extra Gaussian broadening drawn per sightline.
   - **Verb ceiling:** "distinguishable between two simulations under a COS-like model with known
     thermal nuisances" (observed b exceeds simulations by 8–10 km/s, so absolute "observable" is out of
     reach).
6. **Statistics.**
   - Holm is applied to practical nulls (BA vs the rescaling/label null + margin; Δb vs the predicted
     Δb), not to BA > 0.5.
   - n_3σ uses per-sightline/block σ, the 1.25 median-SE factor and the two-sample factor 2, with Δb
     from forward-modelled fits.
   - Axis-held-out sensitivity: train on two projection axes, test on the third.
7. **T6.** The b-dependent matching window is replaced by a fixed |Δv| < 10 km/s window (sensitivity at
   5 km/s).
