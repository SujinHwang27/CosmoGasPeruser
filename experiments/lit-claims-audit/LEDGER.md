# LEDGER — lit-claims-audit

**Branch:** `exp/lit-claims-audit` (off `exp/eigen-voigt-embedding`). **Opened:** 2026-10-19.
**User request:** "explore all the related studies, run their claims to find gaps, so that we can find
our own edge."
**Inputs:** `LIT_SURVEY.md` (36 verified papers, 49 claims) → `CLAIM_TEST_PLAN.md` (PI, six tests T1–T6,
with seven binding defense-panel amendments).
**Binding rule:** every test runs in full; smoke checks engineering only.
**Scope of every result below:** Sherwood z = 0.3, noiseless spectra, unless the test is T2 (forward-modelled).

---

## 1. The Pulse

| Test | Question | Status / verdict |
|---|---|---|
| T5 | Do winds broaden lines? (Viel+2017 vs Nasir+2017) | ✅ **Consistent with N17**: winds do not broaden b |
| T4a | Does feedback move the thermal b-envelope or only the bulk? | ✅ **CONFIRMS Bolton/Hu** (envelope ≤ 1 km/s; bulk +1.3 to +2.5) |
| T4b | T–ρ relation from the LOS temperature field | ✅ Classes 1–3 ≈ unchanged T₀; **WindStrongAGN: volume-filling hot phase** |
| T6 | Line-by-line paired response (shared ICs) | ✅ **EXTENDS**: typical absorber unchanged; broadening carried by ~20% of lines |
| T3 | Γ/UVB degeneracy after CDDF matching | ✅ A-ratios INCONCLUSIVE (just outside the confirm bands); **P_F(k) shape residual EXTENDS** |
| T1 | Does the 2–3 width lead survive line-density matching (noiseless)? | ✅ **SURVIVES (noiseless)** at all matchings and fitter depths; Δb(4−1) EXTENDS |
| T2 | Does it survive a COS-like forward model? (the only test of Khaire+2024a) | ✅ **EXTENDS**: distinguishable per sightline (2–3 BA 0.541, nulls 0.50); ensemble needs ~650 (3−2) / ~200 (4−1) sightlines at S/N 30; the noiseless→observed b shift depends on the continuum/noise systematics |

---

## 2. Methodology
- **Catalogs.** VoigtFit decomposition (`src/core/voigt.py`, patched; see eigen-voigt-embedding [D-03])
  of τ-rescaled spectra.
- **Matching.** For each variant, A_c is solved analytically on the fiducial catalog, then refitted and
  iterated until the matched line density is within 2% of target (class 4 needed 1–2 iterations; its
  line counts respond non-linearly to τ scaling).
  - `dndz12`: 12 ≤ log N < 14.5, Khaire's range
  - `dndz13`: 13 ≤ log N < 14.5
  - `dndz13p5`: 13.5 ≤ log N < 14.5
  - `cddf`: χ² match to class 1 over 13.3 ≤ log N ≤ 14.5
  - `meanF`: equal class ⟨F⟩
- **Statistics.** Paired block bootstrap over 192 spatial blocks: classes share skewers, so one block
  resample is applied to every class.
- **Classification.** Binary per-sightline RF and logistic regression; 5 spatial-block folds × 10 seeds;
  fold-internal imputation.
- **Code.** `src/core/claims.py`, `src/core/forward.py`; scripts `run_claims*.py` / `.sh`.
  Results: `results/lit_claims_audit/`.
- **LOS fields.** `data/raw/Sherwood/*/los2048_n16384_z0.300.dat` (DVC-pulled 2026-10-19): HI-weighted T,
  ρ/⟨ρ⟩, f_HI, v_pec per pixel; format from `data/raw/Sherwood/src/utils.py`.

---

## 3. The Logic (Decision Log)

- **[D-01] Track opened; plan of record = `CLAIM_TEST_PLAN.md`** (PI triage of the 49 claims into
  T1–T6, PROVISIONAL under rule 15). The defense-panel protocol review (NEEDS WORK) produced seven
  amendments before T1/T2 ran. The decisive ones:
  - T1 cannot CONTRADICT Khaire+2024a, whose insensitivity statement is about forward-modelled spectra.
  - The 2–3 pair barely loads the normalisation degeneracy: A₃/A₂ ∈ [0.95, 1.06] under every matching.
  - Δb(4−1) is co-primary.
  - Ensemble primaries replace per-sightline BA as the headline.
  - T2 uses the tabulated COS LSF, continuum error, injection–recovery, a label null and ≥ 5 noise seeds.
- **[D-02] Correction to eigen-voigt-embedding [D-08]/[D-09].** "The 2–3 lead is robust to equalising
  ⟨F⟩" was a weak test: ⟨F⟩ matching changes class 2 vs 3 by only 0.02 dex (A₃/A₂ = 1.047). The 2–3
  difference cannot be a normalisation effect because classes 2 and 3 already have the same line density
  and ⟨F⟩ under every matching (A₃/A₂ 0.95–1.06).
- **[D-03] T5 — winds do not broaden lines (consistent with Nasir+2017, not Viel+2017).**
  - Δmedian b(2−1) at log N ≥ 13: −0.46 [−0.62, −0.34] km/s fiducial; +0.07 [−0.10, +0.20] ⟨F⟩-matched.
    At log N ≥ 12: +0.04 / +0.22. V17's +3 km/s is excluded.
  - Δb(3−2) is +1.5 to +1.8 km/s. N17's Δb(3−1) = +1.8 at z = 0.1 compares with our +1.3 to +1.7.
  - f(b > 50) ratio 3/2 = 1.31–1.34 (AGN adds broad lines, as N17 says).
  - Winds add strong absorbers: n(log N ≥ 14) ratio 2/1 = 1.84 fiducial, 1.47 ⟨F⟩-matched (N17: 2–5×,
    so we are at or below the low end).
  - Class 4: +6.3 km/s over log N ≥ 12 but +1.8 over log N ≥ 13, so the broadening lives in weak lines.
- **[D-04] T4a — feedback moves the bulk, not the thermal floor (CONFIRMS Bolton+2022 / Hu+2024).**
  - b_env(log N = 13.5) (10th-percentile envelope) is 19.6–20.2 km/s for every class:
    Δb_env(4−1) = −0.54 [−0.73, −0.37] fiducial, −1.03 ⟨F⟩-matched; Δb_env(3−2) = +0.19 / 0.00.
  - Bulk median b(13.5): Δ(4−1) = +2.3 / +2.5, Δ(3−2) = +1.5 / +1.3, all CIs excluding 0.
- **[D-05] T4b — temperature–density relation (HI-weighted T, 2048 skewers, block bootstrap).**
  - Cool-phase (T < 10⁵ K) T₀ = 6290 / 6311 / 6721 / 8683 K for classes 1–4, i.e. +0.3%, +6.9% and +38%
    vs class 1; γ_cool = 1.76 / 1.76 / 1.83 / 2.01.
  - Hot (T > 10⁵ K) pixel fraction near mean density (−0.5 ≤ log Δ < 0.5): 11.0 / 12.2 / 16.5 / 64.8%.
  - A naive all-gas T₀ fit gives 309,000 K for class 4 because the median lands in the hot phase.
  - Class 4 is two-phase: 10th-percentile T at mean density matches the other classes (~3,500 K), but
    61–65% of mean-density gas is above 10⁵ K, with f_HI about 10× lower. This explains its higher ⟨F⟩,
    fewer lines, extra broad weak lines and unchanged narrow floor in one picture.
  - Against Bolton+2022 (ΔT₀ ≤ 2.4% across H02/AGN/StrongAGN at z = 0.1): consistent for winds;
    WindAGN +6.9%; WindStrongAGN contradicts (+38% cool-phase plus a dominant hot phase). Caveats: z = 0.3
    vs 0.1, HI-weighted T, and run identity (our class 4 may not be Bolton's StrongAGN).
- **[D-06] T6 — paired absorbers (fixed |Δv| < 10 km/s window; 5 km/s sensitivity agrees).**
  - Matched 2→3 absorbers (74% match rate): median Δb = +0.01 km/s (the typical absorber is unchanged).
    Mean Δb = +0.68 [+0.60, +0.76]. 95% of the positive broadening comes from the top 20% of lines;
    9% broaden by > 5 km/s and 7% narrow by > 5 km/s.
  - 1→4: only 38–39% of absorbers match (strong AGN reshuffles them); median +0.02 to +0.07, mean
    +3.4 to +3.6, top-20% share 94%.
  - **EXTENDS:** feedback acts on a localised minority of absorbers, not as a population-wide shift.
    That is a line-by-line result only shared ICs allow (no published analogue).
- **[D-07] T3 — Γ/UVB degeneracy.**
  - CDDF-matched A₄/A₃ = 1.54 (CONFIRMS band [1.6, 2.3]) and A₃/A₁ = 0.89 (band [0.9, 1.1]): both just
    outside and not contradicted → INCONCLUSIVE.
  - Class 4's CDDF shape is not removable by rescaling (χ²/bin 36 vs 5–9 for classes 2–3).
  - P_F(k) shape at k > 0.04 s/km after CDDF matching: −3.3 [−4.1, −2.4]%, −5.5 [−6.4, −4.4]%,
    −9.4 [−12.1, −6.9]% for classes 2/3/4 vs 1 → **EXTENDS** (a shape channel survives matching, in line
    with the ~4–5% residual Burkhart/Khaire report for AGN).
  - CDDF slope β = 1.94–2.08, steeper than Danforth's observed 1.65 (noiseless catalog vs observed fit;
    not interpreted).
- **[D-08] T1 — noiseless matched characterisation: the 2–3 width lead SURVIVES every matching.**
  - **Ensemble primaries (paired block CIs).** Δb(3−2) at fixed N is +1.2 (12–12.5), +1.9 to +2.3
    (12.5–13), +2.6 to +3.0 (13–13.5) and ≈ 0 above 13.5. This holds for fiducial, ⟨F⟩, dndz12,
    dndz13, dndz13p5 and cddf alike, and agrees with the τ-shift prediction within the CIs, so it is not
    fitter behaviour.
  - Δb(4−1) is +7 to +9 km/s at 12–13 and +3.7 to +4.6 at 13–13.5, then negative above 13.5 (−0.7;
    −2.4 to −2.8). Under dndz12 (Khaire's range, A₄/A₁ = 1.86) it is +8.0 [+7.4, +8.6] at 12.5–13 →
    **EXTENDS** (Δb(4−1) ≥ 3 km/s at fixed N after matching). In the 12–12.5 bin the refit exceeds the
    prediction by +1.9 km/s: the fitter picks up more broad weak (hot-phase) lines at the higher class-4 A.
  - **Per-sightline binary (secondary).**

    | Run | 2-vs-3 RF BA |
    |---|---|
    | fiducial | 0.580 [0.576, 0.583] |
    | dndz12 (primary) | 0.579 [0.576, 0.582] |
    | ⟨F⟩ / dndz13 / dndz13p5 / cddf | 0.578–0.580 |
    | rescaling-only **null** | 0.516 |
    | b-only / N-only / b(N)-residual (dndz12) | 0.566 / **0.516** / 0.553 |
    | residualised on [⟨F⟩, n13, ⟨τ⟩] | 0.580 |
    | axis-held-out | 0.579 |
    | tol 0.005 (35 comps/sightline) | 0.579 |
    | tol 0.002 (53 comps/sightline) | 0.586 [0.582, 0.589] |
    | detect 12.5 | 0.575 |
    | detect 13.0 | **0.539** |

    The primary CI lower bound 0.576 > null + 0.02 = 0.536 → **SURVIVES (noiseless)**. The signal is in
    line widths at fixed N (N/count-only ≈ chance) and **concentrated in weak lines (log N 12–13)**, as
    the detect-13 drop shows. That is the observability risk T2 tests.
  - 1-vs-4 binary: 0.792 fiducial → 0.736–0.746 under every matching.
  - Fitter depth: decomposing more than twice as deeply (25 → 53 components per sightline, tol 0.002)
    leaves the 2–3 BA unchanged or slightly higher, so the lead is not an under-splitting artefact.
- **[D-09] T2 — forward-modelled (COS-like) observability: the matched feedback difference does NOT wash
  out in the sim-vs-sim limit; it needs hundreds of sightlines.** Scope: dndz12 matching (Khaire's
  12 ≤ log N < 14.5 range), tabulated COS G130M LSF (z = 0.3 Lyα would be G160M; mismatch stated), rebin
  ×3, 1% Legendre-≤2 continuum error, Gaussian noise, AICc model selection, 16384 sightlines/class.
  Results in `results/lit_claims_audit/t2_results.json`, `t2_injection_{cos,gauss}.json`.
  - **Fitter fixes before the full run (engineering, no science gate).** The first launch crashed
    (leastsq with more parameters than pixels in a 5-px region). Fixes: components capped at
    (n_pix − 2)/3 per region; each region's fit must beat the no-absorption model by the same ΔAICc
    criterion; any per-sightline exception is recorded as an empty, flagged fit. Zero crashed sightlines
    in every catalog; median χ²_ν 1.10–1.14 (0.41–0.46 at S/N ∞, error floor 10⁻³).
  - **Injection–recovery (400 single lines per class from each class's own fiducial (b, N)).** S/N ∞:
    100% complete, Δb = 0.00. S/N 30: 87–92% complete, median Δb −0.45 to +0.40 km/s (12.5–13: 80–89%).
    S/N 10: 42–48% complete (17–24% at 12.5–13). Isolated-line b is unbiased to < 1 km/s at S/N ≥ 20.
    **False lines on blank spectra** (all from the continuum-error term; pure noise gives 0.02–0.05):
    0.06 / 0.29 / 0.71 / 2.90 per sightline at S/N 10/20/30/50, of which 0.01 / 0.02 / 0.04 / 0.16 have
    log N ≥ 13.
  - **Per-sightline 2-vs-3 (pre-registered primary, S/N 30, 10 CV seeds per noise seed).** RF BA
    0.539 / 0.540 / 0.546 / 0.538 / 0.542 over 5 noise seeds (mean 0.541; min CI lower bound 0.534).
    Label null (class 2 vs class 2, different noise) 0.497 [0.493, 0.501]; rescaling-only null (class 2
    at A₃, independent noise) 0.499 [0.496, 0.502]. Lower bound 0.534 > 0.52 and > null + 0.02 →
    **EXTENDS (distinguishable per sightline under a COS-like model)**. LR agrees (0.546).
    - S/N trend: ∞ 0.581 (LSF + rebinning alone cost nothing), 50 0.550, 30 0.541, 20 0.533,
      10 0.519 [0.515, 0.523] (fails the 0.52 bar at S/N 10).
    - Sensitivities at S/N 30 all 0.535–0.547: Gaussian LSF 0.544, σ ∝ √F 0.541, ΔAIC < −10 0.541,
      b_turb ~ U(0, 20) 0.537, no continuum error 0.547, ⟨F⟩ matching instead of dN/dz 0.535.
    - Ablations: b-only 0.526, N/count-only 0.509, detect 12.5 0.538, **detect 13.0 0.535**. The
      noiseless signal sat in weak lines (detect-13 dropped it to 0.539); forward-modelled, the weak
      lines are gone and what survives is carried by log N ≥ 13 lines, which is also where false lines
      are rare.
  - **1-vs-4:** 0.602 at S/N 30 (0.748 at S/N ∞, 0.555 at S/N 10) → EXTENDS.
  - **Ensemble Δmedian b at fixed N (S/N 30, paired block CIs).** 3−2: +0.9 [0.6, 1.2] (12.5–13),
    **+3.9 [3.5, 4.4] (13–13.5)**, +1.3 [0.9, 1.8] (13.5–14), −1.0 (14–14.5). 4−1: +3.2 (12.5–13),
    **+4.8 [4.2, 5.7] (13–13.5)**, ≈ 0 (13.5–14), −1.5 (14–14.5). With no continuum error: 3−2 +3.4,
    4−1 +3.9 at 13–13.5. The forward model moves the 3−2 signal from 12.5–13 (noiseless +2.6) to 13–13.5.
  - **Forecast (statistical only; one observed sample vs a model prediction, single-sample block SE).**
    - The pre-registered estimator (inverse-variance mean of Δb over N bins) gives n₃σ = 2,800–7,000
      (3−2) and 640–1,140 (4−1) sightlines at S/N 30. It is unstable across noise seeds because Δb
      changes sign at 14–14.5 and the bins partly cancel.
    - **Post-hoc bin-resolved estimator** (Σ(Δ/SE)² = 9 over bins; stated as a deviation): n₃σ ≈
      650–700 sightlines (Δz ≈ 15) for 3−2 and 180–290 (Δz ≈ 4–7) for 4−1 at S/N 30, stable across
      seeds and sensitivities. Most informative single bin: 13–13.5, ~900 sightlines (3−2).
    - 1 sightline = one 60 h⁻¹ cMpc box, Δz = 0.0234 at z = 0.3. Systematic data-vs-model b offsets
      (observed b exceeds simulations by 8–10 km/s; C-08, C-12) are NOT included. They are larger than
      every Δb here, so an observational test needs the thermal/turbulent nuisance marginalised, not
      just more sightlines.
  - **Pre-registered secondary verdicts.**
    - Narrow-line tension (every class's median b < 30 km/s): **not confirmed** at S/N 30 (30.9 / 31.1
      / 32.6 / 34.0).
    - Noiseless→forward-modelled median-b shift ≥ 4 km/s ("consistent with C-08"): met for classes 1–3
      (+4.8 / +4.9 / +4.4; class 4 +2.2) under the pre-registered model. **This shift is
      model-dependent:** +2.4 with no continuum error, −0.5 at S/N ∞, +8.8 at S/N 50 (continuum false
      lines). It comes from selection under noise and continuum placement, not single-line bias
      (injection Δb < 1 km/s). We record it as a systematics finding, not as support for C-08.
  - **Khaire+2024a verdict (verb ceiling: "distinguishable between two simulations under a COS-like
    model with known thermal nuisances", Sherwood z = 0.3).** After dN/dz matching over Khaire's range,
    the b–N difference between feedback variants survives forward modelling, per sightline (BA 0.54–0.60
    vs nulls 0.50) and in the ensemble (Δb 3.9–4.8 km/s at 13–13.5). "Largely insensitive" is not
    reproduced as an information-content statement. Our reading: the insensitivity is a sample-size and
    systematics statement. The difference is real but small (Δb ≤ 5 km/s, narrow N windows), and an
    observational detection needs ~200–700 box-equivalent sightlines plus marginalisation of the 8–10 km/s
    thermal/turbulent offset. Caveats: different simulations (Illustris/TNG vs Sherwood), z = 0.3 vs 0.1,
    VoigtFit vs VPFIT, and a G130M LSF. → ~~EXTENDS / partially CONTRADICTS (information content),
    CONSISTENT (practical observability at current systematics).~~ **Superseded by [D-12]: CONSISTENT
    with Khaire+2024a; EXTENDS by quantification. "Contradicts" is withdrawn.**
- **[D-10] Multiplicity (Holm across the six pre-registered primaries, practical nulls per
  amendment 6).**

  | Test | Primary | Estimate | p | Holm |
  |---|---|---|---|---|
  | T1 | 2–3 BA vs rescaling null + 0.02 (noiseless, dndz12) | 0.579 vs 0.536 | < 10⁻¹⁰ (z ≈ 28) | reject |
  | T2 | 2–3 BA vs max null + 0.02 (S/N 30, worst seed) | 0.538 vs 0.519 | < 10⁻¹⁰ (z ≈ 10) | reject |
  | T3 | P_F(k > 0.04) shape residual after CDDF matching | −3.3 / −5.5 / −9.4% | < 0.002 | reject |
  | T4 | Δb_env(4−1) ≠ 0 | −0.54 [−0.73, −0.37] | < 0.002 | reject (|Δ| < 1 → CONFIRMS-Bolton outcome stands) |
  | T5 | Δb(2−1) ≠ 0, log N ≥ 13 | −0.46 fiducial / +0.07 ⟨F⟩ | < 0.002 / 0.42 | reject / retain (neither has V17's sign) |
  | T6 | paired median Δb(3−2) ≠ 0 | +0.010 [0.007, 0.012] km/s | < 0.002 | reject (statistically ≠ 0, practically zero) |

  Bootstrap p-values have resolution 0.002 (500 resamples); the Holm threshold for the smallest p is
  0.0083.
- **[D-11] Pre-registration of the E1 nuisance-degeneracy criterion (written 2026-10-20 before the E1
  results were opened; the E1 catalogs were still being built).** Test: class 2 (class 1) at dndz12 plus a
  fixed Gaussian broadening b_x ∈ {3, 6, 9, 12, 15, 18, 21} km/s, unknown to the fitter, forward-modelled
  at S/N 30 (noise seed 0), against class 3 (class 4) at S/N 30.
  - **DEGENERATE at b_x** if both hold: (i) the binned Δmedian b profile over the five N bins has χ²
    p > 0.05 at the full 16384 sightlines, **or** p > 0.05 after scaling χ² to the forecast sample
    (χ² × 650/16384 for 3−2, × 250/16384 for 4−1); and (ii) the per-sightline RF BA (all features) CI
    lower bound is ≤ 0.52.
  - **NON-DEGENERATE (fixed-broadening nuisance only)** if no b_x meets the full-sample criterion. The
    forecast-sample criterion is reported alongside.
  - **Declared weakness (defense panel, attack 4):** a uniform quadrature broadening only adds b, so it
    cannot reproduce the sign change of Δb across N (+ at 12.5–13.5, − at 14–14.5). Its nuisance enters
    only the class-2 (class-1) side and is not re-matched in dN/dz. "Non-degenerate" here is therefore
    weak evidence. It excludes only the simplest thermal-like nuisance (a T₀-like offset), not a T₀/γ
    shift (attack 1). That requires τ re-synthesis from the LOS fields with swapped temperatures, which is
    not run in this track.
- **[D-12] Defense-panel review of EDGE_MAP.md: NEEDS WORK. Accepted corrections:**
  - **Khaire+2024a (attack 3, accepted in full).** Khaire's "insensitive" is conditional on their data:
    34 QSOs, Δz = 1.79, S/N > 10, VPFIT, G130M (from the arXiv HTML; still to verify against the PDF).
    At Δz = 1.79 (≈ 76 boxes), our own forecast gives ≈ 1.0σ (post-hoc estimator) or ≈ 0.5σ
    (pre-registered) for 3−2. **Verdict: CONSISTENT. EXTENDS** by quantifying, in Sherwood at z = 0.3, a
    fixed-N residual that would need Δz ≳ 15 at S/N 30 per 7.9 km/s pixel (≈ 40 per resolution
    element), before nuisance marginalisation.
  - **Forecast (attack 5).** Both estimators are quoted together wherever either appears. Amendment 6's
    two-sample factor 2 was dropped: the reasoning (one observed sample vs a model) is stated, but this
    is a deviation, and with the factor n₃σ doubles (≈ 1,300 / Δz ≈ 30 for 3−2). The 192 blocks are 3
    projection axes × 64 cells, and the axes cross the same structures, so SEs may be low by up to √3
    and n₃σ low by up to 3×. Bins share skewers (no covariance used). Δb at S/N 30 is inflated relative
    to the LSF-only value (3−2 at 13–13.5: 2.9 → 3.9).
  - **Verb ceiling (attack 6).** The 13 ≤ log N < 13.5 location was chosen after the forward model had
    moved the signal there, and it was never a primary. It is stated at "observed". T4/T6 "unchanged"
    claims are equivalence claims tested as ≠ 0: the ⟨F⟩-matched Δb_env(4−1) = −1.03 fails a ±1 km/s
    bound, so the **T4a CONFIRMS label is downgraded to "consistent, borderline"**. The T4a bar also
    required Δmedian b ≥ 4 km/s, and the bulk at log N = 13.5 is +2.3 / +2.5.
  - **Contrasts (attack 7).** 4−1 is not "AGN only": NoFeedback may also differ in star-formation
    prescription (unverified). The clean AGN contrasts are 3−2, 4−2 and 4−3; their fixed-N values are
    computed under the mechanism check.
  - **Pending panel items run in this track:** mechanism attribution (attack 1a), transfer function
    (attack 2), suite identity (attack 7) and a combined realistic instrument (attack 8).
  - **Not run** (recorded as open): τ re-synthesis T₀/γ swap and the [T₀, γ, A] fit (attack 1b/c);
    forward-space dN/dz re-matching to Danforth (attack 9); a phase-randomised Δτ perturbation null
    (attack 10); 3D-subcube jackknife and nuisance-marginalised Fisher forecast (attack 5).
- **[D-13] PI-authorised edge checks E2, E3, E4 (`results/lit_claims_audit/edge_checks.json`).**
  - **E2: jitter null for the T6 paired response (noiseless, |Δv| < 10 km/s, log N ≥ 12).**

    | Comparison | mean Δb | \|Δb\| > 5 | Δb > 5 / < −5 | top-20% share |
    |---|---|---|---|---|
    | class 2 → 3 (fiducial) | +0.68 | 15.8% | 9.0% / 6.8% | 0.951 |
    | class 2 → 3 (dndz12) | +0.80 | 16.2% | 9.5% / 6.7% | 0.948 |
    | class 1 → 4 (fiducial) | +3.55 | 29.2% | 18.8% / 10.4% | 0.943 |
    | null: class 2 at A₂ vs A₃ (5% τ refit) | +0.05 | 2.7% | 1.5% / 1.2% | **0.996** |
    | null: class 2 fiducial vs tol 0.005 | −0.39 | 11.6% | 4.7% / 6.9% | 0.971 |

    - The **localisation statistic is uninformative**: the jitter null scores 0.996, higher than the real
      pair. The T6 EXTENDS rule was unfalsifiable and the "~20% of lines carry the broadening" reading is
      **retracted**.
    - What survives: the paired response is about 6× more frequent than refit jitter at matched τ
      amplitude (|Δb| > 5 in 16% vs 2.7%). It is **two-sided** (9% broaden, 7% narrow) with net mean
      +0.7 km/s. The perturbation-matched null (attack 10) and chaotic divergence (Genel+2019) are not
      separated; the paired ΔT test runs under the mechanism check.
  - **E3: cool-phase T₀ vs the temperature cut (2048 skewers, block CIs; volume and density weighting
    agree).**

    | Class vs 1 | cut 10^4.5 | 10^4.75 | 10^5 | 10^5.5 | mode at Δ≈1 | 10th pct at Δ≈1 |
    |---|---|---|---|---|---|---|
    | 2 | +0.2% | +0.2% | +0.3% | +1.1% | 0 | +0.3% |
    | 3 | +0.7% | +2.7% | +6.9% | +18% | 0 | +1.2% |
    | 4 | **+2.7% [2.0, 3.4]** | +11% | +38% | +358% | 0 | +20% [17, 23] |

    The D-05 "+38% cool-phase T₀" was warm gas leaking under the 10^5 K cut. With a 10^4.5 K cut,
    every class is within +2.7% of class 1, **consistent with Bolton+2022's ≤ 2.4%**, and the mode does
    not move (the mode's resolution is ±4.7%). The robust class-4 differences are the hot-phase fraction
    (D-05) and a 20% warmer 10th percentile at mean density. **D-05's "contradicts Bolton for
    WindStrongAGN" is withdrawn.** It is replaced by: feedback leaves the cool T–ρ normalisation within
    ≈ 3% and adds a warm/hot tail whose size depends on the class.
  - **E4: P_F(k) shape after the COS-like forward model (CDDF-matched, 4096 skewers, noise power
    subtracted).** At k > 0.04 s/km: −2.4 [−3.4, −1.3] / −4.4 [−5.5, −3.0] / −6.8 [−9.4, −4.3]% for
    classes 2/3/4 vs 1 at S/N 30, the same as at S/N ∞ (−2.4 / −4.6 / −7.0) and close to noiseless D-07.
    At k 0.01–0.03: +7.8 / +2.5 / +8.6%. It survives the forward model but is a replication (Khaire,
    Burkhart). The ratio is largely LSF-invariant, and significance against COS error bars is not
    assessed → **demoted** to replication.
- **[D-14] E1 nuisance degeneracy (scored against the [D-11] pre-registration).** Class 2 (class 1)
  plus a fixed broadening b_x, forward-modelled at S/N 30, compared with class 3 (class 4).

  | b_x (km/s) | 3−2 profile χ²/5 | 3−2 BA (CI lo) | 4−1 profile χ²/5 | 4−1 BA (CI lo) |
  |---|---|---|---|---|
  | 0 | 353 | 0.539 (0.535) | 474 | 0.601 (0.596) |
  | 6 | 205 | 0.540 (0.535) | 296 | 0.614 (0.610) |
  | **9** | **178** | 0.575 (0.570) | **213** | 0.637 (0.633) |
  | 12 | 478 | 0.631 | 278 | 0.670 |
  | 21 | 4282 | 0.780 | 2355 | 0.784 |

  - **Pre-registered verdict: NON-DEGENERATE** for the fixed-broadening nuisance at full-sample
    precision (no b_x gives p > 0.05 together with a BA lower bound ≤ 0.52).
  - **At the forecast sample the binned b–N profile IS degenerate.** At b_x = 9 km/s, the χ² scaled to
    650 sightlines (3−2) is 7.1 (p = 0.22). Scaled to 250 sightlines (4−1) it is 3.3 (p = 0.66).
  - With this one-parameter nuisance free, a 3σ-equivalent profile detection (χ²₅ = 18.2) needs about
    **1,700 sightlines (Δz ≈ 39) for 3−2 and 1,400 (Δz ≈ 33) for 4−1**. That is 2.6× and 5.6× the
    no-nuisance forecast, and still optimistic: one-sided nuisance, 3 km/s grid, no T₀/γ freedom.
  - b_x ≈ 9 km/s is the size of the observed data-vs-simulation b offset (8–10 km/s), so the realistic
    nuisance sits exactly where it does the most damage.
  - **What the nuisance cannot absorb.** The residual stays peaked at 13 ≤ log N < 13.5: +2.1 (3−2) and
    +3.0 (4−1) at b_x = 9, while the neighbouring bins go negative. A uniform quadrature broadening
    cannot make an N-localised bump, so the N shape of Δb is the separable part. Its separability from a
    T₀/γ (N-dependent) nuisance is untested.
  - Per-sightline BA rises with b_x (broadening also changes line counts), so BA is not a degeneracy
    metric here. The profile χ² is.
- **[D-15] Defense-panel follow-ups: mechanism, transfer function, suite identity, realistic instrument
  (`edge_checks.json` → m1, x2, x7, x8).**
  - **Mapping validated.** A thermal-only τ synthesis from the LOS fields (ρ·f_HI weight, u = x·Δv +
    v_pec, b_th from the HI-weighted T) reproduces the stored τ shape with a mean log-correlation of
    1.000 (class 1) and 0.998 (class 4) at a lag of +1 pixel, which is the known roll(exp(−τ), 1). Each
    component gets a τ-weighted gas temperature T_τ, and b_th = 0.128 √T_τ.
  - **Mechanism (panel attack 1a): the fixed-N width difference is thermal.** Δmedian b equals Δmedian
    b_th in every N bin (paired block CIs):

    | Pair | Catalog | 12.5–13 | 13–13.5 | 13.5–14 | 14–14.5 |
    |---|---|---|---|---|---|
    | 3−2 | noiseless: Δb / Δb_th | +2.3 / +2.5 | +3.0 / +3.0 | +0.1 / +0.5 | −0.3 / 0.0 |
    | 4−1 | noiseless | +8.0 / +7.0 | +3.7 / +4.4 | −0.7 / +0.1 | −2.4 / −2.6 |
    | 3−2 | S/N 30 | +0.9 / +0.8 | +3.9 / +2.8 | +1.3 / +1.5 | −1.0 / +0.1 |
    | 4−1 | S/N 30 | +3.2 / +3.1 | +4.8 / +4.9 | +0.1 / +1.0 | −1.5 / −2.0 |

    - At fixed N, class-3 absorbers' gas is +0.10 to +0.12 dex hotter than class 2's at 12.5–13.5
      (median T_τ 34,300 → 43,400 K at 13–13.5). The fraction above 10^4.5 K rises from 0.54 to 0.62.
      Class 4 is +0.15 to +0.29 dex hotter than class 1.
    - The G7 sign change is also thermal: class-4 gas at 14–14.5 is 0.08 dex cooler than class 1's.
    - The thermal envelope (D-04) stays put while the median moves, and the cool T–ρ normalisation near
      mean density moves by ≤ 3% (D-13 E3). Feedback **heats part of the moderately overdense absorbing
      gas** (a warm tail at fixed N), rather than shifting the T–ρ power law near mean density.
    - **Consequence (accepted from attack 1):** this is the thermal state, the quantity observational
      b–N analyses fit as a nuisance. An observer fitting a [T₀, γ] power law to b–N would likely see a
      steeper γ or a warm tail. Whether a power-law fit absorbs it fully is untested (no τ re-synthesis
      with swapped T).
    - Contrasts: 4−2 at 13–13.5 is +3.9 noiseless and +4.6 at S/N 30 (Δb_th +4.4 / +5.1). 4−3 is +0.9 /
      +0.7 (Δb_th +1.4 / +2.3). Both are thermal as well.
  - **Paired per absorber (noiseless 2→3).** Δb correlates only weakly with Δb_th (r = 0.25; 5.6% of the
    variance), so per-absorber changes are mostly decomposition and kinematics. Broadening tails are
    hotter: P(ΔT > 0 | Δb > 5) = 0.80 against a base rate of 0.64, with a median +0.11 dex. Narrowing
    tails are not cooler: P(ΔT < 0 | Δb < −5) = 0.37 against a base of 0.36. For 1→4: r = 0.51, 26%,
    P = 0.91. **Broadened absorbers in the feedback run are the heated ones; narrowed absorbers are
    re-decomposition or kinematics.**
  - **Transfer function (attack 2).** Matching noiseless dndz12 components to S/N-30 components per
    class: at fixed true (N, b), detection fraction and b bias are class-independent for 3−2 (within
    2 percentage points and 0.26 km/s, except one sparse cell). For 4−1 they are not: class 4 is detected
    1–10 points less often. The observation-stage selection depends strongly on b (12.5–13: 74%
    detected at b < 20, 28% at b > 35) but is the same across classes 2 and 3. So the noise-stage
    change in where Δb sits in N (2.9 → 3.9 at 13–13.5) is a class-blind selection acting on different
    true b distributions, not a class-dependent fitter artefact.
  - **Suite identity (attack 7).** LOS headers are identical across classes (cosmology, box). Void-pixel
    (Δ < 0.3) log-density correlation with class 1 is r = 0.991 / 0.891 / 0.501 for classes 2/3/4, so
    shared ICs hold. AGN, strong AGN especially, rearranges even void gas. The star-formation-prescription
    question for NoFeedback stays open (no run tables).
  - **Combined realistic instrument (attack 8):** Gaussian FWHM 19 km/s, S/N 15 per pixel, σ ∝ √F, 2%
    continuum error, b_turb ~ U(0, 20). 2-vs-3 RF BA is **0.524 [0.520, 0.527]**, b-only 0.520, so it
    **fails the > 0.52 bar (lower bound = 0.520)**. 1-vs-4 BA is 0.566 [0.561, 0.571]. Δb(3−2) at 13–13.5
    is still +3.5 in the ensemble, and median b is 29.9–31.1 km/s. Per-sightline 2–3 discrimination
    does not survive realistic combined conditions. The ensemble residual does, before nuisance
    marginalisation.

---

## 4. Data
| Area | File | Metadata |
|---|---|---|
| Matching factors | `results/lit_claims_audit/matching_factors.json` | A_c per variant + iteration log |
| Matched catalogs | `results/lit_claims_audit/artifacts/catalog_{dndz12,dndz13,dndz13p5,cddf,null2_dndz12,tol005,tol002}_full/` | to DVC-track |
| Test results | `results/lit_claims_audit/t{1,3,4a,4b,5,6}_*.json` | git-tracked |
| LOS fields | `data/raw/Sherwood/Physics*/los2048_n16384_z0.300.dat`, `tauwH1_*` | DVC (pulled 2026-10-19) |
| COS LSF | `data/raw/Sherwood/COS_LSF1_cw1327_75_z0.100.dat` | (offset, value); G130M; core FWHM ≈ 13 km/s, 12.6% power beyond ±20 km/s |

## 5. Evaluation
See `CLAIM_TEST_PLAN.md`: pre-registered outcomes, the amendments, and Holm across practical-null primaries.

## 6. Visualization & Artifacts
- Pending (after T2).

## 7. Session History & Next Handoff
### Session Snapshot: October 19–20, 2026
- Literature survey, PI plan and panel review. T3, T4a, T4b, T5, T6 complete; T1 complete except
  tol 0.002; T2 queued behind it.
### Next
1. T2 forward-model grid → the Khaire verdict and an observability forecast.
2. Synthesis: gap map / edge (PI), defense-panel on the result.
