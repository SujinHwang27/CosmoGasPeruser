# SCOPING — linewidth-cddf (observable-family pivot from fps-sbi, cascade-inheriting from pk-feedback-classifier)

**Proposed track:** linewidth-cddf (per-line b-parameter distribution + column density distribution function — primary observables extracted from τ, not from flux-derived P_F(k))
**Proposed branch:** `exp/linewidth-cddf` (off `main`; sibling to closed `exp/pk-feedback-classifier` and shelved `exp/fps-sbi`)
**Motivating decisions:** `experiments/pk-feedback-classifier/LEDGER.md` [D-26] (cascade-exhaustion close); `experiments/fps-sbi/Q_S1_PARAM_AUDIT.md` verdict (B) (Sherwood discrete-only); `experiments/fps-sbi/Q_S3_LIT_ANCHOR.md` (Sinigaglia+2026 prior-recovered posteriors).
**Status:** DESIGN PROPOSAL — no LEDGER, no branch, no compute committed by this document.

This is the **Path C** scoping pass authored at user request: a fundamentally different observable family (line widths + CDDF from τ, anchored on Tillman+2023 line-broadening / CDDF-redistribution framing) replacing the fps-sbi P_F(k)-SBI proposal that Q-S1+Q-S3 returned structurally weakening on. Per fps-sbi SCOPING §5 alternative path (b), Path C is the rule-2-clean re-entry candidate after both same-observable methodology pivots (classification + SBI on P_F(k)) closed against the cascade.

---

## §1 — Cascade-inherited confidence framing (rule 2)

The inheriting cascade, recorded honestly per [D-37]:

1. `signal-clustering-v2` [D-13] — per-sightline feedback separability is information-limited at the 24-dim RBF-SVM separability-vector basis. FALSIFIED the "richer per-sightline features rescue separability" prior at high confidence.
2. `pk-feedback-classifier` [D-01] (high-k thermal-cutoff prior) — FALSIFIED by Stage 0 [D-07].
3. `pk-feedback-classifier` [D-10]/[D-15] (mid-k line-width-regime prior, with Khaire+2024 / Tillman+2024 partial defense) — FALSIFIED by Stage 1 [D-26] (G3+G6 joint FAIL).
4. `fps-sbi` SCOPING §1 — pre-emptively closed without LEDGER opening: Q-S1 verdict (B) (Sherwood discrete-only, no continuous-θ grid) collapsed SBI to a 4-class likelihood-ratio test mathematically reparameterizing [D-26]'s `predict_proba`; Q-S3 returned Sinigaglia+2026 prior-recovered posteriors at z = 2–3.5 on CAMELS-LH-1000 — the published analogue of the proposed deliverable is itself a NULL-style upper-bound bar.

### Cascade-reset or cascade-inherit? Argue both sides.

**The cascade-reset argument** (the case for treating linewidth-cddf as a fresh track at moderate confidence):
- **Different observable.** Line-width b-parameters and CDDF are local, peak-based, line-by-line statistics. P_F(k) is a global Fourier statistic of the same field. These are mathematically distinct reductions: a forest with the same P_F(k) can have very different b-distributions and column-density distributions (e.g., narrow-blended-many vs broad-isolated-few lines can produce comparable Fourier power). The [D-26] G3+G6 falsification was specifically of *which k-bins drive RF importance*; it is not a falsification of *whether per-line statistics carry feedback information*.
- **Different published prior.** Tillman+2023 (ApJL 945, L17 / arXiv:2210.02467) explicitly anchors low-z AGN-feedback discrimination at **line broadening + CDDF redistribution**, NOT at P_F(k) shape. This is a published prior on the linewidth-cddf observable family that did NOT exist as a P_F(k)-band prior for the mid-k re-anchor in [D-15]. Khaire+2024 / Tillman+2024 predicted ΔP_F(k) magnitudes, never that mid-k *importance* is mechanism-bearing; Tillman+2023's framing is directly on b and N_HI.
- **Q-S1 (B) doesn't bite.** The Sherwood discrete-only verdict collapsed the SBI deliverable because SBI's premise is a posterior on continuous θ; the linewidth-cddf deliverable is a **per-class b-distribution or per-class CDDF comparison**, which is well-defined on 4 discrete recipes (it is the same shape as comparing 4 histograms / 4 KDEs / 4 dN/dN_HI curves). Q-S1's structural weakening of SBI does NOT transfer.
- **Sinigaglia+2026 doesn't bite.** Their null is on P_F(k) → continuous-θ posterior recovery. Their result is silent on whether the same simulations' line-statistic histograms cluster by feedback recipe.

**The cascade-inherit argument** (the case for inheriting two downgrades):
- **Same `tau.npy` input.** `tau.npy` per class is the same on-disk array P_F(k) was computed from (F ≈ e^{−τ}; the Sherwood pipeline went τ → F directly). Information-theoretically, **any deterministic function of τ has no more information about feedback than τ itself has** — and P_F(k) is a near-injective deterministic function of F on long sightlines. So linewidth-cddf is *another reduction of substantially the same input information*. The [D-13] / [D-26] information-ceiling reading at the per-sightline level should be **expected to re-manifest** on a per-sightline b-distribution or per-sightline N_HI list. The cascade-relevant question is whether *aggregating-across-sightlines into a population histogram* of b or N_HI behaves differently from *aggregating-into-stacked-P_F(k)* — and that aggregation is mathematically a different operation, but the input information is the same.
- **C2/C3 null is information-ceiling-coloured.** [D-26] G4 FAIL at C2-vs-C3 binary balanced_acc median 0.347 ≈ binary chance, even at M = 256 stacks, on the P_F(k) basis. The hypothesis that "C2 = StellarWind and C3 = WindAGN produce statistically distinguishable Lyα forest signatures at z = 0.3 at all" is currently under at-least-moderate empirical doubt. A linewidth-cddf observable that delivers the same C2/C3 null is the second independent observable's worth of the same finding — which is scientifically valuable, but the cascade weakened the *prior that this distinction is recoverable from the Lyα forest at z = 0.3 at all*, regardless of observable. Tillman+2023 anchors the *direction* (the CDDF should redistribute) but not the *recoverability under our specific suite + classifier setup*.
- **Author-curated-typology trap (rule 8).** "We tried P_F(k) bands, that closed; line widths + CDDF is a fundamentally different observable" is structurally a 3-point author-curated typology of "the observable axis." Without a formal decomposition criterion for "different observable family," we cannot claim cascade-reset by axis-coverage; we can only claim "third specific intervention" per rule 8 path (b). Two prior interventions in this lineage produced congruent failure signatures (per-sightline info-limited, C2/C3 null). The honest framing of a third intervention is "first test of whether line-by-line statistics escape the failure signature observed on P_F(k)" — verbed at LOWER, not equal, confidence to the [D-10] mid-k re-anchor, because we now have one MORE prior falsification than [D-10] did at its kickoff.

### PI judgement call (binding on §1's verb ceiling)

**Cascade INHERITS — one-level downgrade from [D-10]'s starting confidence, not a full reset.** Rationale:

The cascade-reset argument's strongest plank (different observable, different published prior) is real but partial: Tillman+2023 anchors *direction of redistribution* — it does NOT predict that linewidth or CDDF *is recoverable as a classifier feature* on a 4-recipe sibling-runs Sherwood suite at z = 0.3. We are again in the position [D-15] / [D-22] documented: a published prior on signal-existence direction (Tillman+2023), no published prior on multi-class classifier recoverability on this suite at this z. The K1-HARKing-adjacency attack that the pk-feedback-classifier defense panel raised on [D-10] (and which [D-15] only partially defused) applies symmetrically here.

The cascade-inherit argument's strongest plank (same `tau.npy` input) is also real but partial: information-theoretically τ → b-distribution is a many-to-one map that *destroys phase information* P_F(k) preserves — but the destruction direction is the wrong direction (information loss can never make a problem easier under a Bayes-optimal classifier; it can only make it easier under a *suboptimal* classifier that was distracted by the discarded information). The honest reading: line-statistics-as-features is at best on par with P_F(k) at an information bound, and the *practical* gain (if any) comes from the RF being a finite-feature suboptimal estimator that benefits from the better feature engineering encoding line-blending + line-width physics directly — which is a real engineering effect, but not one that resets the falsified-prior cascade on the underlying physics question.

**Confidence ceiling on the linewidth-cddf headline claim, verbed:**

> "First multi-class classifier test of the Tillman+2023 line-broadening / CDDF-redistribution framing as a discriminator of the 4-recipe Sherwood feedback suite at z ≈ 0.3 (extrapolated from Tillman+2023 z = 0.1); highest-leverage of the remaining candidates on the observable-family axis given the prior P_F(k)-band cascade falsifications, but with the C2/C3 separability expected on physical grounds to re-manifest as a null under the information-ceiling reading of [D-13]/[D-26]."

**NOT admissible** as headline:
- "Line widths recover feedback classes from the Lyα forest at z ≈ 0.3."
- "CDDF redistribution distinguishes AGN feedback recipes."
- "We recover Tillman+2023 line-broadening signatures via classification."

The verbs "recover" / "recovers" / "distinguishes" are reserved until a result with C2/C3 separability above the [D-26] G4 floor + an external-anchor comparison lands. Until then: "first test of," "first multi-class classifier comparison," "consistent / inconsistent with Tillman+2023 direction."

### Falsified-prior ledger line (rule 4):
- `signal-clustering-v2` [D-13] (per-sightline information ceiling at 24-dim basis) — FALSIFIED at high confidence; expected to re-manifest at per-line-statistic basis.
- `pk-feedback-classifier` [D-01] (high-k mechanism on P_F(k)) — FALSIFIED, [D-07].
- `pk-feedback-classifier` [D-10]/[D-15] (mid-k mechanism on P_F(k)) — FALSIFIED, [D-26].
- `pk-feedback-classifier` [D-26] (C2/C3 binary distinguishability on stacked P_F(k) at M ≤ 256) — FALSIFIED to ≈ binary chance.
- `fps-sbi` SCOPING (continuous-θ SBI on Sherwood P_F(k)) — pre-emptively closed by Q-S1 (B) + Q-S3 (Sinigaglia+2026 prior-recovered).

### Anti-degeneracy audit (rule 3)

What does a "per-class b-distribution KS-test" or "per-class CDDF χ² difference" or "RF classifier on (b, N_HI) features" metric leave unconstrained when the supervision signal is weakly informative on the C2/C3 majority of the comparison?

- **For (i) b-distribution.** A KS / Anderson-Darling test on the b-distribution per class is overwhelmingly dominated by the *bulk* of the b-distribution (b ≈ 15–40 km/s thermal-Doppler core); the diagnostic tail (broad, b > 60 km/s, turbulent / heated by feedback) is a small fraction of detected lines and KS is insensitive to tail differences. A "significant KS" between C1 and C4 may be carried entirely by the C4 ⟨F⟩ offset that [D-26] G5 / [D-07] already characterized as a known confounder — the line-detection pipeline itself is ⟨F⟩-sensitive (a 1% mean-flux offset shifts the detection threshold and changes which lines are detected at all). Mitigation must be designed in (Q-C1 / Q-C2 owner).
- **For (ii) CDDF.** dN/dN_HI is the canonical published statistic (Tillman+2023's framing), and the test most-naturally-anchored. But CDDF is normally reported per unit comoving redshift path-length Δz, and our Sherwood snapshot is a single z = 0.30–0.32 slice — the per-class CDDF normalisation will need careful path-length accounting (line density per simulation sightline → published Δz units) before Tillman+2023's effect-size numerics can be compared to ours. Until that normalization is closed, the bar is rule-14-fragile (project-internal normalisation choice produces project-internal effect size).
- **For (iii) joint (b, N_HI).** The b-N envelope is the classical thermal-state diagnostic but its discriminatory power is *primarily on the thermal-state low-density-IGM regime*, and the [D-26] regime-parity finding (per_sightline ↔ global identical-within-noise → DC mean-flux already suppressed by per-sightline normalization) suggests the thermal-state signal at z = 0.3 is what's already-suppressed by Sherwood's photoionization-equilibrium IGM. The b-N envelope's *feedback-driven* shifts at z = 0.3 (CGM heating signatures, b-tail extension) are precisely the small-tail diagnostic that the supervision signal is weakly informative on.
- **The cascade-coloured failure mode.** The most-likely degeneracy is **"the b-distribution and CDDF show a strong C4-vs-others lift and a null on C2-vs-C3, exactly mirroring [D-26]."** Pre-committing a C2/C3 sub-gate (analogous to [D-26]'s G4) is mandatory.

### Prior-failure ledger and verb-ceiling enforcement (rule 4)

Every linewidth-cddf spec must carry, on the face of every claim in paper text and LEDGER §3:
- "(extrapolated from Tillman+2023 z = 0.1)" hedge — composes additively with the z ≈ 0.3 hedge.
- "(C2/C3 null expected on physical grounds per [D-26] cascade; this work tests whether it re-manifests on the linewidth-cddf observable family)" pre-commit on the C2/C3 axis.
- Reservation of the verb "recover" / "distinguish" until the C2/C3 sub-gate clears.

---

## §2 — Three open questions, scoped (NOT decided)

### Q-C1 — Does the τ-to-line-stats extraction pipeline exist? (data-engineer)

**Question.** Sherwood `tau.npy` per class is on disk at `data/preprocessed/Sherwood_z0.3_inf/{1..4}/tau.npy`, shape (16384, 2048) float64 (per pk-feedback-classifier LEDGER §4 + Q_S1 audit §1.1). To extract the linewidth-cddf observables we need:

- **(a) Line detection.** Peak-find local maxima in τ above a threshold (e.g., τ_min = 0.05 or equivalent flux decrement δ_F < 1 − exp(−0.05) ≈ 4.9%), with a deblending pass to handle visually-overlapping line complexes — e.g., the AUTOVP-style or VPFIT-style automatic line-fitting protocols.
- **(b) b-parameter extraction.** For each detected line, fit a Voigt profile (or its weak-line Gaussian-Doppler limit for τ_max < ~1) to recover the Doppler b parameter (km/s) and the line centre. Anchor literature for the algorithm: Hu et al. 1995 / Kirkman & Tytler 1997 / Schaye et al. 1999 / VPFIT (Carswell & Webb 2014). The Voigt convolution kernel uses the Hjerting function or a numerical integration of the Voigt profile.
- **(c) N_HI extraction.** Column density N_HI from line strength: in the optically-thin regime N_HI is the integral of τ across the line; in the saturated regime (τ > ~2) N_HI is extracted from the damping wings if present, or treated as a lower-limit. For Lyα at z ≈ 0.3 the cross-section + oscillator-strength conversion is f = 0.4164, λ_0 = 1215.67 Å, A_21 = 6.265×10⁸ s⁻¹.

**Inventory of existing infrastructure (this scoping pass):**
- `src/core/transforms.py` contains `PCATransform`, `DCTTransform`, `WindowedDCTTransform`, `WaveletTransform`, `FisherTransform`. **None of these are line-detection or Voigt-fitting.** No `voigt` / `Voigt` / `line_detect` / `column_density` / `N_HI` / `b_parameter` / `Doppler` / `CDDF` symbols anywhere in `src/core/` (Grep confirmed this scoping session).
- `src/core/data.py` `DataIngestor` is `filename`-parameterized; loading `tau.npy` is one-line (`DataIngestor(base_path=..., filename="tau.npy")`).
- No `pyVoigt` / `pyVPFIT` / `linetools` / `astropy.modeling.Voigt1D` import in the project. `scipy.special.wofz` (the Faddeeva function, basis of the Voigt convolution) is available via the scipy dependency.
- The `pk-feedback-classifier` `FluxPowerSpectrum` transform was the prior bespoke physics-observable transform; the linewidth-cddf observables would follow that same shape — a new `LineDetector` + `VoigtFitter` + `CDDFCompiler` module under `src/core/` or under `experiments/linewidth-cddf/`.

**Verdict (this scoping pass, pre-Q-C1 dispatch):** **green-field implementation.** No code re-use from prior tracks. The 24h Juno wallclock raised in pk-feedback-classifier [D-25] is NOT applicable — the line-fitting pipeline has very different compute characteristics (n_lines × Voigt-fit ≈ tens of seconds per sightline rather than RF training) and needs its own compute-envelope estimate.

**What the Q-C1 owner must return.**
1. Empirical line-detection feasibility check: load `data/preprocessed/Sherwood_z0.3_inf/1/tau.npy` and `4/tau.npy`, pick 10 random sightlines per class, run a quick peak-find at τ_min = {0.05, 0.1, 0.3} and report the per-sightline line counts. Anchor against published expectation: at z ≈ 0.3 with a 5400 km/s sightline span, published Lyα line densities are ~10–50 lines/sightline above δ_F = 0.05 (Davé+1999, Williger+2010). If we get 0–2 lines per sightline at the lowest threshold, the data are signal-limited and the track is short-cycle (kill before any spec).
2. Voigt-fitting library landscape audit: identify whether to use `astropy.modeling.Voigt1D`, a custom `scipy.special.wofz`-based fitter, or wrap an external tool (`linetools`, `VPFIT`-via-subprocess). Return one preferred + one fallback, with engineering-cost estimate (LOC + dependency cost) for each.
3. b-parameter retrievability check on synthetic data: generate 10 Voigt profiles at known (b, N_HI) on a 2048-pixel grid at Δv = 2.6365 km/s, add noise consistent with Sherwood's mean-preserving LSF/noise (per pk-feedback-classifier LEDGER §2), fit, recover (b, N_HI). If recovered b is biased > 10% or noise-floor > 5 km/s in b on synthetic, the pipeline is "well-defined but signal-limited in practice" and the track is short-cycle.

**Owner.** `data-engineer` (file inventory + library landscape) escalating to `core-implementer` (synthetic-recovery feasibility check) if the line-density returns are encouraging. BLOCKING gate per §4.

### Q-C2 — Observable definition (PI scope)

Three candidates for the primary deliverable, with rationale and recommendation:

- **(i) Per-line b-parameter distribution per class** (1D histogram / KDE of the full line population per class; per-class summary: median b, IQR, broad-line tail fraction f(b > 40 km/s), f(b > 60 km/s)). Most-directly-anchored against Tillman+2023's "line broadening" framing. Mathematically simplest; KS / Anderson-Darling test between class histograms gives a single-number per-pair effect size. **Weakness:** discards N_HI (the bulk of the published-prior discriminator in Tillman+2023's framing is in CDDF redistribution, not pure b-distribution shift).
- **(ii) CDDF dN/dN_HI per class** (column-density distribution function, log-bins from log N_HI = 12 to log N_HI = 16, with the Lyman-α-forest regime mostly at log N_HI = 13–15). Most-directly-anchored against Tillman+2023's "CDDF redistribution" framing. Tillman+2023 / Sorini-Davé-Anglés-Alcázar 2020 / Burkhart+2022 all report CDDF effect sizes. **Weakness:** the Δz path-length normalisation needs careful accounting per §1 anti-degeneracy audit before published numerics are apples-to-apples; the AGN-feedback discriminator in CDDF is mostly at log N_HI > 14 (the high-column tail), which is a small fraction of detected lines and statistically noisy in a 16384-sightline-per-class budget.
- **(iii) Joint (b, N_HI) 2D distribution per class** (2D histogram on (log N_HI, b) grid; per-class summary: 2D KDE comparison via earth-mover-distance / Wasserstein-2). Captures the classical b-N envelope diagnostic; the envelope's *upper* boundary in b at fixed N_HI shifts under thermal-state changes / shock-heating from feedback. **Strength:** the *most physically interpretable* — feedback effects are encoded in shifts of the upper boundary, not in the bulk shape, and a 2D test sees both b-shift and CDDF-shift simultaneously. **Weakness:** 2D-test statistical power per-pair is lower than 1D for the same N, and the comparison statistic (Wasserstein, energy distance) is less standard in the IGM literature than CDDF.

**PI recommendation (preliminary, subject to Q-C1/Q-C3 returns):**

> **Recommend (ii) CDDF dN/dN_HI per class as the primary deliverable, with (iii) joint (b, N_HI) 2D distribution as a paired diagnostic arm. (i) b-distribution-alone is REJECTED at scoping.**

Rationale:
- (ii) is the **literature-anchored** observable. Tillman+2023 / Sorini-Davé-Anglés-Alcázar 2020 / Burkhart+2022 all report CDDF numerics. Q-C3's anchor search should converge most naturally to a CDDF effect-size bar.
- (iii) is a near-zero-marginal-cost addition once the line catalog (b, N_HI) per line is in hand — the same catalog feeds both (ii) and (iii). Pairing them lets the headline be "CDDF effect size [X], joint (b, N_HI) Wasserstein-2 distance [Y]" — two cuts of the same catalog, each with a different anchor.
- (i) is REJECTED at scoping because it discards the load-bearing N_HI dimension that Tillman+2023's actual physical framing names; using (i) alone would be the [D-26]-analogue of discarding the very anchor that justifies the track.

**What would change the recommendation.**
- If Q-C1 returns "Voigt-fitting library landscape is heavy / b-parameter retrievability is signal-limited," demote (iii) → b-N joint becomes too noisy to support; keep (ii) CDDF as primary with (i) b-distribution-as-1D-summary as the paired diagnostic (preserves at least one b-summary in the deliverable).
- If Q-C3 returns no CDDF effect-size anchor at z near 0.3 (i.e., Tillman+2023's CDDF numerics are at z = 0.1 only and not extrapolatable), the rule-14-fragility on (ii) tightens; rescue path is rule-14(iii) deliverable demotion to adjacent-finding only.

**Owner.** PI + defense-panel. Decision deferred until Q-C1 and Q-C3 return.

### Q-C3 — External effect-size anchor (support-researcher with web tools; rule 14 critical)

**Question.** Does Tillman+2023 (arXiv:2210.02467) and adjacent low-z AGN-feedback CDDF / line-broadening papers (Sorini-Davé-Anglés-Alcázar 2020 SIMBA; Bolton+2017 Sherwood; Hummels+2020 SMUGGLE; Burkhart+2022; Tillman+2024 arXiv:2410.05383) report **specific effect sizes** — a numeric Δ on the CDDF normalisation, a numeric Δ on the b-distribution median, a numeric Δ on the broad-line fraction f(b > 40 km/s) — that we can pre-commit as the bar?

The pk-feedback-classifier [D-14] Sub-deliverable C cited Tillman+2023 as the line-broadening + CDDF mechanism citation but did NOT extract specific effect-size numerics (the focus there was on P_F(k) magnitudes from Khaire+2024 / Tillman+2024). Q-C3 is the fresh anchor pull for the linewidth-cddf observable family.

**Two candidate anchor constructions.**

- **Anchor A — Tillman+2023 + adjacent CDDF/line-broadening effect sizes.** "AGN-vs-no-AGN at z = 0.1 shifts the CDDF normalisation by X% at log N_HI = [range], or shifts the b-distribution median by Y km/s, or extends the broad-line fraction f(b > 40 km/s) by Z." This is the directly-comparable anchor for the chosen (ii) + (iii) observables. **Risk:** the published numerics may be on a different sub-statistic (e.g., line-density integrated over a column-range, not the differential dN/dN_HI), requiring forward-translation through our setup → rule-14-fragile if so. Best case: rule-14(i)-clean.
- **Anchor B — Sinigaglia+2026 / CAMELS-LH SBI prior-recovered null translated to the new observable family.** Their NULL is on P_F(k) → continuous-θ SBI; transferring it to a linewidth-cddf observable would require an additional argument that the same cosmic-variance-dominated regime applies to line statistics on the same simulation suite, which is plausible (same boxes, same total volume) but not directly published. Best case: rule-14(i)-clean direction-of-null bar (analogous to fps-sbi Anchor B); not a sharp effect-size threshold.

**What we need before kickoff.**
1. `support-researcher` literature dispatch (WebSearch + WebFetch permitted, per pk-feedback-classifier [D-14] precedent + Q_S3_LIT_ANCHOR.md precedent that the support-researcher's effective dispatch shape now includes web tools): extract from Tillman+2023, Tillman+2024, Sorini-Davé-Anglés-Alcázar 2020, Burkhart+2022, Hummels+2020 the **specific numeric** for: (a) CDDF-normalisation Δ between AGN and no-AGN runs at z near 0.3 (or z = 0.1 extrapolatable); (b) b-distribution median / IQR / broad-tail Δ between feedback recipes; (c) any joint (b, N_HI) effect-size statistic if reported.
2. If a specific effect-size number is found: that becomes the pre-committed bar. PASS = our effect size within 2× of published / NOT-PASS = our effect size > 2× wider (consistent with linewidth-cddf at z ≈ 0.3 on Sherwood being noisier than the literature's z = 0.1 SIMBA-class result).
3. If NO specific effect-size number is found: the bar reverts to a rule-14(ii) pre-committed-process-failure construction (we pre-commit a specific effect-size threshold below which the deliverable is published and above which it is not), AND the deliverable is demoted at scoping to rule-14(iii) adjacent-finding only — same surface as `pk-feedback-classifier` [D-17].
4. Verify the **z-extrapolation discipline**: Tillman+2023 is z = 0.1, our target is z ≈ 0.3. Same hedging discipline as pk-feedback-classifier [D-15] applies — both hedges (z = 0.1 → 0.3 + Sherwood-not-SIMBA suite mismatch) must compose additively on the face of every claim.

**PI recommendation (preliminary).**

> **Recommend Anchor A (Tillman+2023 + adjacent CDDF/line-broadening effect sizes) as primary external anchor.** Anchor B held as supporting null-direction cross-check only (per fps-sbi Anchor B precedent).

Rationale: Anchor A is structurally rule-14(i)-clean IF a specific Tillman+2023-class numeric exists. The most likely failure mode is Anchor A returns "Tillman+2023 reports CDDF redistribution qualitatively, not as a sharp numeric Δ" — in which case the rule-14(iii) demotion at scoping is mandatory and the deliverable surface is locked to adjacent-finding only before any compute is committed.

**Owner.** `support-researcher` with WebSearch + WebFetch (per the fps-sbi Q_S3 dispatch precedent; same web-tool authorization). BLOCKING per §4 gate sequence.

---

## §3 — Branch + LEDGER structure proposal

### Branch
- **Name:** `exp/linewidth-cddf`.
- **Off:** `main` (NOT off `exp/pk-feedback-classifier` or `exp/fps-sbi`; this is a new observable-family track per the CLAUDE.md experiment-isolation rule).
- **Author:** user, on user acceptance of this SCOPING.md + defense-panel APPROVE on the eventual Stage 3 spec. PI does NOT create the branch in this dispatch.

### LEDGER skeleton (proposed; NOT authored here)

Title: `LEDGER — linewidth-cddf`

§1 Pulse row outline (Stage 3 = first stage of this track):

| Stage | Focus Area | Status | Pass Condition | Outputs |
|:---|:---|:---|:---|:---|
| **Stage 3 — Line catalog + CDDF + joint (b, N_HI) on Sherwood z = 0.3 4-recipe suite** | Detect lines from τ, fit Voigt profiles to recover (b, N_HI) per line, compile per-class CDDF and per-class joint (b, N_HI) distributions; compare across recipes with a pre-committed effect-size statistic. | 🟡 PROPOSED — pending Q-C1, Q-C3, defense-panel APPROVE | (deferred to spec — depends on Anchor A return) | (deferred to spec) |

**Stage numbering rationale.** Stage 0 = pk-feedback-classifier de-risking probe; Stage 1 = pk-feedback-classifier stacked sweep; Stage 2 = fps-sbi (scoped, structurally weakened on Q-S1/Q-S3, did not open LEDGER); **Stage 3 = this track**. Keeping cumulative numbering across tracks makes the cascade reading-order obvious in the project narrative.

§3 local D-XX numbering starts at **[D-01]**; [D-01] is the local re-statement of "this track inherits pk-feedback-classifier [D-26] + fps-sbi SCOPING §5 structural-weakening close; cascade-inheritance verb-ceiling per §1 of this SCOPING.md."

### Existing artifact re-use

**Essentially none.** The Stage 1 P_F(k) feature arrays (referenced in fps-sbi SCOPING §3 as candidate SBI training data) are wrong-feature-family for linewidth-cddf: they reduce τ → F → P_F(k), discarding the line-by-line peak structure. The linewidth-cddf pipeline reads `tau.npy` directly and builds a per-line catalog, which is a different reduction.

What IS re-usable:
- The `DataIngestor` in `src/core/data.py` (load `tau.npy` per class via `filename="tau.npy"`).
- The `vel.npy`-derived Δv = 2.6365 km/s grid (pk-feedback-classifier [D-04]) for the Doppler-velocity axis.
- The Juno HPC submission skill + `.env` JUNO_* block (pk-feedback-classifier [D-05]) for compute target, IF Q-C1 estimates a budget that exceeds local laptop.
- The 64-cell PASS/FAIL routing-table discipline (pk-feedback-classifier [D-20] / [D-24]) as a structural precedent for the eventual Stage 3 spec gate-set adjudication — NOT the cells themselves.

### Compute-envelope sanity (preliminary, pre-Q-C1)

Voigt-fitting per-sightline cost scales as O(n_lines × n_iter) with n_lines ~ 10–50 per sightline at z = 0.3 and n_iter ~ 20–100 for nonlinear least-squares. Estimated per-sightline cost: 0.1–1 s on a single CPU core. Total budget: 4 classes × 16384 sightlines = 65,536 sightlines × ~0.5 s ≈ 9 hours on a single core; ~30–60 minutes on a 16-core Juno node. Comfortably inside the pk-feedback-classifier [D-25] 24h Juno ceiling. Q-C1 owner returns refined per-sightline timing on the empirical 10-sightline pilot.

---

## §4 — Pre-kickoff gate sequence

Ordered gates before any compute commits. Each gate is BLOCKING on the prior.

1. **User acceptance of this SCOPING.md.** No downstream dispatch without explicit user GO. PI does not assume acceptance.

2. **`data-engineer` dispatch on Q-C1 (τ-to-line-stats pipeline feasibility).** BLOCKING. Self-contained brief: "Load `data/preprocessed/Sherwood_z0.3_inf/{1, 4}/tau.npy`; for 10 random sightlines per class, run a peak-find at τ_min ∈ {0.05, 0.1, 0.3} and report per-sightline line counts; compare against published z ≈ 0.3 line-density expectation (~10–50 lines/sightline above δ_F = 0.05 per Davé+1999 / Williger+2010). Survey Voigt-fitting library landscape (`astropy.modeling.Voigt1D` vs custom `scipy.special.wofz` vs external `linetools` / `VPFIT`) and return preferred + fallback with engineering-cost estimate. On synthetic Voigt profiles at known (b, N_HI), test b-recovery accuracy and noise-floor. Return: 'feasible' OR 'feasible-in-principle-signal-limited' OR 'infeasible' with empirical evidence." Web tools permitted for library landscape (PyPI / GitHub / docs).

3. **`support-researcher` dispatch on Q-C3 (external effect-size anchor).** BLOCKING. **Parallel with #2** — same session. Self-contained brief: "From Tillman+2023 (arXiv:2210.02467), Tillman+2024 (arXiv:2410.05383), Sorini-Davé-Anglés-Alcázar 2020, Burkhart+2022 (arXiv:2204.09712), Hummels+2020, extract the most specific numeric effect-size on: (a) CDDF normalisation Δ between AGN-on / AGN-off feedback recipes at z near 0.3 (or z = 0.1 extrapolatable); (b) b-distribution median / IQR / broad-tail Δ between feedback recipes at z near 0.3; (c) any joint (b, N_HI) effect-size statistic. Return: numeric Δ + citation; OR 'NULL — no specific effect-size, only qualitative redistribution direction' (rule-14(iii) rescue path mandatory at scoping). Web tools required. The Q_S3_LIT_ANCHOR.md citation-error-correction discipline applies (Walther / Maitra / Wadekar precedents) — verify author names + arXiv IDs against authoritative sources before citing."

4. **PI authors Stage 3 spec ([D-02]+ in linewidth-cddf LEDGER).** Uses Q-C1 return (feasibility → BUILD vs PIVOT vs SHORT-CYCLE-CLOSE fork), Q-C3 return (anchor numeric or rule-14 rescue path), Q-C2 PI recommendation (CDDF primary + joint (b, N_HI) diagnostic). Spec must include: (a) gate set with falsification paths per rule 5 — pre-committed C2/C3 sub-gate analogous to pk-feedback-classifier [D-26] G4 mandatory; (b) cascade-inheritance verb-ceiling per §1 of this SCOPING; (c) anti-degeneracy audit per rule 3 explicitly addressing ⟨F⟩-sensitive-detection-threshold confound; (d) compute envelope per "Economic compute" discipline with refined Q-C1 timing; (e) deliverable-surface verb whitelist + binding non-claim list per `pk-feedback-classifier` [D-17] precedent; (f) rule-14 rescue path explicit (Anchor A direct numeric OR rule-14(ii)/(iii) per Q-C3 return).

5. **defense-panel adversarial review of the spec.** BLOCKING. The pk-feedback-classifier [D-13] panel review precedent applies — adversarial role-play against rules 2/3/4/5/6/7/8/14/15. Panel APPROVE WITH CAVEATS path expected (per [D-13] precedent); PI amendment cycle until APPROVE clean. Particular attention to rule-8 cascade-close-formality (a third specific intervention on the falsification queue, not a typology-coverage claim) and to the [D-37] honest framing of the cascade-inherit verb ceiling.

6. **On panel APPROVE: `core-implementer` + `infrastructure-manager` dispatch.** Implementer authors the line-detection + Voigt-fitter + CDDF compiler script + spec doc; infrastructure-manager scopes compute target (local vs Juno) and tracker. Smoke-timing run before full sweep per pk-feedback-classifier [D-25] precedent. **NOT before this gate.**

**`paper-author` NOT dispatched at any point in this gate sequence.** User-triggered-only per CLAUDE.md / PI binding rule. The freshly-sharpened paper-trigger rule (PI agent spec, Coordination with `paper-author` block) binds further: paper-author is user-triggered-only IFF solid/successful results, and a not-yet-opened track cannot qualify under any framing.

**No compute is committed by this SCOPING.md.** Acceptance of this document is an observable-family scoping authorization only. Gates 2–6 each have their own authorization step downstream.

---

## §5 — Honest assessment per [D-37] (most likely failure mode)

Two candidate failure modes surfaced honestly. The user should accept this SCOPING only if BOTH are acceptable outcome shapes.

### Failure mode (a) — Q-C1 returns signal-limited (short-cycle close)

Sherwood `tau.npy` may not have sufficient signal-to-noise / resolution at the 2.6365 km/s pixel scale + Sherwood's small mean-preserving LSF/noise to support clean line detection at the line densities expected at z ≈ 0.3. Concretely: lines may be heavily blended at the τ values actually present (z ≈ 0.3 has lower line density than the canonical z = 2–4 forest, BUT the Sherwood pixel grid is fixed at 2.6365 km/s which is wider than the b ≈ 15 km/s thermal-Doppler core → individual narrow lines are barely resolved). The Voigt-fitting recovery of (b, N_HI) may be biased > 10% or have a noise floor > 5 km/s in b on synthetic data with the matching pixel + noise model.

In that case the track is **short-cycle close**: Q-C1 returns "feasible in principle, signal-limited in practice," and the track closes before any spec, before any branch creation, with the SCOPING.md and Q-C1 report as the entire deliverable surface. No paper. No compute commit. The cascade picks up one more line in the falsified-prior ledger ("linewidth-cddf as an observable-family pivot was signal-limited on Sherwood z = 0.3 at the pixel scale provided") and the next re-open path is what fps-sbi SCOPING §5 already cataloged as path (b): a fundamentally different observable family (2D flux maps, transverse correlations, metal lines, mock spectroscopic survey).

This is a **valid decision-quality end state per rule 7.** The discipline (pilot-before-spec, feasibility-before-build) is exactly what rule 3 anti-degeneracy auditing was designed to produce.

### Failure mode (b) — Lines are detectable but b-distribution and CDDF show the [D-26] cascade signature

Q-C1 returns "feasible," the pipeline builds clean, the full sweep runs, and the verdict is: **strong C4-vs-others lift (KS / Wasserstein significant; CDDF normalisation Δ at log N_HI > 14 visible) + C2/C3 binary null at the pre-committed sub-gate (≈ binary chance) + the ⟨F⟩-detection-threshold confound is named on the face of the result.**

This is precisely the [D-26] pattern translated to the new observable family: the cascade-inherit reading of §1 PREDICTED this outcome on physical grounds. It delivers a **second independent observable's worth of the same finding** — the C2/C3 distinction (stellar-wind vs wind+AGN) is at-or-near-null at z ≈ 0.3 on Sherwood across two distinct observable families (stacked P_F(k) AND linewidth-cddf), which has scientific value as a rule-8-class-cascade-foreclosure-formality-shaped finding (two specific interventions, two distinct observable families, two congruent C2/C3-null signatures → the cascade-close claim on "C2/C3 distinguishability from the Lyα forest at z ≈ 0.3 on Sherwood" becomes formally stronger). The deliverable is **a quantified second null with rule-14(iii) adjacent-finding framing**, exactly the same shape as [D-17].

This is also a **valid decision-quality end state per rule 7.** The spec hedged per rule 2 cascade-inherit, the C2/C3 sub-gate per rule 5 was pre-committed to falsify the "linewidth-cddf escapes the cascade" hypothesis if false and it did so cleanly, the ⟨F⟩-detection-threshold confound per rule 3 was named in advance and surfaced honestly per [D-37].

### The non-failure outcome (the small positive-result shape that DOES exist)

It is also possible — and the §1 cascade-inherit verb ceiling preserves space for this — that the line-by-line statistics DO produce a C2/C3 lift that P_F(k) did not. Information-theoretically this would mean the RF on stacked P_F(k) at M ≤ 256 was a *suboptimal* classifier on the C2/C3 axis, and that the line-by-line feature engineering (b-tail, CDDF high-column tail, b-N envelope upper boundary) encodes the C2/C3 distinction directly in a way the Fourier reduction smeared out. Tillman+2023 anchors the direction; the gain is plausible at ~moderate-low confidence per the cascade. The verb-ceiling allows "first multi-class classifier test of the Tillman+2023 framing" / "consistent with Tillman+2023 direction" — never "we recover feedback classes," even on a positive result.

### What the user is accepting by accepting this SCOPING

By accepting this SCOPING, the user authorizes the Q-C1 + Q-C3 dispatch sequence under the explicit understanding that:
- The most-likely outcome is failure mode (a) or (b) — both null-shaped end states.
- The headline-positive outcome, if it lands, would be verbed as "first multi-class classifier test of the Tillman+2023 line-broadening / CDDF-redistribution framing" — never as a feedback-classification claim.
- No paper-author dispatch under any conditions until the user separately triggers it AND the result qualifies under the freshly-sharpened paper-trigger rule (solid/successful results gate).
- The next re-open path beyond this track, if both (a) and (b) realize, is `fps-sbi` SCOPING §5 path (b)'s "fundamentally different observable family" — at which point the linewidth-cddf cascade line becomes a third entry in the falsified-prior ledger and the rule-8-class formal-foreclosure claim on "Lyα forest at z ≈ 0.3 on Sherwood as a feedback discriminator" gains a third specific intervention's worth of strength.

If those outcome shapes are NOT what the user wants from a re-open, the more honest path is paper-author trigger on pk-feedback-classifier [D-26] as-is (per fps-sbi SCOPING §5 path (a)), or a wholesale-different-suite scoping (e.g., CAMELS, IllustrisTNG, SIMBA — outside the Sherwood-feedback-as-observable axis entirely).
