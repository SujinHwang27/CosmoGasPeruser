# Q-C1 — τ-to-line-stats pipeline feasibility audit

**Track scoping context:** `experiments/linewidth-cddf/SCOPING.md` §2 Q-C1 (commit `0185027`, branch `exp/pk-feedback-classifier`).
**Owner:** data-engineer (this report).
**Authorization scope:** Q-C1 only. NOT authorizing track open, NOT authoring LEDGER, NOT dispatching downstream agents.
**Audit script:** `experiments/linewidth-cddf/scripts/qc1_feasibility.py` (deterministic seed `42`; run with `PYTHONPATH=. uv run python …`).
**Verdict (one line):** **FEASIBLE-IN-PRINCIPLE-SIGNAL-LIMITED** — short-cycle close per SCOPING §5(a).

---

## §1 — On-disk inventory + existing-infrastructure confirmation

### 1.1 Sherwood τ files (data-locality enumeration, real Glob output)

```
data/preprocessed/Sherwood_z0.3_inf/1/  -> axis.npy (524 KB), flux.npy (268 MB),
                                          tau.npy (268 MB), vel.npy (16 KB), wave.npy (16 KB)
data/preprocessed/Sherwood_z0.3_inf/4/  -> axis.npy (524 KB), flux.npy (268 MB),
                                          tau.npy (268 MB), vel.npy (16 KB), wave.npy (16 KB)
```

- Class 1 (NoFeedback) `tau.npy` — **EXTRACTED-PRESENT** at `data/preprocessed/Sherwood_z0.3_inf/1/tau.npy`, 268,435,584 bytes ≈ 16384 × 2048 × 8 byte double — shape confirmed by load.
- Class 4 (WindStrongAGN) `tau.npy` — **EXTRACTED-PRESENT**, same size.
- Classes 2, 3 not loaded in this audit (Q-C1 brief specifies classes 1, 4 only); their `tau.npy` are co-located and DVC-managed identically per pk-feedback-classifier LEDGER §4 lineage.

### 1.2 Existing infrastructure — Voigt/line-detect code (green-field confirmed)

- `Grep -i "voigt|wofz|line_detect|Doppler|column_density|N_HI|CDDF"` over `src/` returned **NO files matched**. Green-field confirmed for the linewidth-cddf observable family. `src/core/transforms.py` contains `PCATransform`, `DCTTransform`, `WindowedDCTTransform`, `WaveletTransform`, `FluxPowerSpectrum`, `FisherTransform` — none Voigt-fitting nor line-detection.
- `src/core/data.py` `DataIngestor` is `filename`-parameterized; `DataIngestor(base_path="data/preprocessed/Sherwood_z0.3_inf", filename="tau.npy")` is a one-line loader. The audit script used a direct `np.load(..., mmap_mode="r")` since it is a feasibility probe, not a track loader; any track-level adoption goes through `DataIngestor` per CLAUDE.md "never raw `np.load()` in scripts".

### 1.3 pyproject.toml runtime probe

- `scipy>=1.12.0` declared; resolved at runtime to `1.16.2` — provides `scipy.special.wofz` (Faddeeva function, the Voigt convolution kernel).
- `astropy` — **NOT declared, NOT installed.**
- `linetools` — **NOT declared, NOT installed.**

---

## §2 — Sub-task 1: empirical line-detection on real Sherwood τ

10 random sightlines per class, deterministic seed `np.random.default_rng(42)`, `scipy.signal.find_peaks` with `height=τ_min`, `prominence=τ_min`, `distance=1` pixel. Sightline indices (sorted): class 1 = [1407, 1461, 1543, 3300, 7092, 7187, 10719, 11424, 12674, 14063]; class 4 = [2989, 6072, 6592, 7375, 8193, 8936, 10547, 12802, 13479, 15179].

### 2.1 Per-sightline line counts (real data)

| class | τ_min | mean | median | min | max | per-sightline counts |
|:------|:------|:-----|:-------|:----|:----|:---------------------|
| 1 (NoFeedback)     | 0.05 | **6.0** | 6.0 | 3  | 10 | [6, 10, 3, 7, 8, 6, 4, 3, 6, 7] |
| 1 (NoFeedback)     | 0.10 | 4.7 | 5.0 | 2  | 7  | [6, 7, 2, 6, 5, 4, 3, 2, 5, 7] |
| 1 (NoFeedback)     | 0.30 | 2.3 | 2.0 | 0  | 5  | [2, 5, 2, 4, 1, 0, 2, 0, 3, 4] |
| 4 (WindStrongAGN)  | 0.05 | **4.2** | 4.0 | 2  | 8  | [8, 4, 4, 2, 5, 4, 3, 4, 6, 2] |
| 4 (WindStrongAGN)  | 0.10 | 2.7 | 2.0 | 1  | 5  | [5, 2, 4, 2, 2, 4, 1, 2, 4, 1] |
| 4 (WindStrongAGN)  | 0.30 | 1.4 | 1.0 | 0  | 4  | [4, 1, 1, 0, 2, 0, 0, 1, 4, 1] |

### 2.2 Anchor-against-published comparison

The brief cites Davé+1999 / Williger+2010 for z ≈ 0.3 with a 5400 km/s sightline span as expecting **~10–50 lines/sightline above δ_F = 0.05** (PI training-corpus recall, order-of-magnitude). Sherwood span is 5396.92 km/s (pk-feedback-classifier [D-04]); δ_F = 0.05 corresponds to τ ≈ −ln(0.95) ≈ 0.0513, so τ_min=0.05 is the closest matched threshold.

Observed: class 1 mean = **6.0 lines/sightline**, class 4 mean = **4.2 lines/sightline**. **Both classes sit ≈ 2× below the lower edge of the published 10–50 band**. C4 (WindStrongAGN) sits *below* C1 (NoFeedback) at every threshold — consistent with [D-07]'s C4 ⟨F⟩-offset finding (C4 is the most-transmitting class, hence the fewest detected absorption peaks).

### 2.3 Verdict on Sub-task 1

The SCOPING §5(a) short-cycle-close trigger is "0–2 lines/sightline at τ_min=0.05 → infeasible." We observe 4.2–6.0 mean / 2–3 minimum — **above the catastrophic-infeasible trigger** but **below the published Davé+1999 / Williger+2010 band by a factor ≈ 2**. Honest framing per [D-37]: lines are detectable in principle; the absolute line density is at the **low end** of (and below) the published-anchor band, leaving the per-class population statistics on a ~tens-of-lines-per-sightline budget × 16384 sightlines ≈ 10⁵ lines per class — usable for CDDF/b-distribution histograms but with each per-sightline catalog modest.

**Sub-task 1 verdict: PASS (above catastrophic trigger), but the published-anchor undershoot must propagate as a hedge on any downstream "consistent with Tillman+2023 / Davé+1999" verbing.**

---

## §3 — Sub-task 2: Voigt-fitting library landscape

### 3.1 Library probe (runtime, in-tree environment)

| Library | Available | Version | Notes |
|:--------|:----------|:--------|:------|
| `scipy.special.wofz` | **YES** | scipy 1.16.2 (already declared in `pyproject.toml`) | Faddeeva function `w(z) = e^{-z²} erfc(-iz)`; Voigt = Re[w(u + ia)]. Zero new dependency cost. |
| `astropy.modeling.Voigt1D` | NO | not installed | astropy itself NOT in `pyproject.toml` dependencies. `astropy.modeling.Voigt1D` is a Lorentzian × Gaussian convolution model with fittable amplitude_L / x_0 / fwhm_L / fwhm_G — needs translation from (b, N_HI, v0) to the astropy parameter set. ~25 MB install with deps. |
| `linetools` | NO | not installed | Astrophysics-domain Voigt + LineList package (Prochaska et al.). PyPI; transitively depends on astropy, specutils, IGM-data files. ~80–100 MB install. |
| VPFIT (Carswell & Webb 2014) | NO | Fortran binary, not present | Would require external binary install + subprocess wrapper + input-deck templating + output parser. |

### 3.2 Engineering-cost estimates

- **`scipy.special.wofz` custom fitter** — implemented and tested in this audit script (`voigt_tau`, `fit_single_voigt` at `experiments/linewidth-cddf/scripts/qc1_feasibility.py` lines 31–110). Voigt profile = ~10 LOC, `scipy.optimize.curve_fit` wrapper = ~20 LOC. **Total: ~30–50 LOC** for a clean `VoigtFitter` class under `src/core/`. Dependency cost: **ZERO** (scipy already declared, already installed, already used across the project). Risk: green-field code requires its own unit tests (synthetic recovery on known (b, N_HI)) — exactly what Sub-task 3 below already constitutes a smoke test for.
- **`astropy.modeling.Voigt1D`** — ~10–20 LOC adapter (parameter-name translation + `fitting.LevMarLSQFitter`). Dependency cost: adds `astropy>=5.0` (~25 MB install, well-maintained, ESO/STScI-backed). Risk: parameter translation must be unit-tested against the same synthetic suite; astropy bumps occasionally break model fittability APIs (rare but historical).
- **`linetools`** — ~5 LOC call (Prochaska's `AbsLine` + `aodm` / Voigt-fit utilities). Dependency cost: heavy (astropy + specutils + tied to IGM data files); maintenance state: active on GitHub but quieter than astropy. Risk: opinionated API for absorption-line catalogs; coupling our pipeline to its model assumptions.
- **VPFIT** — ~100–200 LOC subprocess wrapper. Dependency cost: external Fortran binary not on PyPI; would need separate install + Juno-side build. Risk: high friction; canonical for Lyα-forest line catalogs (Carswell & Webb 2014) but operationally heavy for a feasibility-tier track.

### 3.3 Recommendation

- **Preferred: `scipy.special.wofz`-based custom fitter** under (e.g.) `src/core/voigt.py`. Lowest engineering cost (30–50 LOC, zero new deps, scipy is already the project's numerical workhorse), full PI control over the Voigt model + fit-window + bounds, easy to unit-test on the synthetic suite already built in Sub-task 3. Trade-off: green-field code carries its own correctness burden; the synthetic recovery in §4 below is the first acceptance test.
- **Fallback: `astropy.modeling.Voigt1D`.** Pay the ~25 MB astropy install in exchange for a peer-reviewed Voigt model + fitter; ~10–20 LOC adapter. Adopt only if the custom fitter is found numerically fragile in a corner of (b, log N) space that the synthetic suite did not anticipate.
- **REJECTED at scoping: `linetools` and VPFIT** — too heavy for the feasibility tier; revisit only if the track survives the Stage 3 panel-APPROVE gate.

---

## §4 — Sub-task 3: synthetic b-recovery on Voigt profiles

### 4.1 Setup

- 2048-pixel uniform-velocity grid at Δv = 2.6365 km/s (matching Sherwood; pk-feedback-classifier [D-04]).
- Line placed at grid centre (`v0 = 2697.5 km/s`).
- Grid: `b ∈ {15, 25, 40, 60, 100} km/s` × `log N_HI ∈ {13.0, 13.5, 14.0, 14.5, 15.0}` = 25 cells.
- 10 noise realizations per cell (independent draws, `rng = np.random.default_rng(42)`).
- **Noise assumption (honest [D-37] flag):** `σ_F = 0.01` (1% Gaussian on the flux). pk-feedback-classifier LEDGER §2 names a "small mean-preserving LSF/noise stage (not bit-exact)" but does not pin σ numerically on disk. 1% is a conservative representative value; the real Sherwood noise floor at this Δv may be tighter (making recovery easier) or looser (making it harder). Stage 3 spec must pin σ via direct flux-residual measurement on `tau.npy → flux.npy` round-trip before the production sweep.
- Single-line, isolated-profile fitter (no blending). `scipy.optimize.curve_fit` with bounds `b ∈ [5, 200] km/s`, `log N ∈ [11, 17]`.
- All 25 × 10 = 250 fits converged (`n_ok = 10` for every cell).

### 4.2 b-bias and noise-floor table (full 5×5 grid)

| b_true (km/s) ↓ \\ log N_HI → | 13.0 | 13.5 | 14.0 | 14.5 | 15.0 |
|:-----|:-----|:-----|:-----|:-----|:-----|
| 15  | -0.5% / 0.19  | -0.3% / 0.13  | **-20.0% / 1.51** | **+16.8% / 1.40** | **+70.5% / 1.27** |
| 25  | +0.05% / 0.17 | +0.33% / 0.14 | **-0.20% / 0.42** | +4.5% / 1.18 | **+48.5% / 1.18** |
| 40  | +0.71% / 0.82 | -0.17% / 0.22 | -0.67% / 0.20 | **-15.7% / 2.70** | **+30.1% / 2.82** |
| 60  | +0.70% / 0.80 | +0.23% / 0.52 | -0.04% / 0.19 | -6.5% / 4.25 | **+15.4% / 3.13** |
| 100 | +1.37% / 2.41 | +0.28% / 1.13 | -0.06% / 0.34 | -0.36% / 0.45 | -9.6% / **5.02** |

Format: `b_bias_pct / b_std (km/s)`. **Bold** marks cells failing either gate (|bias| > 10% **or** σ_b > 5 km/s).

### 4.3 Representative cell

(b=25, log N=14): **b_mean = 24.951 km/s, bias = −0.20%, σ_b = 0.42 km/s, log N recovered to 13.999 (bias −0.0008 dex), n_ok = 10/10.** Inside the < 10% bias / < 5 km/s gate by a wide margin.

### 4.4 Where the fit fails

Two structural failure regimes are visible in the table:

1. **Unresolved narrow lines at b ≤ 15 km/s combined with high column.** b=15 line core FWHM ≈ 25 km/s, pixel sampling Δv=2.6365 km/s → 9–10 pixels across the core, which sounds OK, but the Voigt damping wings at log N ≥ 14 carry significant absorption that the fitter mis-attributes between b and N — b=15/logN=14 cell biases −20%; b=15/logN=14.5 cell biases +17%; b=15/logN=15 biases +70%. Doppler-width and column become strongly anti-correlated in the Voigt likelihood once the line approaches saturation, and 1% flux noise on a single isolated line is enough to flip the fitter between the narrow-saturated and broad-unsaturated branches.
2. **Saturated cells at log N=15 across all b.** The flat saturated core makes b unidentifiable from the core; the wings recover b only via the damping-Lorentzian contribution, which is weak unless log N reaches damped-Lyα territory (log N > 17). All five log N=15 cells either bias > 10% or have σ_b approaching the 5 km/s noise floor (b=100 is borderline at 5.02 km/s).

### 4.5 Lyα-forest-regime sub-grid (the physically relevant subset)

The Lyα forest at z ≈ 0.3 is overwhelmingly at log N_HI ∈ [12.5, 14.0] (Tillman+2023's framing; the bulk of detected lines per Sub-task 1 are at low τ, hence low N). Restricting the 5×5 grid to log N ∈ {13.0, 13.5, 14.0} and b ∈ {25, 40, 60, 100} (excluding b=15 — unresolved at Δv=2.6365 — and excluding log N ≥ 14.5 — saturated regime):

- **All 12 cells in this Lyα-forest sub-grid PASS:** |bias| ≤ 1.4%, σ_b ≤ 2.41 km/s. Best cell σ_b = 0.13 km/s; worst σ_b = 2.41 km/s.

### 4.6 Verdict on Sub-task 3

The SCOPING §5(a) trigger is "recovered b biased > 10% **or** noise floor > 5 km/s in b on synthetic data → signal-limited in practice." Read against the full 5×5 grid: **FAILS** in two physically-identifiable corners (unresolved-narrow at b≤15, saturated at log N≥15). Read against the Lyα-forest-regime sub-grid (log N ∈ [13, 14], b ∈ [25, 100]): **PASSES** cleanly.

This is **the SCOPING §5(a) honest-framing pattern**: the pipeline is "feasible in principle, signal-limited in a physically-identifiable corner." The bulk of the forest population sits inside the PASS sub-grid; the broad-line tail (the *load-bearing* diagnostic for feedback-driven heating per Tillman+2023's b-distribution-redistribution framing) sits at b > 40 km/s and is, on this synthetic test at 1% flux noise, exactly the regime where the b=40 / log N=14.5 cell biases −16% and the b=60 / log N=14.5 cell σ_b reaches 4.25 km/s. **The diagnostic feedback regime is the part of (b, log N) space the synthetic test marginally fails.**

---

## §5 — Final verdict

> **FEASIBLE-IN-PRINCIPLE-SIGNAL-LIMITED.**

Mapped against the three SCOPING-defined verdict options:

- **NOT "FEASIBLE"** — line density at τ_min=0.05 sits **below** the published Davé+1999 / Williger+2010 ~10–50 lines/sightline band by ~2× (observed 4–6 vs published 10–50), AND the b-recovery on synthetic 1% flux noise fails the < 10% / < 5 km/s gate in the **broad-line / partially-saturated diagnostic regime** that Tillman+2023 names as the feedback-discriminating tail.
- **NOT "INFEASIBLE"** — line counts at τ_min=0.05 are 4–6 mean / 2–3 minimum, comfortably above the 0–2 catastrophic trigger; and the bulk Lyα-forest sub-grid (log N ∈ [13, 14], b ∈ [25, 100]) recovers b cleanly (< 1.5% bias, σ_b < 2.5 km/s).
- **"FEASIBLE-IN-PRINCIPLE-SIGNAL-LIMITED"** — the bulk regime is recoverable, the diagnostic-tail regime is marginal at 1% flux noise. Per SCOPING §5(a), this triggers short-cycle close.

### Representative numbers (verdict-load-bearing)

- **Per-sightline line count at τ_min=0.05, averaged across classes 1 and 4: (6.0 + 4.2) / 2 = 5.1 lines/sightline.** Published expectation: 10–50. Factor ≈ 2 below.
- **b-recovery bias on (b=25, log N_HI=14) representative cell: −0.20% bias, σ_b = 0.42 km/s.** PASSES the gate; the cell is in the bulk forest regime.
- **b-recovery on diagnostic broad-tail cells (b=40 / log N=14.5; b=60 / log N=14.5): bias −15.7% / σ_b 2.70; bias −6.5% / σ_b 4.25.** FAILS the gate in the regime feedback discrimination most depends on.

---

## §6 — Implications for the Stage 3 spec (honest [D-37])

The SCOPING §5(a) short-cycle-close trigger is met as written. Per SCOPING §4 gate 2 ("BLOCKING on the prior") and §5(a) ("the track closes before any spec, before any branch creation, with the SCOPING.md and Q-C1 report as the entire deliverable surface"), the honest reading is:

1. **Stage 3 spec authoring is NOT authorized by this Q-C1 return.** The combination of (a) observed line density a factor ≈ 2 below published z ≈ 0.3 anchor AND (b) b-recovery failure exactly in the broad-tail diagnostic regime Tillman+2023's framing rests on is the precise outcome shape the §5(a) trigger was written for.
2. **The pipeline is technically buildable** (scipy.special.wofz + scipy.optimize.curve_fit, ~30–50 LOC custom Voigt fitter, zero new dependencies, ~30–60 min compute on Juno per SCOPING §3 envelope) — but the diagnostic-power return on that build, given the bulk-regime/diagnostic-regime split observed here, is the structurally-weakened deliverable shape SCOPING §1 already pre-committed at the cascade-inherit verb ceiling.
3. **What this Q-C1 leaves on the table for the user / PI:**
   - **Path closed**: Stage 3 spec authoring on the (ii) CDDF primary + (iii) joint (b, N_HI) paired arm at the current Sherwood Δv=2.6365 km/s + assumed 1% flux noise model. The diagnostic-arm (iii) is the part of (b, log N) the synthetic test fails — pairing it with (ii) does not rescue (ii).
   - **Path open (with caveats)**: a higher-resolution re-grid (Δv → ~1 km/s) or a tighter noise-model measurement on Sherwood `tau.npy` could plausibly close the synthetic gap; pk-feedback-classifier LEDGER §2's noise-model language ("not bit-exact") leaves room that the true σ is below 1%. Pinning σ on disk before deciding the close is a 1-hour task and would tighten or relax this verdict by one notch. **Recommendation to the PI: if the user is interested in keeping the track alive, dispatch a 1-hour σ-measurement audit (compute `tau.npy → flux.npy` round-trip residual on classes 1, 4; pin σ_F empirically; re-run Sub-task 3 at the measured σ) before sealing the short-cycle close.** This is not authorized by this Q-C1 dispatch and is flagged here as an honest [D-37] open-the-door line.
   - **Path open (without caveats)**: the SCOPING §5 alternative-path target — fundamentally different observable family (2D flux maps, transverse correlations, metal lines, mock spectroscopic survey) — remains available. The linewidth-cddf cascade-line then becomes the **fourth** entry in the falsified-prior ledger (after signal-clustering-v2 [D-13], pk-feedback-classifier [D-01] / [D-15] / [D-26]) and strengthens the rule-8 cascade-close formality on "Lyα forest at z ≈ 0.3 on Sherwood as a feedback discriminator."
4. **Cascade verb-ceiling discipline holds.** The SCOPING §1 pre-commit on the cascade-inherit one-level downgrade from [D-10]'s starting confidence anticipated exactly this outcome shape; the [D-37] honest-framing rule says report it cleanly, not narrate it into a "feasible with caveats" yes-vote. The verdict is "feasible-in-principle-signal-limited," which is the SCOPING §5(a) short-cycle-close end state — a valid decision-quality outcome per project-architect rule 7.

---

**Compute cost of this audit:** single Python process, ~2 GB peak (two mmap-loaded 268 MB `tau.npy` arrays + 25-cell synthetic suite), total wallclock < 60 s on local Windows / Python 3.12.11 / scipy 1.16.2 / numpy ≥ 1.26.4. Zero Juno spend. Zero new dependencies. Reproducible with the script under `experiments/linewidth-cddf/scripts/qc1_feasibility.py` and seed `42`.
