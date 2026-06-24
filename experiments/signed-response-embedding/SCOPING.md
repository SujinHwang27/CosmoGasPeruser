# SCOPING — signed-response-embedding (thin control-first probe)

Status: SCOPED (PI worth-ruling + design panel, 2026-06-24). Branch
`exp/signed-response-embedding` off `feature/signal-clustering-v2`. Caps: ONE
script · ONE figure · ONE D-XX · NO new data · NOT a paper trigger; parked
verdict unchanged either way.

> Origin: the direct successor to `feedback-response-field` [D-01], which closed
> NULL (productive). That probe established that the response MAGNITUDE
> `||R_c|| = ||F_c - F_1||` restates total absorption (Spearman 0.778, the [D-13]
> deep-line trap) — NULL on magnitude — but that the SIGN of the response is
> recipe-dependent and physical (C-PASS: frac absorption-REMOVED 0.157 StellarWind
> -> 0.290 WindAGN -> 0.711 StrongAGN; mean signed response flips -0.0039 winds-ADD
> to +0.0051 StrongAGN-REMOVES). The sign is the ONE degree of freedom every
> magnitude/content representation provably discards (you cannot recover sign from
> `||R||` or from `|F|`).
>
> Data is NOISELESS (S/N=inf), so `R_c = F_c - F_1` is the EXACT physical feedback
> response on a byte-identical skewer — no measurement-noise residual to fight.

## 0. The new question (NOT classification, NOT magnitude)

The C-PASS that motivates this probe was a **per-recipe global aggregate** (one
posfrac per recipe, pooled over all sightlines × pixels). A monotone *global*
ordering across 3 recipes is exactly what a **per-recipe constant** looks like.
This probe asks the one question that decides whether the sign is embeddable:

> **At fixed total absorption, within a single feedback recipe, do different
> sightlines respond in different directions** — i.e. is the signed/directional
> feedback response a **per-sightline axis** (varies sightline-to-sightline,
> orthogonal to absorption strength), or merely (i) a **per-recipe global
> constant** (a class label dressed as a sightline property) and/or (ii) total
> absorption restated under a sign?

Class is the *treatment index*, not the target.

## 1. The candidate per-sightline signed feature(s)

For each geometry `i` and recipe `c in {2,3,4}`, with `R_c[i] = F_c[i] - F_1[i]`
(positive => absorption REMOVED):

- **Primary — net signed response (threshold-free):**
  `s_c[i] = mean_x( R_c[i, x] )`, shape `(16384,)`. Sign-bearing; the per-sightline
  analogue of the global `mean_signed` that flipped C2->C4. Gating runs on `s_c`.
- **Secondary — responding-pixel sign fraction (reported only):**
  `p_c[i] = frac_{x : |R_c[i,x]| > 0.05}( R_c[i,x] > 0 )`, NaN-guarded (sightlines
  with < 5 responding pixels masked out and counted). Threshold-dependent ->
  reported, NOT gating.

Both computed for c in {2,3,4}. **Primary gating recipe is c=4 (StrongAGN)** —
strongest directional signal, the most favorable case. c=2/c=3 reported for the
cross-recipe re-anchor, not a second bite.

> `s_c` deliberately mixes sign and magnitude (a deep-absorption sightline gives
> large |s_c| trivially). Do NOT pre-residualize; let control E do the work, then
> report the residual axis (`SR_resid`). This is intentional and is exactly what E
> firewalls.

## 2. Pre-committed controls + rule-5 SYMMETRIC bars

All bars two-sided, pre-committed. Compute everything, read the verdict
mechanically. Firewall pair = **F (per-recipe-constant)** and **E (absorption)**;
both must clear for a PASS.

- **F — within-recipe spread (the per-recipe-constant firewall; THE decider).**
  - `spread_within = IQR( s_4[i] over i )` — robust within-recipe dispersion.
  - `sep_between = | mean_i s_4 - mean_i s_2 |` — the cross-recipe shift that drove
    the global C-PASS.
  - **Spread ratio `SR = spread_within / sep_between`.**
  - **PASS (sightline axis):** `SR >= 0.5` — within-recipe sightline-to-sightline
    variation is at least half the between-recipe shift.
  - **NULL (recipe constant):** `SR < 0.25` — every StrongAGN sightline responds
    ~the same direction; the signal is a class label.
  - **AMBIGUOUS `0.25 <= SR < 0.5`** — reported NULL-leaning (does not clear PASS);
    record, do not re-run.
  - Cross-check (reported, non-gating): frac of c=4 sightlines with `s_4[i] > 0`
    (net removal) — a genuine axis has a non-trivial minority on BOTH sides of zero
    (neither < 5% nor > 95% one-sided). ~100% one-sided reinforces "constant."

- **E — absorption orthogonality (the [D-13] / magnitude-restated firewall).**
  - `rho_E = Spearman( s_4[i], total_abs[i] )`, `total_abs[i] = mean_x(1 - F_1[i,x])`
    (identical to the magnitude probe's D, so directly comparable).
  - `SR_resid`: regress `s_4 ~ total_abs` (linear `np.polyfit`, deg 1), take
    residuals, `SR_resid = IQR(resid) / sep_between` — the spread that SURVIVES
    absorption control. This is the real PASS object.
  - **PASS:** `|rho_E| < 0.5` AND `SR_resid >= 0.4`.
  - **NULL:** `|rho_E| >= 0.7` OR `SR_resid < 0.2`.

- **C' — cross-recipe directional ordering (sanity re-anchor, reported, non-gating
  / KILL trigger).** Re-confirm at per-sightline resolution: `mean_i s_2 <
  mean_i s_3 < mean_i s_4` AND `mean_i s_2 < 0 < mean_i s_4`. Expected to reproduce
  the global C-PASS. If it does NOT, that is a feature/load bug, not a finding ->
  KILL and surface.

## 3. Verdict (pre-committed)

- **PASS = F-PASS AND E-PASS** (both firewalls clear). The signed feedback response
  is a **per-sightline directional axis**: sightlines within a recipe respond in
  genuinely different directions, orthogonal to total absorption. -> Worth scoping
  a future signed-response embedding track (separately, defense-panel-gated).
- **NULL** (valid end-state, report numbers-first):
  - **F-NULL** -> signed response is a **per-recipe constant** (class label, not
    sightline axis) — the most-likely null and a clean finding.
  - **E-NULL** -> signed response is **total absorption restated under a sign** (the
    [D-13] deep-line trap, one level deeper than the magnitude probe caught).
  - Either NULL closes the signed-response-embedding idea. Do not re-run hunting a
    pass; do not widen bars post-hoc.
- **KILL (hard stop):** C' fails (per-sightline recipe means contradict the prior
  global C-PASS) -> feature/load inconsistent with [D-01]; STOP, surface to PI, emit
  no verdict.

## 4. The firewall, explicit

| Degeneracy the signal could collapse to | Firewall | Kill condition |
|---|---|---|
| Per-recipe global constant (StrongAGN sightlines ~same direction => class label) | **F** (within-recipe spread ratio) | `SR < 0.25` |
| Total absorption restated under a sign (deep-line sightlines net-remove more) | **E** (absorption-residual spread) | `|rho_E| >= 0.7` OR `SR_resid < 0.2` |
| Inconsistent with prior C-PASS (feature/load bug) | **C'** | per-sightline means contradict pooled global => KILL |

F and E are the two named degeneracies; both must clear. E runs AFTER F establishes
variation, so "spread survives absorption control" (`SR_resid`) is the PASS object.

## 5. Honest risks (PI)

1. **Most-likely outcome is F-NULL.** The C-PASS was a 3-point global ordering;
   global orderings are the canonical signature of a per-recipe constant. A monotone
   class-mean trend says nothing about within-class spread.
2. **`s_c` is sign×magnitude entangled.** Control E (`SR_resid` specifically) is the
   only thing between "real directional axis" and "magnitude leaking back through the
   sign-weighted mean." Weak E => collapses to the magnitude null already in hand.
3. **Pixel-threshold sensitivity.** `p_c` depends on the 0.05 cut; gating runs on
   `s_c` (threshold-free). Do not tune the threshold to move a verdict.
4. **One-recipe gating** on c=4 is deliberate (strongest => most favorable). If even
   the strongest recipe is a per-recipe constant, c=2/c=3 won't rescue it.

## 6. Anti-degeneracy audit (rule-3)

The ~92% low-absorption bulk (cluster-environment-gradient [D-01]) has small `R_c`
everywhere, so `s_c ~ 0 +/- noise` and its SIGN is near-arbitrary there. Mitigation
baked into the bars: F uses IQR (robust to the near-zero mass), and the cross-check
requires a genuine two-sided minority — a PASS cannot be manufactured by a
symmetric noise cloud around zero (that inflates IQR but `SR_resid` would not
survive and the two-sided-minority check distinguishes structured spread from noise
spread). **The near-zero bulk leaves sign unconstrained; F+E must show the spread is
structured (absorption-residual, two-sided), not a zero-centered noise cloud.**

## 7. Prior-failure ledger (rule-4) + inherited verb (rule-2 cascade)

- **[D-13]**: single-sightline feedback separability is intrinsically weak —
  falsified "a better per-sightline feature beats ~0.45."
- **`feedback-response-field` [D-01]** (direct predecessor, same confidence):
  response MAGNITUDE is a new axis -> FALSIFIED (Spearman 0.778 = total absorption).
- **`cluster-environment-gradient` [D-01]**: content clusters = discrete strata ->
  FALSIFIED (92% one bulk).
- **`confusability-residual` [D-01]**: orthogonal confusability residual exists ->
  FALSIFIED.

**Inherited verb (cascade):** the immediately-preceding similar-confidence claim
(magnitude = new axis) was falsified at the same one-level-up position. So the
signed-response claim **cannot be presented at high confidence** and **cannot be
presented as "structurally immune."** Honest verb: this is the **first test of the
per-sightline-axis-hood of the directional signal** — the SIGN survives the
magnitude-restatement trap by construction (sign ⊥ ||·||), but whether it survives
the per-recipe-constant trap is untested and is the entire content of this probe.
**NULL-leaning prior.**

## 8. The binding ceiling (verbatim in figure caption)

> This probe characterizes the DIRECTION of feedback action per sightline (does
> feedback ADD or REMOVE absorption on this line of sight, and does that vary
> sightline-to-sightline). It does NOT beat the ~0.45 per-sightline 4-class
> classification ceiling ([D-13]) and makes NO claim about it. A PASS authorizes a
> future embedding/interpretation axis, not a classifier. A NULL closes the
> signed-response-embedding line (sign, like magnitude, is not a per-sightline
> axis). The parked verdict is unchanged either way.

## 9. Caps (binding)

- ONE script: `scripts/run_signed_response.py` -> ONE module
  `src/core/signed_response.py` (reuses `SignalClusteringData.load_flux_per_class()`
  exactly as `feedback_response.py`; no new loader).
- ONE figure: `results/signed_response_embedding/figs/signed_response.png` (2×2:
  F within-recipe `s_4` dist + IQR; E `SR_resid` residual dist; E `s_4` vs
  total_abs scatter + Spearman; C' three per-recipe `s_c` dists). Caption carries
  verdict + SR/SR_resid/rho_E + the binding ceiling.
- ONE D-XX: `[D-01]` in §10 below (numbers-first verdict).
- NO new data. Reuses `data/preprocessed/Sherwood_z0.3_inf/{1..4}/flux.npy`
  (loaded/verified this session).
- Branch `exp/signed-response-embedding` off `feature/signal-clustering-v2`
  (in-place branch switch per exFAT policy — not a worktree).
- NOT a paper trigger. Outcome lands here + `.claude-memory/`.

## 10. [D-01] — Outcome: PASS (both firewalls cleared; contradicts the NULL-leaning prior)

Numbers (16,384 sightlines; before narrative):

- **C' cross-recipe ordering — reproduced (no KILL):** per-sightline means
  `mean s_2 = -0.00388 < mean s_3 = -0.00318 < mean s_4 = +0.00505`, with
  `s_2 < 0 < s_4`. Reproduces the `feedback-response-field` [D-01] global C-PASS at
  per-sightline resolution → feature/load consistent.
- **F within-recipe spread — PASS:** `IQR(s_4) = 0.00743`, `sep_between =
  |mean s_4 - mean s_2| = 0.00893`, **`SR = 0.83`** (>= 0.5 PASS bar; >> 0.25 NULL
  bar). Within StrongAGN, sightline-to-sightline directional spread is ~83% of the
  whole between-recipe shift — decisively NOT a per-recipe constant. Two-sided
  minority cross-check holds: `frac(s_4 > 0) = 0.862` → **13.8% of StrongAGN
  sightlines net-ADD absorption** while 86% net-remove (structured both-sided split,
  not a one-sided label).
- **E absorption orthogonality — PASS:** `Spearman(s_4, total_abs) = 0.41` (< 0.5
  ortho bar; far below the 0.7 NULL bar) and **`SR_resid = 0.68`** (>= 0.4 PASS
  bar) — most of the within-recipe spread SURVIVES partialling out total absorption.
  Also orthogonal-ish to the separability-vector norm (Spearman 0.21). The
  decorrelation is the theory showing up: the absorption coupling fell from
  **0.778 (magnitude, [D-01] predecessor) → 0.41 (signed mean)** — the SIGN carries
  information the MAGNITUDE could not, because sign ⊥ ‖·‖ by construction.
- Secondary (reported, non-gating): per-sightline median posfrac `p_c` = 0.00 (C2) /
  0.25 (C3) / 0.77 (C4); sightlines with < 5 responding px masked: 4698/2366/136.

**VERDICT — PASS (pre-committed: F-PASS AND E-PASS).** The SIGNED feedback response
is a **per-sightline directional axis**: within a single recipe, different
sightlines respond in genuinely different directions (SR=0.83, a structured 14%
sign-flipped minority), and that variation is largely orthogonal to total
absorption (SR_resid=0.68, |Spearman|=0.41). This is the FIRST positive among the
recent thin probes (confusability NULL, cluster-environment NULL, response-magnitude
NULL) and it **contradicts the PI's explicit NULL-leaning prior** (which expected
F-NULL: a per-recipe constant). Per [D-37], a surprising positive after a string of
nulls warrants MORE scrutiny — the robustness checks done here (IQR is
outlier-robust; the two-sided minority is structured not a noise sliver on noiseless
data; SR_resid survives absorption control) all hold, but this should not become
load-bearing until a defense-panel adversarial pass confirms it.

**CEILING (binding):** this characterizes the DIRECTION of feedback action per
sightline (ADD vs REMOVE Lyα absorption, varying sightline-to-sightline). It does
NOT beat the ~0.45 per-sightline 4-class classification ceiling ([D-13]) and makes
NO claim about it — it is an interpretive/embedding axis, not a classifier. Per §11,
this PASS authorizes ONLY the SCOPING of a future signed-response embedding track,
defense-panel-gated BEFORE any compute commitment; it does not authorize that
track's execution. NOT a paper trigger; parked verdict unchanged.

### Verification addendum — defense-panel adversarial pass (2026-06-24)

The PASS was stress-tested by the `defense-panel` (verdict: NEEDS WORK, named the
few-pixel test [#4] + SR calibration [#1] as the deciders). All decisive checks were
computed (`src/core/signed_response.py::verify_signed_response_pass` →
`results/signed_response_embedding/verification.json`). **The PASS SURVIVES**; the
panel forced the right computations and they came back in favor of the finding, with
one correction and one refinement:

- **#1 SR calibration — REFUTED.** Sign-permutation null (destroy within-sightline
  pixel-sign coherence): IQR(s_4) collapses 0.00743 → 0.00108, **sign-coherence
  ratio = 6.9×**; the signed mean is 7× larger than random sign-cancellation → the
  per-sightline signs are physically COHERENT, not a constant+noise cloud. SR
  bootstrap CI **[0.79, 0.88] excludes 0.5** → the bar is cleared with margin, not a
  small-denominator fluke.
- **#4 few-pixel domination — REFUTED.** Participation ratio (effective contributing
  pixels) = **156 (all) / 360 (extreme tail)**; top-5 pixels carry only **12.6% /
  3.6%** of the signed sum. The response is spatially distributed; "per-sightline
  axis" is not a per-line-core artifact. (The panel's #4 premise — that a few pixels
  could drive a *mean* of −0.85 — is also arithmetically impossible for a mean over
  2048 bounded pixels.)
- **#3 unphysical low-absorption tail — REFUTED (panel was fed a wrong figure
  reading).** The 100 most-negative-s_4 sightlines sit at total_abs **0.0385 vs
  0.0216 overall** — *above*-average absorption, NOT the diffuse bulk. Consistent
  with Nasir/Bolton+2017 (feedback acts in overdense gas). The extremes (min s_4
  −0.91; only 8 sightlines with |s_4|>0.3) are rare, coherent whole-region flips.
- **#7 c=4-only / forking path — STRENGTHENED.** All three recipes show coherent
  signs (ratio 6.8–9.6×) and within-recipe spread scaling with feedback strength
  (IQR 0.0026 → 0.0037 → 0.0074); the absorption coupling **sign-flips**
  Spearman −0.48 (StellarWind: adds more where there's more gas) → +0.41 (StrongAGN:
  removes more where there's more gas). A physically-coherent recipe-ordered pattern
  noise cannot produce.
- **#8 magnitude 0.778 unverifiable — CONFIRMED in-session.** Spearman(‖R_4‖,
  total_abs) recomputed = **0.778**, exactly the predecessor ledger value → the
  0.778→0.41 sign-decorrelation narrative is verified, not recalled.
- **#2/#5 orthogonality leak — MOSTLY REFUTED, one valid REFINEMENT.** Total
  absorption explains only **1.5% of Var(s_4)** (nonlinear/binned; linear 1.8%) →
  the DIRECTION is 98.5% orthogonal to absorption, far past the bar. BUT the
  conditional IQR of s_4 grows **10× across absorption deciles** (0.0018 → 0.0188) —
  the panel's "fan" is real: the *amplitude* of the directional spread is
  heteroscedastic in absorption. **Honest refinement:** the SIGN/direction is
  orthogonal to absorption strength; the MAGNITUDE of the signed response is not
  (you can only add/remove absorption where gas exists). This characterizes the axis;
  it does not dissolve it.

**Post-verification disposition: PASS CONFIRMED (refined).** The signed-response
direction is a real, coherent, recipe-ordered, physically-sensible per-sightline
axis whose central tendency is orthogonal to absorption strength and whose amplitude
is gas-gated. The defense-panel gate for *scoping* a future embedding track is met.
Open items the panel raised for that downstream track (NOT for this probe): a full
P_F(k) orthogonality test (#10, beyond scalar mean-absorption), a de-entangled sign
feature (#9, median/sign-fraction corroboration), and the heteroscedastic-amplitude
handling (#5) in any learned embedding. Ceiling and parked verdict unchanged.
