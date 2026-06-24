# SCOPING — feedback-response-field (thin control-first probe)

Status: SCOPED (PI design-panel + worth-ruling, 2026-06-23). Branch
`exp/feedback-response-field` off `feature/signal-clustering-v2`. Caps: ONE
script · ONE figure · ONE D-XX · NO new data · NOT a paper trigger; parked
verdict unchanged.

> Origin: user reframe — "produce a representation of sightlines from CONTEXT,
> like word embeddings." The sightline's context = its OWN 4 counterfactual
> outcomes (same byte-identical skewer under 4 feedback recipes). Verified gate:
> `axis/vel/wave.npy` byte-identical across all 4 classes; response `|F1-F4|`
> mean 0.0135, ~6% of pixels move >0.05. Data is NOISELESS (S/N=inf), so
> `R_c = F_c - F_1` is the EXACT physical feedback response — no measurement-noise
> residual to fight; the real risk is degeneracy (is it just total absorption /
> the separability vector restated?).

## 0. The new question (NOT classification)
Prior work asked "can I tell the 4 classes apart from ONE spectrum?" → no (~0.45
ceiling). This asks a different question: **where, in what direction, and with
what morphology does feedback act on each line of sight?** — class as *treatment
index*, not target. The response field `R_c(x) = F_c(x) - F_1(x)` (positive =
absorption REMOVED) is the candidate per-sightline axis; this probe de-risks
whether it is real/orthogonal/structured BEFORE building the full embedding.

## 1. Pre-committed controls + bars (rule-5 symmetric; in src/core/feedback_response.py)
- **A — geometry-binding:** same-geometry response `||F4[i]-F1[i]||` vs
  cross-geometry `||F4[j]-F1[i]||` (j≠i). PASS: median(same) < 0.5·median(cross)
  AND frac(same<cross) > 0.9. (Confirms the 4 paired spectra are a tight
  counterfactual family; if this fails the pairing is void.)
- **B — concentration:** top-10% of geometries by response carry > 40% of total
  response (a responsive minority, not uniform).
- **C — directionality:** sign of response differs across recipes — fraction of
  responding pixels with R>0 (absorption removed) differs between StrongAGN(4)
  and StellarWind(2) by > 0.10. (The directional DOF magnitude reps discard.)
- **D — orthogonality (the key anti-degeneracy):** response magnitude is NOT a
  proxy for total absorption OR the separability-vector norm —
  |Spearman(rho4, mean(1-F1))| < 0.6 AND |Spearman(rho4, ||sep-vector||)| < 0.6.

## 2. Verdict
- **PASS** = A AND D AND (B OR C): a real, geometry-bound, orthogonal new axis
  (concentrated and/or directional) → worth building the full response embedding
  + the OT (kinematic) axis. CEILING (binding): characterizes WHERE/HOW feedback
  acts; does NOT beat the ~0.45 per-sightline 4-class classification ceiling and
  claims nothing about it.
- **NULL** (valid end-state, report as-observed): D fails (response = total
  absorption / separability restated) OR A fails (pairing void) OR no B/C
  structure. Lead with the numbers; do not re-run hunting for a pass.

## 3. Honest risks (PI)
Even on noiseless data: the response could (i) merely track total absorption
("deep-line sightlines respond more" — the deep-line proxy the content scalars
already covered), or (ii) replay the [D-13] failure (high-response minority =
high-saturation minority, not new physics). Control D is the firewall. Cascade
verb: highest-leverage unexplored structural axis (the paired counterfactual),
expected on physical grounds, but at-risk of replaying [D-13] one level up —
cannot be presented at high confidence; the controls force the risk to declare.

## 4. Status
RUN & CLOSED — **NULL (productive)**, 2026-06-23. `src/core/feedback_response.py`
+ `scripts/run_feedback_response.py`; outputs `results/feedback_response_field/`.

## 5. [D-01] — Outcome: NULL on magnitude, but a real DIRECTIONAL finding

Numbers (16,384 sightlines; before narrative):
- **A geometry-binding PASS:** median same-geometry response 1.868 vs cross-geometry
  4.120; 95.8% of sightlines respond less to feedback than to being a different
  skewer → the 4 paired spectra ARE a tight counterfactual family.
- **D orthogonality FAIL (decider):** Spearman(rho4, total absorption) = **0.778**
  (> 0.6 bar) → the response *magnitude* `||F4-F1||` largely restates total
  absorption ("deep-line sightlines respond more"). (Orthogonal-ish to the
  separability-vector norm, 0.457.) The response-MAGNITUDE embedding is NOT a new
  axis.
- **B concentration FAIL:** top-10% carry only 25.6% of total response (the
  magnitude is as spread as total absorption — consistent with D).
- **C directionality PASS — the genuine positive:** sign of the response is
  strongly recipe-dependent and physically coherent. Fraction of responding
  pixels with absorption REMOVED (R>0): StellarWind **0.157** → WindAGN 0.290 →
  StrongAGN **0.711**; mean signed response flips **−0.0039 (C2 ADDS absorption)**
  / −0.0032 (C3) / **+0.0051 (C4 REMOVES absorption)**. Gap 0.554. This directional
  degree of freedom is discarded by every magnitude/content representation.

**VERDICT — NULL (pre-committed: PASS needed A∧D∧(B∨C); D failed).** The response
*magnitude* as posed is not a distinct embedding axis — it is mostly total
absorption restated (the [D-13] deep-line trap, caught by control D). BUT the
probe delivered one real, pre-registered positive: **the SIGN of the feedback
response carries strong, physically-sensible, recipe-discriminating structure**
(stellar/wind feedback ADDS Lyα absorption; strong-AGN REMOVES it; consistent with
H3's C4-lowest-absorption). This REFINES rather than rescues the idea: a future
representation should embed the response DIRECTION/sign, NOT its magnitude.
CEILING (binding): this characterizes WHERE/HOW feedback acts; it does NOT beat
the ~0.45 per-sightline 4-class classification ceiling and claims nothing about
it. NOT a paper trigger; parked verdict unchanged. Follow-on (signed-response
embedding) is a NEW question, not auto-pursued.
