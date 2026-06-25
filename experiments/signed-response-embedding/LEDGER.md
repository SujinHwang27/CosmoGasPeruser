# LEDGER — signed-response-embedding

Command center for the signed-response embedding track. Single source of truth.
Branch `exp/signed-response-embedding` off `feature/signal-clustering-v2`.
Design: `TRACK_SPEC.md` (plan-of-record). Probe origin: `SCOPING.md`.

> **CEILING (binding, every figure/claim):** characterizes the DIRECTION of feedback
> action per sightline. Does NOT beat the ~0.45 4-class classification ceiling ([D-13]
> on the parent track) — an interpretive/embedding axis, not a classifier. NOT a paper
> trigger. The parked `reframe-suite` verdict is unchanged regardless of outcome.

---

## 1. The Pulse (Progress & Roadmap)

| Stage | Description | Status |
|---|---|---|
| S0 — Gate-1 P_F(k) orthogonality de-risk | direction ⊥ canonical content basis? | ✅ **DONE** — PASS |
| S1 — embedding construction + Gates 2,3 | handcrafted direction⊕magnitude; de-entangled; fan removed | ✅ **DONE** — Gate-2 ✓ (Spearman −0.06), Gate-3 ✓ (fan 1.4×) |
| S2 — structure analysis (THE deliverable) | cluster c=4 direction sub-block; surrogate-calibrated AMI/stability/gas-gate | ✅ **DONE** — PASS (modest, p=0.039) |
| S3 — stability + physical interpretation | cross-feature + bootstrap vs surrogate; Nasir/Bolton+2017 | ✅ **DONE** — TEMPERED (feature-sensitive) |

### ✅ Completed Milestones
- **2026-06-23**: probe — signed-response DIRECTION is a coherent per-sightline axis ([D-01], PASS).
- **2026-06-24**: defense-panel verification of the probe PASS — survives (refined: sign ⊥ absorption, magnitude gas-gated).
- **2026-06-24**: Gate-1 — direction ⊥ P_F(k) content basis ([D-02], PASS; RF-OOB R²=0.030, tail 0.034).
- **2026-06-24**: track OPENED (user unpark); defense-panel APPROVE-WITH-7-MODIFICATIONS to §4 ([D-03]).
- **2026-06-24**: S1 built (Gates 2,3 pass); S2 structure test PASS (modest, p=0.039) ([D-04]).
- **2026-06-25**: S3 cross-feature stability TEMPERED — direction is feature-sensitive, localized to
  the responding regions ([D-05]). Track complete.

---

## 2. Methodology & Architecture

**Question (NOT classification):** does the verified per-sightline *direction* of feedback response
support a handcrafted embedding that reveals incremental structure no existing representation
(separability vectors, wavelet, raw, total absorption, P_F(k)) encodes — because all five are
sign-blind?

**Counterfactual response field:** `R_c[i,x] = F_c[i,x] − F_1[i,x]` (>0 ⇒ absorption REMOVED), on
byte-identical skewers, noiseless S/N=∞ ⇒ R_c is the exact physical feedback response.

**Embedding (handcrafted, NOT learned — interpretability is the deliverable; a learned objective
would collapse to the gas gate on the ~92% bulk):**
- DIRECTION block (amplitude-free), primary = c=4 sub-block: `dir_unit₄ = ΣR₄/Σ|R₄| ∈[−1,1]`,
  `signfrac₄` (θ=0.05), `sign-autocorr₄`, `mean-run-length₄`.
- cross-recipe profile (c=2,3,4) → interpretation only (§4.2), NOT in the primary clustering.
- MAGNITUDE block (gas gate) carried SEPARATELY.

**Gates (pre-committed, TRACK_SPEC §3 + §4.0):** Gate-1 P_F(k) orthogonality ✅; Gate-2 de-entangled
feature; Gate-3 amplitude-fan removal; + 7 panel-required execution controls on the structure test
(responding-subset, sign-permutation surrogate null, stability-not-silhouette, closed gradient list,
AMI population-anchoring, c=4-only primary, gas-gate η² audit).

---

## 3. The Logic (Decision Log)

- **[D-01] Probe: signed-response DIRECTION is a coherent per-sightline axis (PASS)**: the SIGN of R_c
  is a per-sightline axis (sign-coherence 6.9× over a random-sign null, SR bootstrap CI [0.79,0.88]),
  orthogonal to scalar absorption in central tendency (1.5% of Var), present in all 3 recipes with the
  coupling sign-flipping −0.48 (winds ADD) → +0.41 (StrongAGN REMOVES). Defense-panel-verified. The
  *magnitude* is total absorption restated (NULL); only the *sign* is the new axis. `SCOPING.md` §10.
- **[D-02] Gate-1: direction ⊥ P_F(k) content basis (PASS)**: the amplitude-free direction
  (ΣR₄/Σ|R₄|) is not recoverable from the baseline per-sightline flux power spectrum
  (ridge-CV R²=0.022, RF-OOB R²=0.030, CCA ρ₁=0.324; high-absorption-tail RF-OOB=0.034 ⇒ not a
  bulk-noise false PASS). A power spectrum is sign-blind by construction; confirmed empirically. The
  axis is novel against the field's real content representation, not just scalar absorption.
- **[D-03] Track opened; defense-panel APPROVE-WITH-7-MODIFICATIONS**: user authorized unpark
  (2026-06-24). The panel approved the S1–S3 execution plan conditional on 7 modifications to the
  structure-test success criteria (§4.0) — chiefly: cluster the responding subset not the full
  population (with a <70%-largest-cluster degeneracy guard), calibrate every bar against a
  sign-permutation surrogate null (not absolute), use stability+gap-statistic not silhouette,
  pre-register a closed 2-quantity physical-gradient list, anchor AMI on this population, restrict the
  primary clustering to the c=4 sub-block, and add a gas-gate η²(total_abs)<0.30 audit. Without these,
  a degenerate one-giant-cluster partition would score a false "new structure" PASS (the
  cluster-environment-gradient ghost). Execution of S1+ proceeds under the revised §4.
- **[D-04] S2 structure test: PASS (modest, robust)**: the c=4 direction embedding partitions the
  responding subset (n=15,805, 96%) into a non-degenerate (largest cluster 34%, 4 real clusters +
  a 4-sightline shard), absorption-orthogonal (total_abs between-cluster eta^2=0.079, far below the
  0.30 gas-gate bar) structure that is **modestly but reproducibly more stable than the
  sign-permutation surrogate** (real ARI 0.958; all 3 seeds 0.952/0.958/0.967 > null 95th-pct 0.947;
  permutation **p=0.039**) — so the COHERENT DIRECTION adds real cluster structure beyond magnitude
  alone. It is a different axis from the existing absorption-ordered reps (AMI 0.022/0.032, below even
  the 0.109 the two existing reps share with each other) and relates to the recipe-ordered coupling
  (Spearman(dir_unit4, signfrac4-signfrac2)=0.43; that coupling quantity is absorption-clean,
  Spearman 0.086 with total_abs). Gate-2 ok (Spearman(dir_unit4,total_abs)=-0.06), Gate-3 ok (fan 1.4x).
  **HONEST STRENGTH: MODEST** — p~0.04 is real-but-not-strong; the 2D view reads as a directional
  CONTINUUM sliced into bins more than discrete islands; the AMI "different-axis" evidence is weak in
  absolute terms (the panel-flagged low-AMI regime, anchor only 0.109) and leans on the not-gas-gate +
  coupling-gradient evidence. Clears all panel-hardened §4.1 bars. CEILING: interpretive direction
  axis; does NOT beat the ~0.45 ceiling; NOT a paper trigger. Any downstream use needs a defense-panel
  review of the RESULT (per §5.3 / SCOPING §11).
- **[D-05] S3 cross-feature stability: TEMPERED (feature-sensitive); track complete**: the direction
  axis is robust between the two responding-pixel-weighted variants (unit-direction vs sign-fraction
  Spearman **0.89**) but NOT vs the bulk-weighted median-sign (0.26 / 0.19; **min 0.19 « the 0.6 bar**)
  — the median is ~0 for the transparent bulk and does not see the direction. So the directional
  structure is **localized to the responding regions**, not a feature-robust whole-sightline embedding;
  the claim narrows accordingly (consistent with §4.4's pre-committed re-verb to the responding
  minority). Physical interpretation SOLID and unchanged: recipe-ordered coupling — net signed
  response flips −0.0039 (StellarWind ADDS) → +0.0051 (strong-AGN REMOVES); posfrac 0.16 → 0.71; the
  direction tracks this coupling (Spearman 0.555), which is absorption-clean (Spearman −0.31 with
  total_abs); maps to Nasir/Bolton+2017 feedback-in-overdensities (winds enrich/compress → more Lyα;
  AGN heats → less HI/Lyα). **NET TRACK OUTCOME:** the embedding's incremental structure over the
  probe's scalar directional axis is **modest AND feature-sensitive** — the durable contribution is
  the directional PHYSICS (probe [D-01] + Gate-1 [D-02]), not a robust new embedding representation.
  CEILING holds; NOT a paper trigger; parked verdict unchanged.

---

## 4. The Data (Lineage & Governance)

| Area | File | Metadata |
|---|---|---|
| **Input flux** | `data/preprocessed/Sherwood_z0.3_inf/{1..4}/flux.npy` | (16384, 2048) float64; noiseless; Δv=2.6365 km/s/px (from `vel.npy`) |
| **Existing K=5 labels** | `data/feature_discovery/labels_{wavelet,raw}_k5.npy` | (16384,) int; the AMI comparison targets + population anchor |
| **Separability vectors** | `data/feature_discovery/fingerprints_{wavelet,raw}.npy` | (16384, 24) float64 |
| **Gate-1 P_F(k) basis** | (computed in-memory) | `FluxPowerSpectrumTransform`, 52 log-k bins, baseline C1 |
| **Gate-1 result** | `results/signed_response_embedding/gate1_pk_orthogonality.json` (+ fig) | PASS |
| **S1/S2 embedding+structure** | `results/signed_response_embedding/structure_*.json` (+ figs) | (to be written by the build) |

NO new data; all inputs already local and verified this session. Results CSVs/JSON are git-tracked
(`cache:false`). Heavy `.npy` (if any embedding arrays > 10 MB) go through DVC per the heavy-artifact rule.

### Responsibility Matrix
- core-implementer: `src/core/signed_response_embedding.py`, the structure pipeline.
- support-researcher: clustering evaluators / figures (S2/S3) if delegated.
- PI: gate sign-off; defense-panel: S1–S3 plan APPROVE-WITH-MODS ([D-03]) + any PASS before downstream.

---

## 5. Evaluation Plan

Primary (S2, TRACK_SPEC §4.1 REVISED): cluster the c=4 direction sub-block (responding subset, ≥20
responding px), K=5. **PASS** = largest cluster <70% AND bootstrap-ARI stability > 95th pct of the
sign-permutation surrogate AND total_abs between-cluster η² <0.30 AND AMI vs both existing clusterings
below the related-axes anchor AND structured (gap-stat positive OR a pre-registered Holm-significant
physical gradient). **NULL** = any of: largest cluster ≥70% / stability ≤ null 95th pct / η² ≥0.30 /
AMI ≥0.40 with either existing.
Secondary (§4.2): physical interpretation via the recipe-ordered coupling flip.
Tertiary (§4.3): cross-feature rank-corr ≥0.6 + stability CI above the surrogate null.
Pre-committed NO-PUBLICATION path (§4.5): Gate-1 NULL or §4.1 NULL ⇒ recorded NULL, not a paper claim.

---

## 6. Visualization & Artifacts
- **Gate-1**: `results/signed_response_embedding/figs/gate1_pk_orthogonality.png` — direction not
  predictable from P_F(k) (OOB band flat). Insight: the direction axis is novel against content.
- **S2 structure**: `results/signed_response_embedding/figs/structure_s2.png` (+ `structure_s2.json`).
  Insight: the direction embedding partitions the responding subset into a stable (p=0.039 vs
  sign-permutation null), absorption-orthogonal directional structure — modest but real; reads as a
  directional continuum sliced into bins.
- **S3 stability/interpretation**: `results/signed_response_embedding/figs/s3_stability_interpretation.png`
  (+ `s3_stability_interpretation.json`). Insight: direction robust across responding-pixel-weighted
  variants (0.89) but feature-sensitive vs the bulk-weighted median (0.19) → localized to responding
  regions; recipe-coupling flip (winds ADD → strong-AGN REMOVES) is solid.
- Tracker run_ids: (none — CPU in-process; no MLflow run for the de-risk/structure probes).

---

## 7. Session History & Next Handoff

### Session Snapshot: June 24–25, 2026 (track complete: S0→S3)
- Probe PASS ([D-01]) defense-panel-verified; Gate-1 PASS ([D-02]); track opened ([D-03]).
- §4 success criteria hardened with the 7 panel-required execution controls.
- S2 PASS ([D-04]) — modest (p=0.039), absorption-orthogonal, not the gas gate.
- S3 TEMPERED ([D-05]) — direction is feature-sensitive (robust unit/sign-fraction 0.89 but not
  bulk-weighted median 0.19), localized to the responding regions. Physical coupling solid.
- **TRACK COMPLETE.** Net: a real, physically-coherent, but MODEST and FEATURE-SENSITIVE directional
  structure. Durable contribution = the directional physics (probe + Gate-1); the embedding adds only
  modest, feature-sensitive incremental structure over the scalar axis already in hand.

### Immediate Next Steps
- None required. Track is closed at S3. The learned-embedding variant (§2.2) remains explicitly
  DEFERRED and separately-gated; the bulk/feature-sensitivity lessons here argue against it unless a
  new-input gate opens (higher-z / density catalog). NOT a paper trigger.

### Blockers
- None. Track complete; parked verdict unchanged.
