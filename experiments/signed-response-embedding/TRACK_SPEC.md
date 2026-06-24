# TRACK DESIGN SPEC — signed-response-embedding (plan-of-record)

**Status:** SCOPED-BUT-DEFERRED (PI worth-ruling + design, 2026-06-24). Execution gated.
**Branch:** `exp/signed-response-embedding` (the thin probe lives here; reuse in-place per exFAT policy).
**Origin:** the word2vec "meaning-from-context" reframe; seed = the defense-panel-VERIFIED
PASS of the thin probe ([D-01], `SCOPING.md` §10 + Verification addendum).
**This is the plan-of-record. A full `LEDGER.md` is created only at execution-authorization (§5.1).**

---

## 0. WORTH-RULING (the load-bearing section)

**Recommendation: SCOPE-BUT-DEFER. Record the design now; do NOT authorize execution now.
Execution is gated behind the named condition in §0.4.**

### 0.1 What pulls AGAINST opening now
- **(a) The binding ceiling is real and this track does not touch it.** Interpretive/embedding
  axis; cannot beat the ~0.45 4-class ceiling ([D-13]). The usable-classifier question was
  already hardened on the parked `reframe-suite` track. A signed-response embedding adds zero
  classification headroom; its value is interpretation/structure — a softer, degeneracy-prone surface.
- **(b) The project is PARKED for paper-writing; paper-trigger is user-owned.** A heavy track now
  competes with the close-and-write disposition two panel rounds judged defensible.
- **(c) PI's own prior DECLINE of the near-identical follow-on** (`cluster-environment-gradient`,
  2026-06-23): a heavier instrument would re-hit the ~92%-bulk population-level null with a more
  degeneracy-prone tool. The bulk is still there; the probe's §6 audit confirms the sign is
  **near-arbitrary** on the majority of the domain.

### 0.2 What CHANGED — does the verified positive move the calculus?
- Does NOT move (a) — ceiling unchanged.
- Does NOT move (b) — parked verdict / user-owned paper trigger unchanged.
- DOES move (c), partially: the decline rested on "re-hits the SAME population null." The
  signed-response probe already produced a **non-null, panel-survived positive** (sign-coherence
  6.9×, SR CI [0.79,0.88], 98.5% orthogonal to absorption in central tendency, recipe-ordered
  coupling sign-flip −0.48→+0.41) — a verified signal the existing five reps provably discard
  (sign ⊥ ‖·‖, sign ⊥ |F|). So that specific failure mode does NOT apply. **BUT** the verified
  positive is a population-aggregate scalar axis already in hand; the open *track* question is
  whether promoting it to a richer per-sightline EMBEDDING reveals **structure the content/
  separability reps do not** — and that is exactly where the ~92% bulk null can re-bite (sign
  unconstrained on the bulk). The prior concern is **narrowed, not eliminated.**

### 0.3 Synthesis
The verified positive **earns the design and the entry-gate work**, not immediate execution:
1. the probe already delivered the headline (directional axis is real, orthogonal-in-central-tendency);
2. the entry gates are load-bearing pre-work that can NULL the track — esp. Gate 1 (full P_F(k)
   orthogonality), which needs a per-sightline P_F(k) basis that **does not exist on this branch**;
3. the deliverable surface is rule-7-fragile and requires a rule-14 no-publication rescue (§4.5).

### 0.4 The named gating condition (what lifts the deferral)
> **Execution is authorized when, and only when, ALL of:**
> **(i)** the user explicitly authorizes opening this as a methodology track (user owns unpark); AND
> **(ii)** Gate 1 (full P_F(k) orthogonality, §3.1) PASSES on the de-risk pre-work — a self-contained,
> ≤2-script, no-new-data analysis runnable WITHOUT opening the full track; AND
> **(iii)** a dedicated `defense-panel` review APPROVES the S1–S3 execution plan.

The deferral is **not "wait indefinitely"**: run the Gate-1 de-risk first (probe-class), and only
if it PASSES does opening the full track go back to user + panel. If Gate 1 NULLs, the track closes
before it opens, at probe cost — the economically and scientifically correct ordering.

**Cascade verb (rule-2):** the scalar directional axis is now **empirically established**, not a
hypothesis. But the *embedding-track* claim ("a richer signed-response embedding reveals incremental
structure over existing reps") is a NEW one-level-up claim that inherits no credit from the scalar
PASS and sits behind the falsified magnitude prior. Honest verb: **"first test of whether the
verified scalar directional axis supports a per-sightline embedding with incremental structure over
existing representations — bulk-threatened, P_F(k)-orthogonality-untested, NULL-plausible."**

---

## 1. The scientific question (that existing reps do NOT answer)

Does the *direction* of feedback action — verified per-sightline ([D-01]) as a coherent ADD-vs-REMOVE
axis orthogonal in central tendency to absorption strength — support a **per-sightline representation
capturing structure none of the five existing reps encode**, because all five are sign-blind?

| Representation | Encodes | Why sign-blind to response direction |
|---|---|---|
| Separability vectors (24-dim) | class separability across 6 OVO RBF boundaries | sign of R_c not recoverable from decision distances; Spearman(s_4,sep-norm)=0.21 |
| Wavelet (db8) | within-recipe absorption structure | no counterfactual ΔF across recipes |
| Raw A=1−F | absorption strength/profile | magnitude only; ‖R‖ Spearman 0.778 w/ total_abs; sign ⊥ \|F\| |
| Total absorption (scalar) | mean depth | firewall E target; explains 1.5% of Var(s_4) |
| **P_F(k)** (flux power spectrum) | the canonical sufficient statistic feedback modifies | **a power spectrum is phase/sign-insensitive by definition — mathematically cannot carry the response sign.** The crux of Gate 1 and the strongest a-priori novelty argument. |

The signed-response axis is the **one DOF the field's canonical content rep (P_F(k)) is structurally
guaranteed to discard.** Strong *in principle* — but must be **measured** (Gate 1), because the
*embedding* may still correlate with P_F(k) through the gas-gated amplitude even if the *direction*
does not. NOT "improve classification"; NOT "discrete environment strata" (NULLed); NOT "a new
magnitude axis" (NULLed).

---

## 2. Embedding CONSTRUCTION — HANDCRAFTED (recommended), not learned

**Do NOT build a learned (autoencoder/contrastive) embedding in this track.** Reasoning:
1. interpretability IS the deliverable — a learned latent would need post-hoc interpretation to
   recover the signed/spatial structure we can write down directly;
2. infra cost is real (no torch in `src/core/`) on a parked project for an interpretive axis;
3. degeneracy-proneness: with sign near-arbitrary on the ~92% bulk, a contrastive/AE objective will
   collapse to encoding absorption amplitude (the gas-gated fan) — re-deriving the magnitude null
   with a heavier instrument (the exact `cluster-environment-gradient` decline rationale);
4. the handcrafted features are de-entangled by construction (Gate 2).

### 2.1 Candidate feature set (handcrafted, de-entangled, per-sightline)
For sightline `i`, recipe `c∈{2,3,4}`, `R_c[i,x]=F_c[i,x]−F_1[i,x]` (>0 ⇒ absorption REMOVED):

**A. De-entangled sign features (Gate-2: amplitude-free):**
- `signfrac_c[i] = frac_x(R_c>0 | |R_c|>θ)` — responding-pixel sign fraction (θ swept, NOT tuned).
- `med_sign_c[i] = sign(median_x R_c)`, `median_R_c[i]` — amplitude-robust vs the mean `s_c`.
- `dir_unit_c[i] = s_c[i] / (‖R_c[i]‖₁ + ε)` — amplitude-normalized net direction (Gate-3 component).

**B. Spatial sign-structure (the "context"/word2vec payload):**
- `sign_runs_c[i]` — contiguous same-sign run count/length (ADD some regions + REMOVE others, or
  coherent whole-sightline flip? probe #4 verified 156–360 effective pixels → real to measure).
- `sign_autocorr_c[i]` — lag autocorrelation of `sign(R_c)` (spatial coherence length).

**C. Per-recipe direction profile (the cross-recipe "context vector"):**
- `(dir_unit_2, dir_unit_3, dir_unit_4)` and `(signfrac_2, signfrac_3, signfrac_4)` — the
  per-sightline trajectory of direction across the feedback-strength ladder (literal word2vec
  analogy: meaning = profile of counterfactual outcomes; verified recipe-ordered coupling flip).

**D. Gas-gated magnitude, SEPARATED (Gate-3):** `‖R_c[i]‖₁` carried as a separate, labeled magnitude
block, NOT mixed into the direction block. Embedding = **direction-block (amplitude-free) ⊕
magnitude-block (gas gate)** so structure analysis can run on direction alone.

### 2.2 Learned variant
If the handcrafted embedding PASSES, a learned variant is a **separate, separately-gated, infra-heavy
follow-on track** — not part of this spec, not pre-authorized.

---

## 3. Entry GATES (pre-committed, run BEFORE the embedding is built)

### 3.1 GATE 1 — Full P_F(k) orthogonality (#10) — THE HARD DECIDER (standalone de-risk)
**Pre-work (must be built):** `src/core/transforms.py` has NO power-spectrum transform. Add
**`FluxPowerSpectrumTransform(BaseTransformer)`** computing per sightline `P_F(k)=|rFFT(δ_F)|²` on
`δ_F = F/⟨F⟩_global − 1` (GLOBAL mean-flux normalization — per-class ⟨F⟩ would leak the recipe label,
the [D-13] confounder), binned to the reframe-suite k-grid, physical k-axis from the
data-engineer-confirmed per-pixel Δv. Owner: `core-implementer` + `data-engineer`. Port from the
parked `reframe-suite` cloud code (`cloud_runs/pkstage1-*`). Reuses `SignalClusteringData`; NO new data.

**Statistic:** compute `pk_1[i]` (P_F(k) on baseline C1 — the sightline's intrinsic content). Then,
on the **amplitude-free direction** `dir_unit_4 ⊕ signfrac_4` (NEVER raw `s_4` — the magnitude would
leak a false NULL):
- Linear: ridge/CCA → report R² and first canonical correlation ρ₁.
- Nonlinear: RF-regress direction ~ `pk_1` → out-of-bag R² (strongest test).
- **PASS (orthogonal/novel):** linear R² < 0.10 AND RF-OOB R² < 0.15 AND ρ₁ < 0.40.
- **NULL (P_F(k) shadow):** RF-OOB R² ≥ 0.30 OR ρ₁ ≥ 0.60 → track NULLs at Gate 1, no embedding built.
- **CHARACTERIZE (ambiguous):** between bars → proceed only if panel judges the residual-orthogonal
  component worth embedding. Default NULL-leaning.

### 3.2 GATE 2 — De-entangled sign feature (#9) — construction constraint
Primary direction features must be amplitude-free; the verified axis must corroborate on them.
- **PASS:** |Spearman(`dir_unit_4`, total_abs)| < 0.41 (not worse than the entangled `s_4`) AND a
  non-trivial two-sided sign minority (5–95%) on `signfrac_4`.
- **NULL:** axis survives only on `s_c` and collapses on de-entangled features → the probe PASS was
  magnitude leaking through the sign-weighted mean → track NULLs at Gate 2.

### 3.3 GATE 3 — Heteroscedastic-amplitude handling (#5) — construction constraint
The verified 10× gas-gated amplitude fan must be handled by separating direction from magnitude.
- **PASS:** `dir_unit` decile-IQR-across-absorption is flat-to-mild (fan factor < 3×, vs 10× on raw
  `s_4`) — normalization removed the fan; magnitude carried separately, labeled as the gas gate.
- **NULL/FAIL:** fan survives normalization (`dir_unit_4` decile-IQR still ≥ 5×) → direction
  irreducibly amplitude-contaminated → hard CHARACTERIZE/NULL, surface to panel.

---

## 4. SUCCESS CRITERIA (honest, ceiling-aware, rule-7/14 compliant)

**NOT classification accuracy** (ceiling-barred). Embedding is valid iff:

### 4.1 Primary — INCREMENTAL STRUCTURE over existing reps
Does clustering/2D-embedding the direction block reveal structure not in the separability-vector
and content/wavelet clusterings?
- **PASS:** AMI < 0.15 with EACH existing clustering (genuinely different axis) WHILE itself
  structured (silhouette ≥ 0.25 on the direction block, OR a physical gradient Spearman ≥ 0.5 to a
  named quantity that is NOT total absorption).
- **NULL:** AMI ≥ 0.40 with either existing clustering (re-derives a known partition) OR no structure
  beyond a zero-centered noise cloud (silhouette < 0.10, no physical gradient) — the bulk null
  re-biting. Either NULLs the track.

### 4.2 Secondary — defensible physical interpretation
- **PASS:** per-recipe direction profile reproduces the recipe-ordered coupling sign-flip at
  per-sightline resolution AND maps to a literature-citable mechanism (Nasir/Bolton+2017
  feedback-in-overdensities).
- **NULL:** no coherent physical reading → CHARACTERIZE-only, not a usable embedding.

### 4.3 Tertiary — cross-representation stability
- **PASS:** axis reproduces across de-entangled feature variants (rank-corr ≥ 0.6 among
  median-sign / sign-fraction / unit-direction) AND a multi-seed bootstrap CI on the primary
  structure statistic excludes the NULL bar (rule-9 statistically-confirmed stability).
- **NULL:** structure flips with feature choice or its bootstrap CI straddles the NULL bar.

### 4.4 Anti-degeneracy audit (rule-3, MANDATORY)
On the ~92% low-absorption bulk the sign of R_c is **near-arbitrary** → the embedding has near-zero
signal on the majority of sightlines. Every PASS bar is un-manufacturable by a zero-centered noise
cloud (AMI-low-AND-structured fails silhouette under noise; two-sided-minority/IQR checks distinguish
structured from noise spread). **If the PASS lives only on the ~8% high-absorption tail, the honest
claim is "a direction axis on the responding minority," NOT "a per-sightline embedding" — pre-commit
to re-verbing to the narrower claim.**

### 4.5 Pre-committed NO-PUBLICATION failure path (rule-14(ii), BINDING)
If Gate 1 NULLs OR the primary criterion (4.1) NULLs → recorded as a NULL D-XX, explicitly NOT a
paper-headline claim and NOT a paper trigger. No "weak structure" rescue, no bar-widening. A NULL
closes the signed-response *embedding* line; the scalar directional axis from the probe stands alone.

---

## 5. STAGES, structure, execution gate

### 5.1 Track structure
- Branch `exp/signed-response-embedding` (reuse, in-place).
- Promote this design to a full `experiments/signed-response-embedding/LEDGER.md` **only at
  execution-authorization** (§0.4). Until then the design lives here; the LEDGER is created at unpark.
- **NOT a paper trigger** at any stage. Findings land in the track LEDGER + `.claude-memory/`.

### 5.2 Stages (dependency order)
| Stage | Scope | Owner | Artefact | Gate |
|---|---|---|---|---|
| **S0 — Gate-1 de-risk (PRE-WORK; runnable WITHOUT opening the track)** | build/port `FluxPowerSpectrumTransform`; compute `pk_1`; run P_F(k)-orthogonality on amplitude-free direction (§3.1) | core-implementer + data-engineer | `transforms.py::FluxPowerSpectrumTransform`; `results/.../pk_orthogonality.json`; 1 figure | **Gate 1.** NULL ⇒ track closes here at probe cost. |
| **S1 — embedding construction** | build the direction⊕magnitude embedding (§2.1); verify Gates 2,3 | core-implementer | `src/core/signed_response_embedding.py`; feature arrays | **Gates 2,3.** |
| **S2 — structure analysis** | cluster/2D-embed direction block; AMI vs existing; physical-gradient test (§4.1,4.2) | support-researcher | `results/.../structure_*.csv`; AMI table; figure | **Primary 4.1.** |
| **S3 — stability + interpretation** | cross-feature + bootstrap stability (§4.3); physical mechanism write-up | support-researcher | bootstrap-CI table; LEDGER §3 note | **4.2, 4.3.** |

### 5.3 What GATES execution (binding, layered)
1. **User authorization** to open the methodology track (project parked; user owns unpark).
2. **Gate 1 (S0) PASSES.** S0 runs under probe-class caps (no new data, ≤2 scripts, ≤1 new transform,
   ≤1 figure) WITHOUT opening the full track. NULL ⇒ track never opens.
3. **Dedicated `defense-panel` review of the S1–S3 plan APPROVES before any S1+ compute** (re-checks
   the rule-14 no-publication path, the anti-degeneracy audit, the Gate-1 result).

### 5.4 Compute budget
Full track is **CPU-only, in-process** (handcrafted features + RF/clustering on 16,384×~few-hundred-dim;
probe ran in seconds). No GPU, no cloud, no torch. The learned variant (if ever) is a separate track
with its own GPU/cloud budget sign-off — not pre-authorized.

---

## 6. CEILING, parked verdict, cascade verb (restated)

**CEILING (binding, verbatim for any caption / LEDGER §3):**
> This track characterizes the DIRECTION of feedback action per sightline. It does NOT beat the ~0.45
> 4-class ceiling ([D-13]) and makes NO claim about it — an interpretive/embedding axis, not a
> classifier. A PASS yields an interpretation axis; a NULL closes the signed-response *embedding* line
> (the scalar axis from the probe [D-01] stands regardless). The parked verdict is unchanged either way.

**Relationship to the parked verdict:** does NOT reopen the parked `reframe-suite` verdict (concerns
the usable classifier / stacked P_F(k)); this track is orthogonal by construction (sign/direction;
P_F(k) is sign-blind). No new Sherwood-Lyα classification compute implied. Only the already-loaded
`Sherwood_z0.3_inf` flux is touched; no new data/snapshot.

**Review provenance (rule-6/15):** PI-only worth-ruling + design, with deferred-panel-review tracked
(§5.3(3) required before execution). PI sign-off on this spec is **PROVISIONAL**, lifted only by the
§5.3 panel APPROVE.

---

**Disposition for the user:** SCOPE-BUT-DEFER — design recorded in full; the next *authorized* action
is the Gate-1 P_F(k)-orthogonality de-risk (probe-class, no new data, runnable without opening the
track). The methodology track opens only if Gate 1 PASSES **and** the user authorizes unpark **and**
the defense-panel approves the execution plan.
