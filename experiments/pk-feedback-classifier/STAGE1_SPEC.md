# Stage 1 Spec — 64-cell G1–G6 PASS/FAIL routing table

**Binding:** `experiments/pk-feedback-classifier/LEDGER.md` [D-09], [D-15]–[D-22], [D-23] C1/C3/C5/C7. Methodology document per [D-23] C5 — the routing decision per cell is a PI call, NOT an implementation detail. The core-implementer enumerates and proposes; **PI APPROVE required before infrastructure-manager fires sbatch**.

This document does NOT change protocol numerics — gate thresholds, fold protocol, and metric definitions remain as the LEDGER specifies. It maps each of the 2⁶ = 64 PASS/FAIL combinations of G1–G6 to one of the pre-committed outcome bands from [D-09].

---

## 1. Gate definitions (reproduced for routing purposes)

| Gate | Metric | PASS condition | Source |
|:---|:---|:---|:---|
| **G1** | 4-class 5-fold-CV balanced_acc, 10-seed bootstrap | lower 16th-pct CI ≥ 0.65 at some (regime, M) | [D-09] |
| **G2** | 3-class {C1,C2,C3} balanced_acc at M=64, per_sightline regime | observed median > 99th percentile of 1000-permutation null | [D-18] |
| **G3** | RF importance fraction in k ∈ [0.04, 0.15] s/km | median across (seeds × folds × M-sweep × regimes) ≥ 0.50 | [D-09] / [D-11] |
| **G4** | C2-vs-C3 binary sub-confusion balanced_acc | median ≥ 0.60 — **load-bearing scientific gate** per [D-23] C3 | [D-09] / [D-23] C3 |
| **G5** | Per-class \|Pearson r(⟨F⟩_los, predict_proba_max)\| | all 4 classes \|r\| < 0.3 on median across (10 seeds × 5 folds × M-sweep) | [D-21] / [D-23] C7 |
| **G6** | Ablation Δ on top-3 mid-k bins (idx 10/11/12) at M=64, per_sightline regime | Δ ≥ `max(5pp, 2 × empirical_SD)` where SD is computed from the 50-fold (10 seed × 5 fold) ensemble | [D-16] / [D-23] C1 |

**G5 sub-states** (binary in the truth table; routed via the side-column):

- **G5 PASS** — all 4 classes \|r\| < 0.3.
- **G5 FAIL → side column `G5_C4_only`**:
  - `G5_C4_only = true` ⇔ C4 alone has \|r\| ≥ 0.3, C1/C2/C3 all < 0.3. Routes to **"Partial PASS — C4-confounder confirmed via G5"** band per [D-23] C7.
  - `G5_C4_only = false` ⇔ any of C1/C2/C3 has \|r\| ≥ 0.3 (whether or not C4 also does). Routes to **"Partial PASS — mechanism mis-anchored"** or NULL depending on joint G3/G4 status per [D-23] C7(iii).

The 64-cell table is enumerated in §3 below. Each cell carries the binary G5 PASS/FAIL value; rows with G5 FAIL annotate which sub-state controls the route.

---

## 2. Band-routing legend (pre-committed; from [D-09] + [D-23] C3/C7 amendments)

| Code | Band | Plain-English meaning |
|:---|:---|:---|
| **CP** | Clear PASS | All 6 gates clean on lower-CI edge; deliverable surface = "stacked-P(k) population-statistics signal of mid-k feedback variance at z ≈ 0.3, consistent with Khaire+2024 / Tillman+2024 extrapolated from z = 0.1." |
| **PP-C4** | Partial PASS — C4-confounder confirmed | The 4-class lift exists, BUT it is carried by C4's structural ⟨F⟩ offset (≈ 0.5% per [D-07]) leaking into mid-k via P(k) *shape*. The mid-k mechanism on the physics-relevant {C1, C2, C3} subspace is not confirmed. |
| **PP-C4G5** | Partial PASS — C4-confounder confirmed via G5 (per [D-23] C7 (ii)) | Subset of PP-C4 specifically triggered by G5 with `C4_only_violator=true`. Distinct from PP-C4 routed via G4 because the *evidence channel* differs: G5 surfaces the mean-flux ↔ predict_proba leak directly, where G4 surfaces it via the C2/C3 confusion floor. Both belong in the C4-confounder family but the diagnostic chain is split — paper text must report which channel triggered. |
| **PP-MIS** | Partial PASS — mechanism mis-anchored | Either (i) G3 fails (signal exists but not in the mid-k band Khaire+2024 / Tillman+2024 predict), OR (ii) G5 fails with a physics-class violator (mean-flux contamination on C1/C2/C3 confirms a leak channel beyond C4). The classification works but the mid-k line-width mechanism story is falsified. |
| **PP-C4+C1** | Partial PASS — C4+C1-carried, C2/C3 null (per [D-23] C3) | Specifically gated on G4 FAIL with G1 PASS: the 4-class headline lift survives but the C2/C3 distinction (the actual stellar-wind-vs-wind+AGN feedback comparison) is at-or-below chance. Headline accuracy is honest but **not** "feedback identification" — it's "the strongest-AGN class is identifiable, C1 is identifiable as 'something is happening', and the two finer feedback variants are indistinguishable." |
| **PP-DEC** | Partial PASS — decorative mid-k (new band; see §4) | G6 fails alone with everything else passing. The mid-k peak is RF feature-importance-bias on adjacent log-bins (Strobl et al. 2007), not a load-bearing channel — but the headline accuracy and the C2/C3 distinction both survive on the next-best channels. Treated as a sub-case of PP-MIS in the [D-09] band set; broken out here for clarity. Justification in §4. |
| **NULL** | NULL | Cascade-exhaustion close per [D-20] S6 — at least one of {G1, G4} fails AND no honest partial-PASS framing survives the remaining gates. NO Stage 2 retry; falsified-prior cascade applies; the result folds into the eventual [D-13] paper (user-triggered) as a second null. |
| **AMB** | Ambiguous (default-to-NULL absent panel re-review per rule 14) | Joint-gate signature that does not fit any of the above clearly. Defaults to NULL operationally; recorded as Ambiguous so the eventual paper-author trigger can re-pose to the panel. |

**Hard constraints from [D-23] (any cell-routing violating these is a spec-violation):**

- **From C3:** any cell with **G4 = FAIL** routes to PP-C4+C1 / PP-MIS / NULL / AMB — **NEVER CP**, regardless of G1/G2/G3/G5/G6.
- **From C7:** G5 FAIL with `C4_only_violator=true` → PP-C4G5 band, distinct from PP-MIS. G5 FAIL with any of C1/C2/C3 violating → PP-MIS or NULL per joint G3/G4.
- **From C1:** G6 threshold is `max(5pp, 2 × empirical_SD)` — data-derived. The 64-cell table treats G6 binary (PASS/FAIL) based on that resolved threshold.

---

## 3. 64-cell PASS/FAIL routing table

Cell ID encodes (G1, G2, G3, G4, G5, G6) as a 6-bit string, MSB=G1, LSB=G6. `P` = PASS, `F` = FAIL. The `G5_C4_only` column is `N/A` when G5=PASS, and `true|false` when G5=FAIL — the routing in §2 binds it.

Cells marked with **★** are PI-flagged load-bearing cells (called out explicitly in §4 below).

| Cell | G1 | G2 | G3 | G4 | G5 | G6 | Route (G5_C4_only=true) | Route (G5_C4_only=false) | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| 1 ★ | P | P | P | P | P | P | **CP** (n/a) | n/a | All-PASS — Clear PASS. The Stage 1 headline-positive cell. |
| 2 ★ | P | P | P | P | P | F | PP-DEC (n/a) | n/a | G6-only fail = decorative mid-k. See §4. |
| 3 | P | P | P | P | F | P | PP-C4G5 | PP-MIS | G5-only fail; C7 split-band binds. |
| 4 | P | P | P | P | F | F | PP-C4G5 | PP-MIS | G5+G6 fail. PP-C4G5 if C4-only (G6 decorative on a contaminated signal); else PP-MIS. |
| 5 ★ | P | P | P | F | P | P | **PP-C4+C1** (n/a) | n/a | G4 fail = C2/C3 null. C3-binding: NEVER CP. The "headline survives but feedback-question fails" cell. |
| 6 | P | P | P | F | P | F | PP-C4+C1 (n/a) | n/a | Add G6 fail: mid-k is decorative AND C2/C3 null. Headline weak. |
| 7 | P | P | P | F | F | P | PP-C4G5 | NULL | G4+G5 fail; if physics-class violator → NULL (every gate the C2/C3 question depended on failed); if C4-only the headline is still C4-driven so PP-C4G5. |
| 8 | P | P | P | F | F | F | PP-C4G5 | NULL | As 7 plus G6 fail. Routing unchanged: C4-only keeps PP-C4G5 frame, physics-class violator → NULL. |
| 9 | P | P | F | P | P | P | PP-MIS (n/a) | n/a | G3 fail = signal exists but not in mid-k band. Mechanism mis-anchored. |
| 10 | P | P | F | P | P | F | PP-MIS (n/a) | n/a | G3+G6 fail. Mechanism mis-anchored; G6 fail is on a band the run already says isn't the channel — consistent. |
| 11 | P | P | F | P | F | P | PP-C4G5 | PP-MIS | G3+G5 fail; band wrong AND mean-flux leak. If physics-class violator clearly PP-MIS. |
| 12 | P | P | F | P | F | F | PP-C4G5 | PP-MIS | G3+G5+G6 fail. Same routing as 11. |
| 13 | P | P | F | F | P | P | PP-C4+C1 (n/a) | n/a | G3+G4 fail; C3 binding (G4 fail → never CP). C4+C1 carried headline. |
| 14 | P | P | F | F | P | F | PP-C4+C1 (n/a) | n/a | Add G6 fail; same family as 13. |
| 15 | P | P | F | F | F | P | PP-C4G5 | NULL | G3+G4+G5 fail; physics-class violator → NULL. |
| 16 | P | P | F | F | F | F | NULL (n/a) | NULL | G3+G4+G5+G6 fail; C4-only G5 still leaves the C2/C3 question null per G4. NULL. |
| 17 | P | F | P | P | P | P | AMB | n/a | G2 fail with rest PASS is structurally weird (G2 = "any label structure exists" — failing it while G1 passes contradicts the null-hypothesis test). Route to Ambiguous → re-review. |
| 18 | P | F | P | P | P | F | AMB | n/a | As 17 + G6 decorative. Ambiguous. |
| 19 | P | F | P | P | F | P | AMB | AMB | G2 inconsistent with G1; G5 fail adds; ambiguous → panel re-review. |
| 20 | P | F | P | P | F | F | AMB | AMB | As 19 + G6. Ambiguous. |
| 21 | P | F | P | F | P | P | PP-C4+C1 (n/a) | n/a | C3 binding (G4 fail → not CP); G2 fail noted but G4 binding dominates. |
| 22 | P | F | P | F | P | F | PP-C4+C1 (n/a) | n/a | As 21 + G6. |
| 23 | P | F | P | F | F | P | PP-C4G5 | NULL | As 7 + G2 fail; if physics-class violator clearly NULL. |
| 24 | P | F | P | F | F | F | NULL (n/a) | NULL | All scientific gates failed except G1 headline; C2/C3 question null. |
| 25 | P | F | F | P | P | P | AMB | n/a | G2+G3 fail with G1 PASS = G1 driven by something other than mid-k AND not consistent with label structure under permutation. Internal contradiction → Ambiguous. |
| 26 | P | F | F | P | P | F | AMB | n/a | As 25 + G6. |
| 27 | P | F | F | P | F | P | PP-MIS | PP-MIS | G2+G3+G5 fail; mis-anchored regardless of C4-only (G3 already says the band itself is wrong). |
| 28 | P | F | F | P | F | F | PP-MIS | PP-MIS | As 27 + G6. |
| 29 | P | F | F | F | P | P | PP-C4+C1 (n/a) | n/a | C3 binding. |
| 30 | P | F | F | F | P | F | PP-C4+C1 (n/a) | n/a | As 29 + G6. |
| 31 | P | F | F | F | F | P | NULL (n/a) | NULL | G2+G3+G4+G5 fail; cascade close. |
| 32 ★ | P | F | F | F | F | F | NULL (n/a) | NULL | Only G1 passes — accuracy without any anchoring mechanism gate. Cascade-exhaustion. |
| 33 | F | P | P | P | P | P | AMB | n/a | G1 fail with rest PASS = accuracy below 0.65 lower-CI but per-class structure, G3, G4 all clean. The bar may be too high for the signal size — Ambiguous → panel review on the bar. |
| 34 | F | P | P | P | P | F | AMB | n/a | As 33 + G6 decorative. |
| 35 | F | P | P | P | F | P | AMB | AMB | G1+G5 fail; the C2/C3 distinction passes G4 but G1 doesn't clear; below-bar with mean-flux leak. Ambiguous. |
| 36 | F | P | P | P | F | F | AMB | AMB | As 35 + G6. |
| 37 | F | P | P | F | P | P | NULL (n/a) | NULL | G1+G4 fail = no headline AND no C2/C3 distinction. Cascade close. |
| 38 | F | P | P | F | P | F | NULL (n/a) | NULL | As 37 + G6. |
| 39 | F | P | P | F | F | P | NULL | NULL | G1+G4+G5 fail. Cascade. |
| 40 | F | P | P | F | F | F | NULL | NULL | As 39 + G6. |
| 41 | F | P | F | P | P | P | NULL (n/a) | NULL | G1+G3 fail; even with G4 PASS, headline fails and signal isn't in the predicted band. |
| 42 | F | P | F | P | P | F | NULL (n/a) | NULL | As 41 + G6. |
| 43 | F | P | F | P | F | P | NULL | NULL | G1+G3+G5 fail. |
| 44 | F | P | F | P | F | F | NULL | NULL | As 43 + G6. |
| 45 | F | P | F | F | P | P | NULL (n/a) | NULL | G1+G3+G4 fail. |
| 46 | F | P | F | F | P | F | NULL (n/a) | NULL | As 45 + G6. |
| 47 | F | P | F | F | F | P | NULL | NULL | G1+G3+G4+G5 fail. |
| 48 | F | P | F | F | F | F | NULL | NULL | All scientific gates fail; G2 alone passes (label structure exists at all = trivially true). NULL. |
| 49 | F | F | P | P | P | P | NULL (n/a) | NULL | G1+G2 fail = below-bar AND fails permutation null = signal indistinguishable from chance label structure. NULL. |
| 50 | F | F | P | P | P | F | NULL (n/a) | NULL | As 49 + G6. |
| 51 | F | F | P | P | F | P | NULL | NULL | G1+G2+G5 fail. NULL. |
| 52 | F | F | P | P | F | F | NULL | NULL | As 51 + G6. |
| 53 | F | F | P | F | P | P | NULL (n/a) | NULL | G1+G2+G4 fail. NULL. |
| 54 | F | F | P | F | P | F | NULL (n/a) | NULL | As 53 + G6. |
| 55 | F | F | P | F | F | P | NULL | NULL | G1+G2+G4+G5 fail. |
| 56 | F | F | P | F | F | F | NULL | NULL | As 55 + G6. |
| 57 | F | F | F | P | P | P | NULL (n/a) | NULL | G1+G2+G3 fail. |
| 58 | F | F | F | P | P | F | NULL (n/a) | NULL | As 57 + G6. |
| 59 | F | F | F | P | F | P | NULL | NULL | G1+G2+G3+G5 fail. |
| 60 | F | F | F | P | F | F | NULL | NULL | As 59 + G6. |
| 61 | F | F | F | F | P | P | NULL (n/a) | NULL | G1+G2+G3+G4 fail. |
| 62 | F | F | F | F | P | F | NULL (n/a) | NULL | As 61 + G6. |
| 63 | F | F | F | F | F | P | NULL | NULL | G1+G2+G3+G4+G5 fail. |
| 64 ★ | F | F | F | F | F | F | NULL (n/a) | NULL | **All-FAIL** — cascade-exhaustion close per [D-20] S6. Kill-or-pivot fork to PI; track is closed (NO Stage 2 retry on a third candidate). |

**Cell-count distribution by route:**
- CP: 1 cell
- PP-DEC: 1 cell (cell 2 — broken out per §4 below; absorbs into PP-MIS in [D-09]'s native band set if PI prefers)
- PP-C4G5 / PP-MIS: 16 routing decisions in 8 cells (G5-fail branches)
- PP-C4+C1: 8 cells (G1 PASS + G4 FAIL branches)
- AMB: 8 cells (internally contradictory G1/G2 combinations that warrant panel re-review)
- NULL: 38 cells (the cascade-exhaustion + scientific-question-null majority)

---

## 4. Load-bearing cells (PI explicitly cares about these)

### Cell 1 — All-PASS → CP
Routes to **Clear PASS**. The Stage 1 headline-positive outcome. Deliverable surface per [D-17] is "stacked-P(k) population-statistics adjacent finding, first multi-class classification test of Khaire+2024 / Tillman+2024 scale-dependent low-z P_F(k) feedback signal at z ≈ 0.3 extrapolated from z = 0.1." Verb hedges from [D-15] / [D-20] S1 stack on every claim.

### Cell 5 — G4-only fail → PP-C4+C1
Per [D-23] C3: G4 = FAIL routes here regardless of how the other five gates land. The headline 4-class accuracy survives, but the C2/C3 stellar-wind-vs-wind+AGN distinction (the actual feedback-physics-relevant comparison) is null. Honest framing per [D-37]: this is **not** "feedback identification" — it is "C4 + C1 separable from each other and from {C2,C3}-pooled; the feedback gradient inside {stellar-wind, wind+AGN} is not resolved at z = 0.3 by this stacked basis."

### Cell 3 — G5-only fail → split via `G5_C4_only`
- `G5_C4_only = true` → **PP-C4G5** (per [D-23] C7 (ii)) — distinct from PP-MIS. The mid-k signal on {C1, C2, C3} is still trustworthy (G3 PASS, G4 PASS); only C4 is mean-flux-confounded. C4 \|r\| value reported on the face of the results table per [D-23] C7.
- `G5_C4_only = false` (any of C1/C2/C3 has \|r\| ≥ 0.3) → **PP-MIS** per [D-23] C7 (iii). The mean-flux leak reaches the physics-relevant classes; the mechanism interpretation is contaminated.

### Cell 2 — G6-only fail → PP-DEC (new band; absorbs into PP-MIS optionally)
**PI decision needed.** Two routings on the table:
- **(A) Break out as a new band PP-DEC ("decorative mid-k").** Justification: G6 fail with G1+G2+G3+G4+G5 all PASS is a *specific* diagnostic claim — the mid-k peak is RF importance-bias on adjacent log-binned k-channels (Strobl et al. 2007), not a load-bearing channel — but the headline accuracy, the C2/C3 distinction, the label structure, AND the mean-flux discipline all survive. The classification problem is solvable; the mechanism band identified by the importance map is decorative. This is its own physical interpretation, not "mis-anchored." The [D-09] band set as written has no slot for this — adding PP-DEC is honest per [D-37].
- **(B) Fold into PP-MIS.** Justification: from a paper-text perspective, "the mid-k mechanism story is falsified" is the same downstream consequence as G3 fail — the headline survives but the mid-k narrative does not. Avoids band proliferation.

**Recommendation: (A), with the understanding that PP-DEC paper text reads "the classification signal exists in the stacked-P(k) basis but the RF importance localization to k ∈ [0.04, 0.15] s/km is not load-bearing — the signal is carried by the broader k-distribution, and the mid-k peak is an importance-bias artifact on adjacent log-bins."** This is a strictly stronger / more informative null on the mid-k claim than PP-MIS, and the diagnostic chain (G6 falsifies, G1/G3 survive) is distinct enough to warrant the separate label. If PI prefers (B), routing for cell 2 becomes PP-MIS and PP-DEC drops from the band set.

### Cell 32 — Only G1 PASS, all other gates fail → NULL
Headline accuracy without mechanism, without label-structure validation, without C2/C3 distinction, without mean-flux discipline, without ablation-survival. Cascade-exhaustion despite a passing headline. The cleanest case where a naive reading of "G1 cleared, we PASSED" would be wrong — every supporting gate said the lift is non-mechanism. Per [D-37] the honest framing is **NULL** with the headline number explicitly NOT cited as the deliverable.

### Cell 64 — All-FAIL → NULL (cascade-exhaustion close per [D-20] S6)
The track is closed. NO Stage 2 retry on a third candidate basis. Falsified-prior cascade applies (rule 2): the prior under test in this track — *"the P_F(k) basis breaks the [D-13] ceiling"* — is falsified at this end state. Next decision surfaces to PI as a kill-or-pivot fork.

---

## 5. Open questions for PI APPROVE

1. **PP-DEC band (cell 2):** create new band, or fold into PP-MIS? Recommendation: create PP-DEC. (§4 cell 2 above.)
2. **AMB cells (17–20, 25–26, 33–36):** the "G1 PASS, G2 FAIL" subset is structurally weird — accuracy clears the lower-CI bar but the permutation null says "no label structure." Operationally these default to NULL per [D-09] (Ambiguous → default-to-NULL absent panel re-review). The routing-table marks them AMB to force a panel touch on the eventual posterior. Is that intended, or should they directly route to NULL? Recommendation: keep as AMB so the eventual paper-author trigger (user-triggered) requires panel re-review.
3. **G3 + G4 fail with G1 PASS (cells 13–14):** routed to PP-C4+C1. Alternative routing as PP-MIS. Recommendation: PP-C4+C1 — the C2/C3-null framing is more diagnostic than "mechanism mis-anchored" because the latter doesn't specify *which* class structure carries the lift.
4. **Cell 33 (G1 fail with everything else clean):** routed to AMB. Alternative: NULL (G1 is the headline gate; if it fails, the rest are accessory). Recommendation: AMB — the configuration (G2/G3/G4/G5/G6 all clean, only G1 misses the lower-CI bar) is rare enough that a panel touch is warranted before defaulting to NULL.

---

## 6. PI APPROVE record

**PI APPROVAL required before infrastructure-manager fires sbatch (per [D-23] C5).**

- [ ] PI signs off on the routing table as drafted, OR PI authors targeted edits + signs off.
- [ ] On APPROVE, this document is referenced by a new D-XX entry appended to LEDGER §3 (per [D-23] C5 — append-only, not edited into [D-20] S2).
- [ ] On APPROVE, infrastructure-manager is unblocked on C5 (C6 unit-test green is verified independently — see `tests/test_stage1_fold_leakage.py`).
- [ ] On APPROVE, `run_stage1.py` is unchanged — the routing logic lives in this spec doc, not in the run script. The run script emits the raw G1–G6 verdicts; the spec doc's table maps verdict-tuples to bands; the eventual paper-author trigger consumes both.
