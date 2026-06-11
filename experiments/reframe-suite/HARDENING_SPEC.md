# HARDENING_SPEC — reframe-suite defense-panel neutralization sprint

**Track:** reframe-suite hardening (user-authorized 2026-06-10; lifts the park gate for exactly this scope)
**Branch:** `exp/pk-feedback-classifier` · **Spec file:** `experiments/reframe-suite/HARDENING_SPEC.md`
**Authorization record:** user invocation 2026-06-10 ("Continue with PI agent") following defense-panel verdict NEEDS WORK (durable record: `.claude-memory/diagnosis-2026-06-10-defense-panel.md`). Paper-author remains NOT dispatched — this sprint is not a paper trigger.
**Pre-commitment:** the git commit introducing this file predates every hardening run; the H-gates in §5 are committed before any compute. This is the first genuinely pre-committed gate set in the reframe-suite line.
**Sign-off provenance:** PI-only **PROVISIONAL** per rule 15 — this sprint is *promotion-direction* (it decides whether the R1/R5 headlines survive into any future paper). The rule-15 narrowing exemption invoked in `CLOSE_OUT.md` §5 does **not** apply here, and this spec says so explicitly. Provisional status is lifted only by defense-panel re-review APPROVE on the empirical returns (§8). All gate verdicts authored under this spec inherit PROVISIONAL until that APPROVE.

## §0 — Prior-failure ledger (rule 4) and verb ceiling (rule 2)

Falsified priors of similar confidence in this track: [D-01] high-k thermal-cutoff (falsified, [D-07]); [D-10]/[D-15] mid-k mechanism (falsified, G3+G6 FAIL, [D-26]); signal-clustering-v2 [D-13] per-sightline separability (falsified). Additionally the defense panel has shown that three *claims* in the close-out stack were over-verbed: "intrinsic C2/C3 null" (contradicted by our own `reframe5_lattice.csv` M-trend), "triply-replicated" (replication inflation), and every lattice number's estimator (4-class sub-block, not binary). The inheriting verb level for this sprint is therefore maximally hedged: **every item below is a first test of its question; no outcome direction is presumed.** Pre-committed honest summary if everything KILLS: "the lattice was mean-flux ordering measured with an uncalibrated estimator" is a valid, reportable end state (rule 7).

Anti-degeneracy audit (rule 3), sprint-level: the shared unconstrained axis across items (i)–(iii) is that all measurements live on **one realization of one simulation suite at fixed cosmology/UVB/thermal history**; no item below can constrain realization variance or nuisance-parameter degeneracy. This conditioning enters the binding non-claim list (item iv, edit E7) regardless of outcomes.

## §1 — Track placement

**Decision: this sprint lives at `experiments/reframe-suite/` on the existing `exp/pk-feedback-classifier` branch.** New scripts under `experiments/reframe-suite/scripts/hardening_*.py`; results under `results/reframe_suite/hardening/`; figures under `results/reframe_suite/hardening/figs/`. Decision-log entries continue the pk-feedback-classifier numbering in `experiments/pk-feedback-classifier/LEDGER.md` §3 at **[D-27]–[D-30]** (verified [D-26] is the current maximum).

Justification (judgment call): the experiment-isolation rule mandates a new branch for new *methodologies*; this sprint introduces no new methodology — it re-measures existing claims with defensible estimators on the same data, protocol, and Juno wiring already resident on this branch. The reframe-suite precedent (track-internal docs on this branch, decisions in the pk LEDGER) is followed. A new branch would orphan the `dvc.lock`/data lineage the items depend on.

## §2 — Item (i): dedicated binary pair classifiers (lattice v2)

**Question (hedged):** first test of whether the R5 lattice and R1 detection numbers survive measurement with a calibrated binary estimator, and first dedicated measurement of the C2-C3 M-curve.

**Scope decision: all 6 pairs + a 7th task (binary C4-vs-rest), not C2-C3 alone.** The panel's attack 1 applies to every lattice cell *and* to the R1 headline (0.971/0.960 is also a confusion-derived number); the marginal cost is small; symmetric disclosure (rule 5) wants the full lattice re-measured whatever the outcome. The dedicated-binary numbers become the **numbers of record** for every cell; `pair_balacc_from_cm` outputs are demoted to "derived sub-block estimates, superseded" (resolves the three-inconsistent-numbers attack by fiat: one estimator, one number per cell).

**Estimator (one and only one):** `RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)` trained on the 20-bin stacked P_F(k) features of the two classes only (binary labels); metric `sklearn.metrics.balanced_accuracy_score` on the held-out fold. No SVM arm, no hyperparameter search — identical to the Stage-1 RF so numbers are protocol-comparable.

**Data + protocol (must reuse Stage-1 code paths, not re-implement):** flux from `data/preprocessed/Sherwood_z0.3_inf/{1..4}/flux.npy`; P_F(k) via `src/core/transforms.FluxPowerSpectrum` exactly as in `experiments/pk-feedback-classifier/run_stage1.py`; stacking via `_stack_pk_within_class` with the identical SEEDS = [42..51] so stack memberships are bit-identical to Stage 1 and to item (ii); 5-fold `StratifiedKFold(shuffle=True, random_state=seed)` over post-stacking group indices, membership built once per (M, seed) and reused across folds ([D-20] S3); source-sightline train/test disjointness assertion inherited.

**Grid:** regimes {per_sightline, global} (regime-parity check inherited from [D-07]); M ∈ {1, 4, 16, 32, 64, 128, 256} (the *actual* existing grid); 10 seeds × 5 folds. **Saturation extension:** M ∈ {512, 1024} for the C2-C3 pair only, per_sightline regime only, flagged `exploratory_high_variance` (32768/M pair-groups = 64 and 32 total; CI dominates per panel P2 precedent) — **not gate-bearing**.

**Outputs:** `results/reframe_suite/hardening/pair_binary_cv.csv`, append-as-completed per row ([D-20] S5 checkpointing), schema:
`regime,M,pair,seed,fold,n_train_groups,n_test_groups,balanced_acc,exploratory_flag,runtime_sec`
(pair ∈ {C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4, C4-rest}). Aggregate `pair_binary_summary.csv`: `regime,M,pair,n,median,p16,p84`.

**Anti-degeneracy line item (rule 3):** a binary RF on 20 correlated log-bins still leaves *which-k-band* unconstrained; this sprint does not re-litigate band attribution (the mid-k mechanism is already falsified, [D-26]) and no band claim may be derived from item (i) importances.

## §3 — Item (ii): ⟨F⟩-scalar baseline (cheapest-decisive)

**Question (hedged):** first test of whether the lattice is mean-flux ordering in disguise (panel attack 4: Spearman ≈ 0.94 between lattice p16 ranking and per-class |Δ⟨F⟩| from [D-02]).

**Estimator (one and only one per claim):** same `RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)`, same CV protocol, same stack memberships (same `_stack_pk_within_class` groups; reuse the existing per-group ⟨F⟩ helper at `run_stage1.py:225`). **Primary feature set: stack-level ⟨F⟩ only (1-dim), computed from raw flux F (pre-normalization — the per_sightline δ_F regime destroys ⟨F⟩ by construction, so the baseline is regime-independent and recorded with `regime=raw_flux`).** Secondary feature set (disclosed, non-gate-bearing): [⟨F⟩, var(F)] 2-dim. Tasks: same 7 (6 pairs + C4-vs-rest) plus the 4-class problem for completeness. Same M-grid {1,...,256}, 10 seeds, 5 folds.

**Outputs:** `results/reframe_suite/hardening/fbar_baseline_cv.csv` schema `M,task,feature_set,seed,fold,n_train_groups,n_test_groups,balanced_acc,runtime_sec`; aggregate `fbar_baseline_summary.csv` (median/p16/p84 per M × task × feature_set).

**Honest dependency note:** item (ii) alone is necessary but not sufficient for the decisive comparison — the H3 gate compares this baseline against item (i)'s dedicated-binary P_F(k) numbers (the sub-block numbers are not valid comparators). Running (ii) first validates the harness and gives the baseline column; the verdict waits for (i).

**Pre-registered tension:** R9 (CLOSE_OUT §1) showed C4 detection robust to ⟨F⟩-removal, which *predicts* the baseline will undershoot P_F(k) — but R9's two removal controls were degenerate (⟨F⟩ ≈ 0.97 ≈ 1 caveat). If H3 instead finds baseline ≈ P_F(k) at the C4 cell, the pre-committed resolution verb is "⟨F⟩ and P_F(k) shape are redundant encodings of the same separability; P_F(k) is not uniquely load-bearing" — not a contradiction to be argued away.

## §4 — Item (iii): injection-recovery sensitivity curve

**Question (hedged):** first calibration of the pipeline's minimum detectable P_F(k)-shape effect, converting the vacated "methodology eliminated" verb into "sensitive down to δ_min(M)".

**Design.** Template: ΔP(k) = P̄_C4(k) − P̄_C2(k), per k-bin, full-sample means of per-sightline P_F(k) in the per_sightline regime (template definition uses C4 data; the classifier never sees C4 — no leakage). Surrogate class C2′: per-sightline P(k) + α·ΔP(k), injected **before stacking** (preserves the variance-vs-M structure), then standard stacking. Train the item-(i) estimator on C2 vs C2′ under the identical CV protocol. **Calibration floor:** α = 0 must return balanced_acc within 0.50 ± 0.05 (p16–p84 spanning 0.5); if not, the instrument fails validation (see H4 routing).

**Grids:** α ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}; M ∈ {64, 256} (the two report points); 10 seeds × 5 folds → 800 binary fits on ≤512-group inputs (small).

**Effect-size mapping (pre-committed):** β = ⟨ΔP_{C3−C2}, ΔP_{C4−C2}⟩ / ‖ΔP_{C4−C2}‖² (projection coefficient of the actual C3−C2 difference onto the injection template), with both ΔP curves emitted for the figure. **δ_min(M)** = smallest α with balanced_acc p16 ≥ 0.60 (threshold matches the G4 bar precedent; pre-committed here, before any run). **α_equiv(M)** = inverse-read of the observed dedicated C2-C3 balanced_acc (item i) through the recovery curve.

**Pre-committed scope caveat (rule 3 / [D-37]):** injection is in P(k) space, not flux space — it cannot capture flux-space nonlinearity or phase structure. The resulting claim is strictly *"pipeline sensitivity to P_F(k)-shape perturbations of amplitude α·ΔP_{C4−C2}"*, never "sensitivity to feedback physics of strength α." This caveat appears verbatim in any report of δ_min.

**Outputs:** `results/reframe_suite/hardening/injection_recovery_cv.csv` (`M,alpha,seed,fold,n_train_groups,n_test_groups,balanced_acc,runtime_sec`); `injection_recovery_summary.csv`; `injection_templates.csv` (`k_s_per_km,deltaP_C4_C2,deltaP_C3_C2,beta`); figure `figs/injection_sensitivity_curve.png` (balanced_acc vs α at M=64/256, δ_min and β annotated, caption self-sufficient per the caption rule).

## §5 — Pre-committed gates (numeric, symmetric, committed before any run)

All gate-bearing reads are at **(per_sightline, M=64)** on **p16** unless stated, matching the SCOPING §4 read points. Re-verbed honestly per panel finding 7: these are the *qualification thresholds of record for the hardening sprint*, committed in git before any hardening run (this commit).

**H1 — lattice v2 integrity (item i).**
- Per cell: |dedicated p16 − sub-block p16| ≤ 0.05 → cell CONFIRMED; dedicated p16 < sub-block p16 − 0.05 → cell number of record drops to the dedicated value.
- **R5 survives** iff, recomputed entirely with dedicated-binary numbers: ≥ 3 of 5 non-(C2-C3) pairs at p16 ≥ 0.75 AND the SCOPING §4 structural check holds. **KILL:** < 3 cleared → CLOSE_OUT §1 R5 verdict "QUALIFIES" is rescinded by dated amendment (not deletion); the lattice leaves the headline set; paper shape (if ever triggered) demotes to C4-detection-only or methods-only.
- **R1 survives** iff dedicated binary C4-vs-rest p16 ≥ 0.90. **KILL:** p16 < 0.90 → R1 verdict rescinded; the project loses its only positive headline; escalate to user (paper-shape decisions are user-owned).
- **HARDENS:** both survive → headlines re-anchored on the dedicated numbers (which replace 0.960/0.978/etc. everywhere).

**H2 — C2-C3 M-conditional exclusion bound (item i). Pre-committed reporting format** (used verbatim regardless of values): *"Under a dedicated binary RF on stacked P_F(k) (Stage-1 protocol), C2-C3 balanced accuracy is consistent with chance for M ≤ M_null and rises with stacking depth; M_null = largest gate-bearing M with p16 ≤ 0.55 and median ≤ 0.60. At M=256: p16 = [x], median = [y]. This is an M-conditional exclusion bound, not an intrinsic degeneracy floor."* Routing:
- Dedicated p16 at M=256 ≥ 0.60 → the C2/C3-null verb is **retired project-wide**; the cascade-close framing partially REOPENS ("the population-level C2/C3 question is not closed"); mandatory defense-panel review + user escalation before any further framing decision.
- Dedicated p16 at M=256 < 0.60 but median trend rising → exclusion-bound format above, full curve published, no "floor" verb anywhere.
- M ∈ {512, 1024} exploratory values are reported but cannot move M_null or trigger routing.

**H3 — ⟨F⟩-baseline headroom (items ii + i jointly).** Per gate-bearing cell, Δ_cell = dedicated_PFk_p16 − fbar_baseline_p16 (⟨F⟩-only primary feature set):
- **HARDENS (cell):** Δ ≥ 0.05 → "P_F(k) carries information beyond stack-mean flux" admissible for that cell.
- **KILLS (cell):** Δ < 0.02 → cell re-verbed "consistent with mean-flux ordering alone."
- 0.02 ≤ Δ < 0.05 → AMBIGUOUS, defaults to the weaker verb (rule-14 default-to-null precedent).
- **KILLS (headline):** if ≥ 3 of the 4 R5-qualifying cells OR the C4-vs-rest cell land in the KILL band → the lattice/detection headline is re-verbed "a stack-mean-flux separability lattice"; the Spearman-0.94 attack is confirmed; the P_F(k)-methodology contribution is withdrawn from any future paper shape. Symmetrically: if all 5 land in HARDENS, the Spearman-0.94 coincidence is answered and recorded as such.

**H4 — sensitivity instrument (item iii).**
- Instrument valid iff the α=0 floor calibrates (0.50 ± 0.05 on p16–p84). **If invalid:** report instrument failure; no sensitivity claim; "methodology eliminated" remains simply *vacated* (by item iv) with no quantitative replacement — a valid rule-7 end state.
- If valid: report δ_min(M=64), δ_min(M=256), β, α_equiv(M=256). Consistency check (pre-committed): α_equiv should be ≈ β if the C3−C2 difference is template-shaped; a large mismatch is reported as "the C2−C3 difference is not C4-template-shaped," not smoothed over.
- No KILL direction exists for H4 beyond instrument failure — item (iii) is an instrument, not a claim promotion; its output may not be verbed as a headline (rule 13 scope-lock: instrument → instrument).

## §6 — Item (iv): re-verb checklist (docs-only; two passes)

**Pass 1 (immediately, before any compute — uses only on-disk numbers):**

- **E1** `experiments/reframe-suite/CLOSE_OUT.md` §1 R5 table, C2-C3 row: replace "0.501 ≈ chance (excluded — cascade-anchored null)" with "0.501 p16 at M=64; climbs monotonically with M to p16 0.599 / median 0.678 at M=256 (`results/reframe_suite/phase1/reframe5_lattice.csv`); M-conditional exclusion bound, dedicated-binary re-measurement pending (HARDENING item i)". Same-file "Substantively new finding" paragraph: "the C2-C3 pair alone is null" → "C2-C3 is consistent with chance at M ≤ 64 and rises with stacking depth".
- **E2** `experiments/reframe-suite/CLOSE_OUT.md` §2 combined-narrative blockquote: "stellar-wind-vs-wind+AGN null at chance" → "stellar-wind-vs-wind+AGN consistent with chance at M ≤ 64, rising toward 0.60–0.68 by M=256 (M-conditional exclusion bound, not an intrinsic floor)".
- **E3** Replication re-count. `experiments/reframe-suite/CLOSE_OUT.md` §2 and `experiments/linewidth-cddf/CLOSE_OUT.md` §4 rule-8 text + §6 learning 2: re-verb "three specific interventions ... produced congruent C2/C3-null signatures" → "two correlated empirical nulls on the same Sherwood τ/flux realization plus one scoping close that produced no measurement, externally anchored by Nasir+2017 — replication count: 2 (correlated) empirical + 1 literature anchor". Also soften "The C2-C3 distinguishability claim is closed." → "closed at the tested protocols and stacking depths (see HARDENING H2 for the M-conditional bound)".
- **E4** `experiments/linewidth-cddf/CLOSE_OUT.md` §3 (i) (Δb 1.8 km/s vs 2.5 km/s bin width): append "— this bin-width comparison is directional, not a statistical-power argument; no detectability forecast at our sample size was computed."
- **E5** Post-hoc gates disclosure. `experiments/reframe-suite/SCOPING.md` §4 preamble and `CLOSE_OUT.md` §5 rule-5 bullet: annotate "pre-committed" → "post-hoc qualification thresholds: authored after the Stage-1 confusion matrices were on disk; pre-committed only relative to the Phase-1 re-aggregation run (commit `e090b52`)".
- **E6** `experiments/reframe-suite/PHASE2_REPORT.md` "C2-vs-C3 null at chance (~0.55)" → M-conditional phrasing per E1.
- **E7** Fixed-nuisance non-claim. Append to the binding non-claim list in `experiments/pk-feedback-classifier/LEDGER.md` [D-17] and to `CLOSE_OUT.md` §2: "All lattice/detection numbers are conditional on fixed cosmology, UVB, and thermal history within one Sherwood realization; no nuisance marginalization was performed."
- **E8** Novelty log: add arXiv:2509.18260 (AGN feedback variations on Lyα P1D) to the [D-30] entry + `experiments/reframe-suite/FIGURE_LIST.md` notes, flagged "must-cite + novelty-delta check required before any paper trigger". (Sinigaglia+2026 z-range note carried alongside.)
- **E9** `experiments/pk-feedback-classifier/LEDGER.md` §1 Stage-1 row + [D-26] milestone: append dated annotation (LEDGER entries are append-only — annotate, never rewrite): "2026-06-10: 'C2/C3 null' refers to the 4-class sub-block estimator at the gate-read M; superseded by the HARDENING item-(i) dedicated-binary M-conditional bound on completion."

**Pass 2 (after item (i) returns):** substitute dedicated-binary numbers of record into E1/E2/E6 placeholders; execute H1/H2/H3 routing edits per §5.

All edits are dated amendments; no historical text is deleted.

## §7 — Shared-ICs pin (no compute)

Dispatch a **web-enabled general-purpose agent** ([D-14] precedent: the support-researcher tool list lacks web tools) to verify against Bolton et al. 2017 (MNRAS 464, 897, arXiv:1605.03462 — the Sherwood suite paper) and Nasir+2017 §2: **do the four feedback variants share identical initial conditions (same box, same random seed)?** Record the answer, with the quoted sentence and source, in `experiments/pk-feedback-classifier/LEDGER.md` §2 + [D-30], and as an annotation in `CLOSE_OUT.md` §2. Pre-committed consequences, symmetric: (a) **shared ICs** → the two empirical nulls are same-realization-correlated (E3 count stands); the per-position cross-class pairing used by the signal-clustering-v2 probe and Reframe 7 is justified; (b) **not shared** → escalate: the per-position pairing assumption in the probe design acquires an audit flag (affects signal-clustering-v2 [D-13] interpretation), and E3 gains "realizations independent" in the replication wording. If the literature is ambiguous, record "UNRESOLVED — treated as shared (conservative for replication counting)" and keep the flag open.

## §8 — Compute routing, budget, ordering

| Step | Item | Target | Est. wall | Cap | Cost |
|---|---|---|---|---|---|
| 1 | (iv) pass 1 + §7 ICs pin | orchestrator / web agent | <1 h | — | $0 |
| 2 | (ii) ⟨F⟩ baseline | laptop | 0.5–1.5 h (M=1 dominates) | 2 h | $0 |
| 3 | (i) smoke (M=64, seed=42, all 7 tasks, both regimes) | laptop | ~10 min | — | $0 |
| 4 | (i) full sweep | **Juno** CPU partition, 1 node × 16 cores × 24 GB | 1.5–3 h (anchor: Stage-1 full = 52m31s for ~1400 4-class fits; this is ~4400 cheaper binary fits) | 8 h wallclock | $0 marginal (Fairshare) |
| 5 | (iii) injection | laptop | 1–2 h incl. per-sightline P(k) recompute | 3 h | $0 |
| 6 | (iv) pass 2 + H-gate verdicts | orchestrator/PI | <1 h | — | $0 |
| 7 | defense-panel re-review | panel | — | — | $0 |

Economic-compute completeness for step 4: (a) instance = Juno CPU partition as in [D-05]; (b) ≤ 8 h; (c)/(d) $0 marginal, Fairshare-queued; (e) auto-stop = SLURM wallclock + `set -euo pipefail`; (f) lifecycle = scratch wiped post copy-out, artifacts to `${JUNO_WORK}/cloud_runs/<RUN_TAG>/`, pulled via `scripts/sync_results_from_juno.sh`, results git-tracked under `results/reframe_suite/hardening/` (`cache: false` convention; DVC-track anything > 10 MB). Data mirror presence on Juno re-verified with a remote `ls` before submission, not assumed (rule 15).

**Ordering rationale:** panel's "(ii) first" is CONFIRMED, with one honest correction — (ii) is cheapest but not unilaterally decisive: the H3 verdict requires item (i)'s like-for-like dedicated-binary numbers. (ii) first still buys harness validation and the baseline column. Item (i) runs regardless of (ii)'s preview (cost trivial; H2 needs it independently). (iii) runs after (i) because α_equiv consumes the dedicated C2-C3 numbers.

## §9 — Agent routing

| Deliverable | Owner |
|---|---|
| `hardening_pair_binary.py` (item i, incl. `--smoke`) + `hardening_injection.py` (item iii); both reuse `run_stage1.py` helpers + `src/core/transforms.FluxPowerSpectrum`, no re-implementation | core-implementer |
| `hardening_fbar_baseline.py` (item ii); summary stats; figures: lattice-v2-vs-baseline comparison, C2-C3 M-curve with exclusion-bound annotation, injection sensitivity curve — all captions self-sufficient (config + headline number + comparison bar) | support-researcher |
| `scripts/submit_juno_hardening.sh` (clone of `submit_juno_stage1.sh`; PCV hard-asserts `pair_binary_cv.csv` + `pair_binary_summary.csv` nonempty with distinct FATAL exit codes; no `\|\| true` on artifact paths) | infrastructure-manager |
| §7 ICs pin + arXiv:2509.18260 novelty read | web-enabled general-purpose agent |
| data-engineer | NOT dispatched (confirm-only if item (iii) template work surfaces a data question) |
| item (iv) doc edits per §6 checklist | orchestrator (no judgment calls required) |
| **defense-panel re-review of all H-gate verdicts — MANDATORY.** This sprint is promotion-direction; the rule-15 narrowing exemption does NOT apply. No headline re-instatement, no paper-shape statement, and no park-state restoration until panel APPROVE. | defense-panel |
| paper-author | NOT dispatched (user-triggered-only; unchanged) |

## §10 — Decision-log reservations

[D-27] = adoption of this spec (cites this file + commit hash). [D-28] = H1/H2/H3 empirical verdicts. [D-29] = H4/injection verdict. [D-30] = re-verb pass record + shared-ICs pin + novelty-log additions. All four PROVISIONAL until defense-panel APPROVE per §8 step 7.

---

**Judgment calls made in this spec (vs binding rules):** track placement on the existing branch (§1); all-6-pairs + C4-vs-rest scope for item (i) (§2); P(k)-space injection rather than flux-space (§4, caveat pre-committed); the specific gate numerics 0.05/0.02 headroom bands and the 0.60 detection threshold (§5 — project-internal by necessity, disclosed as such; no external anchor exists, rule-14(ii) rescue is the pre-committed KILL routing itself); append-don't-rewrite doc amendments (§6). Everything else follows binding rules cited inline.

**Authored by:** project-architect (PI), 2026-06-10, PROVISIONAL per rule 15. Committed by orchestrator 2026-06-11 as the pre-commitment record.
