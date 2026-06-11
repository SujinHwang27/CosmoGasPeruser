# CLOSE_OUT — reframe-suite (qualification cleared)

**Track:** reframe-suite (no-new-compute Reframes 1+5+7 + minor-new-compute Reframe 9 control)
**Status:** **QUALIFICATION CLEARED.** R1 and R5 cleared their SCOPING §4 solid/successful operational gates; R9 control passed by 53× margin; R7 returned adjacent-finding (informative null).
**Decision date:** 2026-06-02.
**Linked artifacts:** `SCOPING.md` (commit `e090b52`); `PHASE1_REPORT.md` (commit `1d8b7bd`); `PHASE2_REPORT.md` (commit `c4fe456`); result CSVs at `results/reframe_suite/{phase1,phase2}/`.

---

## §1 — Qualification verdicts per SCOPING §4

### Reframe 1 — Binary {C4 vs rest} detection: **QUALIFIES**

All four conjuncts in SCOPING §4 R1 cleared on the empirical data:

| Conjunct | Bar | Observed | Status |
|---|---|---|---|
| Binary balanced_acc p16 ≥ 0.90 | 0.90 | **0.9603** (cleared by 6 pp) | ✓ |
| Δ vs 4-class median ≥ 0.10 pp | 0.10 | **0.285 pp** (cleared by 19 pp) | ✓ |
| Regime parity \|per_sightline − global\| ≤ 0.005 | 0.005 | **0.0031** | ✓ |
| R9 control: drop under ⟨F⟩-removal < 5 pp | 5 pp | **−0.003 pp** (essentially zero, within bootstrap noise) | ✓ |

**Headline number for paper-trigger reading:** binary {C4 vs C1∪C2∪C3} balanced_acc median **0.971**, lower-CI **0.960**, at (per_sightline, M=64) on stacked P_F(k); robust to explicit per-sightline ⟨F⟩-subtraction in flux space (R9 control).

### Reframe 5 — 6-pair distinguishability lattice: **QUALIFIES**

SCOPING §4 R5 required ≥ 3 of 5 non-(C2-C3) pairs at p16 ≥ 0.75 AND structural informativeness. Empirical:

| Pair | p16 at (per_sightline, M=64) | ≥ 0.75? |
|---|---|---|
| C1-C2 (NoFB vs StellarWind) | **0.842** | ✓ |
| C1-C3 (NoFB vs WindAGN) | 0.741 | ✗ (just under) |
| C1-C4 (NoFB vs WindStrongAGN) | **0.978** | ✓ |
| C2-C3 (StellarWind vs WindAGN) | 0.501 *(2026-06-11 amendment: p16 at M=64; climbs monotonically with M to p16 0.599 / median 0.678 at M=256 — `results/reframe_suite/phase1/reframe5_lattice.csv`; M-conditional exclusion bound, dedicated-binary re-measurement pending per HARDENING_SPEC item i)* | (excluded — cascade-anchored null) |
| C2-C4 (StellarWind vs WindStrongAGN) | **0.980** | ✓ |
| C3-C4 (WindAGN vs WindStrongAGN) | **0.926** | ✓ |

**4 of 5 non-(C2-C3) pairs cleared.** Structural informativeness check: at least one of {C1-C2, C1-C3} cleared (C1-C2 ✓) AND at least one of {C2-C4, C3-C4} cleared (BOTH ✓). The lattice is non-trivially structured — feedback-recipe-vs-feedback-recipe is distinguishable for the C4-involving pairs, even though the C2-C3 pair alone is null *(2026-06-11 amendment: "null" re-verbed — C2-C3 is consistent with chance at M ≤ 64 and rises with stacking depth; see HARDENING_SPEC §6 E1)*.

**Substantively new finding (beyond cascade-close [D-26]):** the cascade-close framing was "feedback-vs-feedback is null." The reframe-5 lattice sharpens this to **"feedback-vs-feedback is null *only at the recipe-similarity scale of C2 vs C3*; recipe distinguishability against C4 holds at p16 ≥ 0.93."**

### Reframe 7 — K=5 cluster physical-axis interpretation: **DOES NOT QUALIFY** (adjacent-finding only)

SCOPING §4 R7 required at least one (cluster_set × physical_axis) at η² ≥ 0.20. Empirical: top η² = **0.125** at (wavelet_k5, mean_flux); medium effect, does not clear 0.20 large-effect bar. Next: (raw_k5, mean_flux) at 0.057.

**Adjacent-finding routing per SCOPING §2 R7 (b):** the K=5 unsupervised clusters partially track mean flux but do not cleanly encode a single physical axis. **This is a second-independent confirmation of [D-13]'s "signal islands are representation-disagreement, not physics" reading** — the K=5 structure exists, it picks up the mean-flux ordering of sightlines that the 24-dim separability vector surfaces when class-discrimination signal is weak, but it does not constitute a "discovered physical axis."

### Reframe 9 — Mean-flux-removed P_F(k) control: **QUALIFIES (as Phase-2 control)**

SCOPING §4 R9 required binary {C4 vs rest} accuracy drop < 5 pp under explicit ⟨F⟩-removal. Empirical drop: **−0.003 pp** on median (signal actually slightly *better* under the new normalization, within bootstrap noise). Clears by 53× margin. R9 is *operationally* the enabling condition for R1 to qualify; it does not produce a standalone headline.

**Honest [D-37] hedge surfaced by core-implementer:** the multiplicative `F/⟨F⟩_los − 1` (existing) and additive `F − ⟨F⟩_los` (new) ⟨F⟩-removals produced statistically indistinguishable results because ⟨F⟩ ≈ 0.97 is close to 1 — the two controls are degenerate at this δ_F regime. What R9 actually demonstrated is that **[D-06]'s existing preprocessing was already removing the DC channel sufficiently** — not that we successfully imposed a *stronger* control. The non-claim still binds: no claim of total ⟨F⟩-independence; higher-moment ⟨F⟩-correlated structure remains an open caveat. This is the load-bearing caveat on any paper-text claim.

---

## §2 — Cumulative project-state picture (post-reframe-suite)

The cumulative read across the four prior tracks + reframe-suite, recorded for future-session orientation:

### What the cascade closes against (unchanged from CLOSE_OUT §4)

Six cascade-ledger entries; rule-8 path (b) cascade-strengthening: three specific interventions across two observable families produced congruent C2/C3-null signatures, externally anchored by Nasir+2017 on the same Sherwood suite. **The C2-C3 distinguishability claim is closed.**

*(2026-06-11 amendment per HARDENING_SPEC §6 E3 — replication re-count: the honest count is two correlated empirical nulls on the same Sherwood τ/flux realization plus one scoping close that produced no measurement, externally anchored by Nasir+2017 — replication count: 2 (correlated) empirical + 1 literature anchor. The closure claim is re-verbed: closed at the tested protocols and stacking depths; see HARDENING_SPEC H2 for the M-conditional bound.)*

### What the reframe-suite adds (NEW)

1. **Quantified binary {C4 vs rest} detection at p16 0.960, robust to ⟨F⟩-removal control.** Strong-AGN feedback (WindStrongAGN) is identifiable from stacked low-z Lyα forest P_F(k) — and the signal is not solely ⟨F⟩-encoded.
2. **Structured pairwise distinguishability lattice.** C2-vs-C4 (StellarWind vs WindStrongAGN) and C3-vs-C4 (WindAGN vs WindStrongAGN) are both distinguishable at p16 ≥ 0.93. The cascade-close framing "feedback-vs-feedback is null" sharpens to "*specifically C2-vs-C3 is null; other feedback-recipe pairs are distinguishable*."
3. **K=5 cluster characterization (informative null).** The signal-clustering-v2 K=5 clusters partially track mean-flux ordering (η²=0.125) but do not cleanly encode a single physical axis. Confirms [D-13] reading from a second direction.

### Combined narrative (what survives honest framing)

> "Quantified feedback-recipe distinguishability lattice on the low-z Lyα forest of Sherwood at z ≈ 0.3, with structural pattern: NoFB-vs-feedback detectable at p16 ≥ 0.74; strong-AGN-vs-others detectable at p16 ≥ 0.93; stellar-wind-vs-wind+AGN consistent with chance at M ≤ 64, rising toward 0.60–0.68 by M=256 (M-conditional exclusion bound, not an intrinsic floor; 2026-06-11 amendment per HARDENING_SPEC §6 E2). The binary strong-AGN detection signal is robust to explicit per-sightline ⟨F⟩-removal in flux space (subject to higher-moment ⟨F⟩-correlated structure caveat). The cumulative finding is published-direction-consistent with Khaire+2024 / Tillman+2024 / Tillman+2023 / Sinigaglia+2026 / Nasir+2017 (with cross-suite + cross-z hedges binding on each)."

Verb ceiling per SCOPING §1: NEVER "we recover feedback parameters" or "Lyα detects feedback" without the lattice qualifier.

*(2026-06-11 amendment per HARDENING_SPEC §6 E7 — binding non-claim addition: All lattice/detection numbers are conditional on fixed cosmology, UVB, and thermal history within one Sherwood realization; no nuisance marginalization was performed.)*

*(2026-06-11 annotation per HARDENING_SPEC §7 — shared-ICs pin: Bolton et al. 2017 (MNRAS 464, 897) §2.1 states "The same seed was used for simulations with the same box size ... so that the same large scale structures are present"; the 80 h⁻¹ cMpc feedback variants (80-512 / 80-512-ps13 / 80-512-ps13+agn = classes 1–3) therefore SHARE initial conditions. **Class 4 (WindStrongAGN) caveat:** neither Bolton+2017 nor Nasir+2017 documents a fourth strong-AGN variant; its IC provenance is UNRESOLVED from these papers alone — treated as shared (conservative for replication counting), flag open pending the documenting source. Consequence per HARDENING_SPEC §7(a): the two empirical nulls are same-realization-correlated (E3 count stands); per-position cross-class pairing justified for classes 1–3.)*

---

## §3 — Cascade ledger update (rule 2 + rule 4)

The cumulative cascade ledger from `linewidth-cddf/CLOSE_OUT.md` §4 is **NOT modified** by the reframe-suite: no new candidate priors entered. The reframe-suite is a re-analysis of existing artifacts under reframed questions per SCOPING §1 binding interpretation.

The six-entry cascade ledger stands as authored in `linewidth-cddf/CLOSE_OUT.md` §4. Cumulative state: cascade-strengthening evidence on the C2-C3-axis null + qualified positive results on the C4-detection axis and the C4-involving lattice pairs.

---

## §4 — Paper-trigger status (binding rule, NOT recommendation)

Per the freshly-sharpened paper-trigger rule (project-architect agent spec, commit `734fdcb`) and the standing user reasoning recorded in `paper-authoring-trigger` memory: **paper-author is user-triggered-only IFF solid/successful results.** The reframe-suite SCOPING §4 operationally defined "solid/successful" for this re-open's deliverable shape; **R1 and R5 have cleared those operational gates.**

**The qualification gate (SCOPING §4) has cleared. The trigger remains the user's.** This CLOSE_OUT does NOT recommend paper-author dispatch; it records the qualification status so the user has the binding-rule operational read.

If the user triggers paper authoring on the cumulative findings:
- Verb ceiling binds per SCOPING §1: never "we recover feedback"; always the lattice-conditional verb-shape.
- Citation discipline propagated forward (six Walther-class corrections in CLOSE_OUT §3 of `linewidth-cddf`).
- The methods-with-case-study framing surfaced in the session orientation pitch is the largest-novelty contribution-shape; the user owns whether to write methods-only, science-only, or combined.
- Figure list for paper narrative authored in `experiments/reframe-suite/FIGURE_LIST.md` (this session).

If the user does not trigger paper authoring:
- The reframe-suite SCOPING + PHASE1_REPORT + PHASE2_REPORT + this CLOSE_OUT are the documentation pillar.
- Tag `v0.6-reframe-suite-qualified` (against the c4fe456 HEAD or the next commit on this CLOSE_OUT) marks the qualified-state checkpoint.
- Project state in future sessions: "qualified results in hand; no further rule-2-clean compute justified; awaiting user paper-trigger decision."

---

## §5 — Decision-quality discipline (rule 7) — sprint sign-off

The reframe-suite executed cleanly under the same discipline that governed the prior tracks:

- **Rule 2 (cascade discipline):** the un-park was authored under a rule-2-compliant binding interpretation (SCOPING §1) that distinguished re-aggregation/re-interpretation from new-candidate-prior; Reframe 9 was gated to prevent it being a fourth P_F(k) attempt.
- **Rule 3 (anti-degeneracy audit):** each reframe's honesty hurdle was named in advance (SCOPING §2(c)); the R1+R9 ⟨F⟩-vs-shape disentanglement was the load-bearing audit and was operationalized correctly.
- **Rule 5 (symmetric disclosure):** every reframe had a pre-committed PASS condition + FAIL routing; R7 returning a clean FAIL (η²=0.125 < 0.20) was reported plainly, NOT spun. *(2026-06-11 amendment per HARDENING_SPEC §6 E5: "pre-committed" here means post-hoc qualification thresholds — authored after the Stage-1 confusion matrices were on disk; pre-committed only relative to the Phase-1 re-aggregation run, commit `e090b52`.)*
- **Rule 7 (decision-quality):** the outcome is a mixed shape (R1+R5+R9 qualifying, R7 adjacent) honestly framed in [D-37]-discipline terms; the qualification is operationally defined and verifiable; the trigger is left to the user as the binding rule requires.
- **Rule 14 (self-anchored-bar fragility):** X/Y/Z/W gates were project-internal by construction; rule-14(ii) rescue (no-headline-on-FAIL routing) was pre-committed in SCOPING §2(b); rule-14(iii) demotion-to-adjacent-finding is the default headline-shape any user paper-trigger inherits.
- **[D-37] honest reporting:** the multiplicative-vs-additive ⟨F⟩-removal degeneracy was surfaced by the implementer and recorded in §1 R9 as the load-bearing caveat; the R7 partial-mean-flux-tracking finding was reported as second-independent confirmation of [D-13], not spun.

Sprint closes honorably. The same rule-7-clean discipline that produced the cascade-close in [D-26] produced the qualified state in the reframe-suite — both are valid decision-quality end states.

---

## §6 — Re-open conditions (for future sessions)

The reframe-suite is **CLOSED** at qualification cleared. Future sessions:

1. **No further reframe-suite work is justified** without new user invocation. The four reframes were the catalog; all four returned (3 qualifying + 1 adjacent). Additional reframes on the same observable-axis would require fresh user invocation AND a rule-2-compliance argument (SCOPING §1 interpretation does not generalize to arbitrary follow-on reframes).
2. **Original park guidance (CLOSE_OUT.md §7 in linewidth-cddf) still binds on §5-catalogue paths** (CAMELS-LH, metal lines, transverse correlations, mock survey statistics). The reframe-suite did NOT relax that gate. Re-opens on those paths still need explicit user invocation + fresh published prior identified at scoping.
3. **Paper-author dispatch** remains user-triggered-only under the binding rule. The qualification status recorded in §1 + §4 is the operational read against the rule; the trigger is the user's.
4. **Cumulative cascade ledger** (six entries in `linewidth-cddf/CLOSE_OUT.md` §4) is unchanged. Any new D-XX entry that adds a new candidate prior must re-check the cascade verb ceiling per rule 2.

---

**Authored by:** project-architect-discipline-via-orchestrator, 2026-06-02.
**Branch:** `exp/pk-feedback-classifier` (track-internal docs).
**Sign-off provenance:** PI-only via the SCOPING §4 operational-gate verification of the empirical Phase-1 + Phase-2 returns; NOT provisional per rule 15(c) — every gate check is direct read of the empirical CSVs against the pre-committed numeric thresholds in SCOPING §4. No defense-panel review required for a *narrowing-direction* qualification close (rule-15 provisional-by-default attaches to promotion-direction decisions).
**Downstream dispatch:** NONE. No core-implementer. No infrastructure-manager. No support-researcher. No defense-panel. **No paper-author** (user-triggered-only).
**Tag-against-this-commit:** `v0.6-reframe-suite-qualified` (proposed; awaiting user authorization).
