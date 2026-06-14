# HARDENING_OUTCOME — reframe-suite defense-panel neutralization sprint

**Status:** COMPLETE. Compute on Juno (job 213282, COMPLETED 1h44m, ExitCode 0:0, RUN_TAG `rshardening-20260612-172444-06d7ac`), 10-seed × 5-fold, dedicated binary RandomForest. Two defense-panel re-review rounds + data-engineer provenance audit. **Decision date:** 2026-06-12.
**Sign-off:** PI-PROVISIONAL lifted to defensible-park-state on H1/H3/H4; H2 defensible as a controlled weak-signal finding with one literature-pinned (not on-disk-confirmed) resolution caveat. Paper-author remains user-triggered-only.
**Numbers of record:** `results/reframe_suite/hardening/{pair_binary,fbar_baseline,injection_recovery}_summary.csv`. All reads per_sightline, p16 = 16th percentile of 50 seed×fold values.

---

## H1 — lattice integrity (dedicated binary vs old confusion sub-block): **HARDENS**

Dedicated-binary p16 at M=64 replaces the superseded 4-class sub-block estimates as the numbers of record:

| cell | old sub-block p16 | dedicated p16 | verdict |
|---|---|---|---|
| C1-C2 | 0.842 | 0.854 | confirmed |
| C1-C3 | 0.741 | **0.814** | confirmed — **newly clears 0.75** |
| C1-C4 | 0.978 | 0.971 | confirmed |
| C2-C4 | 0.980 | 0.961 | confirmed |
| C3-C4 | 0.926 | 0.951 | confirmed |
| C4-rest (R1) | 0.960 | 0.954 | confirmed |
| C2-C3 | 0.501 | 0.520 | (see H2) |

**R5 SURVIVES, strengthened: 5/5 non-(C2-C3) pairs now clear p16 ≥ 0.75** (was 4/5; C1-C3 newly clears). **R1 SURVIVES: C4-rest p16 = 0.954 ≥ 0.90.** The panel's attack-1 (lattice = preprocessing/estimator artifact) and three-inconsistent-numbers attacks are NEUTRALIZED — one estimator, one number per cell, pre-committed in HARDENING_SPEC §2 before the run.

## H2 — C2-C3 (StellarWind vs WindAGN) M-conditional behavior: **NULL RETIRED → weak P_F(k)-shape signal (gate-marginal, controlled)**

Dedicated-binary C2-C3 M-curve (per_sightline median): 0.36 (M=1) → 0.43 → 0.50 → 0.51 → 0.57 (M=64) → 0.61 (M=128) → **0.686 (M=256, p16 0.603)** → 0.77 (M=512*) → 0.83 (M=1024*). (* = exploratory, few-groups.)

Pre-committed routing (HARDENING_SPEC §5 H2): p16@M256 ≥ 0.60 → **"C2/C3-null verb retired project-wide; cascade-close partially reopens; mandatory panel review."** Trigger met (p16 = 0.6026), and the panel review it mandates has been done (this doc).

**Three controls confirm the rise is real and physical, not artifact:**
1. **Not stacking-variance** — within-class C2-vs-C2 self-pair (injection α=0 arm, disjoint C2 halves, same physics) stays at chance: 0.509 (M=64) → 0.500 (M=256). Per the round-1 panel's own pre-stated criterion ("self-pair climbs above 0.55 at M=256 → variance artifact"), the criterion is NOT met.
2. **Not mean-flux** — ⟨F⟩-only baseline for C2-C3 stays flat at chance at every M (≤ 0.512); the rise is carried entirely by P(k) shape.
3. **Not box-systematic** — `vel.npy` + `axis.npy` bit-identical across classes (data-engineer audit 2026-06-12): same box size, same IC skewer geometry. The published-suite 40-1024-vs-80-512 mismatch worry is ruled out from disk. C2 and C3 are the same density realization differing only in feedback.

**Honest caveats (load-bearing, [D-37]):**
- The gate is MARGINAL: p16 = 0.6026 is a 0.003 hug of the 0.60 line; bootstrap P(p16 ≥ 0.60) = 0.739 (95% CI on p16 [0.577, 0.637], straddles 0.60). The retirement of the null verb is mandatory per the pre-committed routing, but the strength is hedged: "weak signal, requires M ≥ 128 to clearly exceed chance," NOT a clean threshold pass.
- The signal is nearly **orthogonal to the C4-detection template**: α_equiv ≈ 0.004 vs β = 0.539 (template would predict acc ≈ 0.96 at α = β; observed 0.57). The C3−C2 P(k)-shape difference is a distinct, weaker spectral signature, not a scaled-down C4 signal. The H4 injection instrument (built on the C4 template) therefore does NOT calibrate the C2-C3 sensitivity floor — H2 rests on controls 1–3, not on H4.
- **Provenance caveat — RESOLVED 2026-06-13 (verb PROMOTED).** `data/raw` was pulled (granular `dvc get` from the S3 remote) and the native Sherwood LOS `.dat` headers read for all 4 classes. **All four are byte-identical in box size, cosmology, redshift, and sampling:** z=0.300, Ω_m=0.3080, Ω_L=0.6920, Ω_b=0.0482, h=0.6780, **box = 60.00 cMpc/h** (60000 ckpc/h), nbins=2048, numlos=16384 — **including class 4** (previously UNRESOLVED, now confirmed matched). The four are the packaged `Physics1-4` feedback-variant comparison set ("Sherwood Simulation Suite, Bolton et al. 2017"; data README), processed through a single shared `COS_LSF1` line-spread kernel (class-independent LSF — `Sherwood/src/utils.py`). Combined with bit-identical `vel.npy`/`axis.npy` (same IC skewer geometry), the four classes are the same realization with only feedback varied. **The box/resolution-confound is closed from disk; the H2 verb is promoted to "weak feedback signal" (StellarWind-vs-WindAGN).** Particle resolution (512³ vs 1024³) is the one property not stored in the LOS format, but it cannot differ between classes given identical box+cosmology+IC-geometry. **Correction to prior pin:** the measured box is **60 cMpc/h**, NOT the "80-512" label inherited from the HARDENING_SPEC §7 web pass — that label was imprecise; reconcile against the Bolton+2017 run table before citing the specific run in any paper (60 cMpc/h is non-canonical for Bolton+2017's 40/80/160 boxes — flag for paper-author).

This is published-direction-consistent with Nasir+2017 (same Sherwood suite: "feedback has only a small effect on low-z forest statistics") and the milder radiative-AGN regime of Tillman+2024.

## H3 — mean-flux baseline headroom: **NARROWED (not "rebutted")**

Dedicated P_F(k) p16 minus ⟨F⟩-only-scalar baseline p16. At the saturated M=64 read the C4 cells have thin headroom (C4-rest +0.062, C2-C4 +0.049, C3-C4 +0.048 — two in the AMBIGUOUS band). At the **non-saturated** M=16/32 reads the headroom is clean and large (M=16: C4-rest +0.121, C2-C4 +0.114, C3-C4 +0.078; all clear the +0.05 HARDENS bar). NoFB-vs-feedback cells (C1-Cx) clear by +0.12 to +0.20 everywhere.

**Honest verb (per panel):** the lattice ordering correlates with per-class ⟨F⟩ (Spearman ≈ 0.94 — NOT overturned), but P_F(k) carries genuine separability beyond ⟨F⟩ at every cell when measured off the accuracy ceiling. The headline-KILL condition (≥3 of 4 R5-qual cells OR C4-rest in the <0.02 KILL band) is NOT triggered. The "lattice = mean-flux in disguise" attack is **narrowed, not rebutted** — mean-flux ordering is real and substantial, with a real P(k)-shape residual on top.

## H4 — injection-recovery sensitivity instrument: **VALID at M=64; off-template for C2-C3**

α=0 calibration: median 0.509 (M=64), 0.500 (M=256) — centered at chance; CI spans 0.5. The M=256 arm is variance-dominated (α=0 CI [0.369, 0.631], few groups) so its δ_min=0.20 is unreliable; **δ_min(M=64) = 0.02** is the instrument read — the pipeline detects a P(k)-shape perturbation of 2% of ΔP_{C4−C2} at M=64. β = 0.539. The δ_min(M=64)≪β-but-C2C3-at-chance result is resolved by the orthogonality finding (H2 caveat 2): the instrument calibrates sensitivity along the C4 template direction only. **"Methodology eliminated" is replaced with the quantitative statement: sensitive to ≥2%-of-C4-template P(k)-shape perturbations at M=64.**

---

## Net disposition

| original panel killer (2026-06-10) | status after sprint |
|---|---|
| lattice = preprocessing/estimator artifact | **NEUTRALIZED** (H1) |
| three-inconsistent-numbers / wrong estimator | **NEUTRALIZED** (H1) |
| lattice = mean-flux in disguise (Spearman 0.94) | **NARROWED** (H3) — real ⟨F⟩ ordering + real P(k) residual |
| C2/C3-null over-verbed | **CORRECTED both ways** (H2) — was over-claimed as "intrinsic null"; sprint initially over-corrected to "feedback signal"; final honest verb is "weak P(k)-shape signal, gate-marginal, controls-confirmed not-variance/not-mean-flux/not-box, resolution-provenance literature-pinned" |
| methodology-eliminated = sensitivity fallacy | **QUANTIFIED** (H4) — δ_min(M=64)=0.02 |

**Project state: defensible to park-and-write** on (H1) a strengthened distinguishability lattice + (H3) a narrowed mean-flux caveat + (H2) a controls-confirmed weak C2/C3 P(k)-shape signal + (H4) a calibrated sensitivity floor. The paper trigger remains user-owned.

**Pre-publication provenance item — DONE 2026-06-13.** `data/raw` LOS headers read for all 4 classes (granular `dvc get`): matched box (60 cMpc/h), cosmology (Ω_m=0.308, Ω_b=0.0482, h=0.678), redshift, sampling, IC geometry, and a shared class-independent COS LSF. **H2 verb promoted to "weak feedback signal"; class-4 provenance resolved.** One residual flag for paper-author: the measured box (60 cMpc/h) does not match Bolton+2017's canonical 40/80/160 boxes — reconcile the exact run/citation before publication. (Absolute particle resolution 512³-vs-1024³ is not stored in the LOS format and cannot differ between classes given the matched box+cosmology+IC-geometry.)

**Decision-log:** [D-27] = HARDENING_SPEC adoption (committed 2a81f67-era). [D-28] = H1/H2/H3 empirical verdicts (this doc). [D-29] = H4/injection verdict (this doc). [D-30] = re-verb pass + shared-ICs pin + box-size on-disk confirmation + novelty log (this doc + pass-1 commit).

**Authored by:** orchestrator under PI discipline, 2026-06-12, after two defense-panel rounds (round-1 NEEDS WORK → controls → round-2 NEEDS-WORK-one-blocker → data-engineer box-size resolution). Honest [D-37] framing throughout.
