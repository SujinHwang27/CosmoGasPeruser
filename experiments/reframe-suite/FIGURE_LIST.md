# FIGURE_LIST — paper-narrative figures (for paper-author dispatch IF the user triggers)

**Purpose.** Catalog of figures that would land the cumulative reframe-suite + cascade-discipline narrative *intuitively*. Curated for the methods-with-case-study framing. MP4 / animated formats noted where they would carry more information than static.

**Status.** Authored 2026-06-02, this session. NOT a paper-author dispatch (paper-author remains user-triggered-only per binding rule). This list is the visual-inventory prep for IF the user triggers; for the rule-7-clean "no paper" path, this list sits in the LEDGER trail as future-decision surface.

**Source data.** All figures draw from artifacts already on disk: `results/pk_feedback_classifier/stage1/*.csv` (Phase 1 source), `results/reframe_suite/{phase1,phase2}/*.csv` (reframe-suite outputs), `results/signal_clustering_v2/figs/*.{png,html,mp4,gif}` (existing v0.4 figures), `data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,tau,vel}.npy` (raw inputs).

**Novelty/citation flags (2026-06-11, HARDENING_SPEC §6 E8).** (1) **Pirecki, Tillman, Burkhart, Tonnesen, Bird — arXiv:2509.18260** ("Exploring the impact of AGN feedback model variations on the Lyman-α Forest Flux Power Spectrum", accepted ApJ; CAMELS-Simba parameter variations, z = 0.1–2.0 incl. low-z): **must-cite + novelty-delta check required before any paper trigger.** It preempts the bare statement "AGN feedback measurably alters the low-z Lyα P1D"; what it does NOT contain: a pairwise recipe-vs-recipe distinguishability lattice, M-scaling of distinguishability, or matched-IC discrete recipes — that delta is our surviving contribution shape. (2) **Sinigaglia+2026 (arXiv:2603.13011)** covers z = 2.0–3.5 only — no low-z overlap; cite as adjacent SBI prior. (3) **F1 (lattice heatmap) and F2 (accuracy-vs-M) must be drafted from the HARDENING item-(i) dedicated-binary numbers, not the superseded sub-block estimates** — do not draft before HARDENING H1/H2 verdicts land.

---

## §1 — Headline figures (3 figures, the "if you only read the abstract" set)

### F1 — Distinguishability lattice (the load-bearing result)

**Form.** 4×4 heatmap. Rows and columns = the 4 feedback classes (C1=NoFB, C2=StellarWind, C3=WindAGN, C4=WindStrongAGN). Off-diagonal cells = binary balanced_acc at (per_sightline, M=64), color from green (≥ 0.95) through yellow (~0.75 the qualification bar) to red (≤ 0.55 near chance). Diagonal cells grayed out or labeled "self." Each cell annotated with the numeric p16 / median / p84.

**Why.** This single figure carries the substantively new finding — feedback-vs-feedback structure (C2-C4=0.98, C3-C4=0.93) AND the specific null (C2-C3≈0.5). A reader sees in 5 seconds: "C4 separable from everything, C1 separable from feedback, ONE specific pair fails."

**Source.** `results/reframe_suite/phase1/reframe5_lattice.csv` + R1 row from `reframe1_binary_c4_vs_rest.csv` (to add as a diagonal-or-summary row).

**Variant (MP4 candidate, ~5 sec).** Lattice fills in column-by-column as M sweeps 1 → 256. Shows visually how the structure emerges with stacking depth. Frame rate: ~0.5s per M-value.

---

### F2 — Accuracy-vs-M curves (binary + 4-class + 6 pairs overlay)

**Form.** Line plot. X-axis: M ∈ {1, 4, 16, 32, 64, 128, 256} log scale. Y-axis: balanced_acc, [0, 1]. **Bold curves:** (a) 4-class (closes at 0.83 at M=256 per [D-26]); (b) binary {C4 vs rest} (climbs to 0.98 at M=256 per R1). **Lighter curves:** the 6 pairwise binaries from R5, color-coded — C2-C3 sits at chance (0.50), C4-pairs cluster near top, C1-feedback pairs intermediate. Horizontal dashed lines at 0.50 (binary chance), 0.25 (4-class chance), 0.90 (R1 qualification bar). Shaded bands = 16th-84th percentile bootstrap CI.

**Why.** Shows the M-sweep monotone climb is real, the binary lift over multiclass is real, and the lattice structure is layered on top of the bulk-stacking effect. Conveys "stacking is the load-bearing operation" + "the binary framing recovers more than the multiclass" + "C2-C3 is the specific null."

**Source.** `results/reframe_suite/phase1/{reframe1_binary_c4_vs_rest,reframe5_lattice}.csv` + `results/pk_feedback_classifier/stage1/summary.csv` (for the 4-class curve).

---

### F3 — The cascade-discipline timeline (methods-paper headline)

**Form.** Vertical / horizontal timeline diagram showing the five project tracks (eda → baseline-rf → signal-clustering-v2 → pk-feedback-classifier → fps-sbi (scoped-closed) / linewidth-cddf (scoped-closed) → reframe-suite). Each track shown as a box with: name, start date, close date, closure reason (color-coded: green = qualified / yellow = partial / red = falsified / gray = scoped-closed-before-compute). Arrows between tracks labeled with the cascade event ("[D-13] info ceiling", "[D-07] high-k falsified", "[D-26] mid-k falsified", "Q-S1 discrete-only", "Q-C1 signal-limited"). The reframe-suite box at the end shows the green qualification.

**Why.** Methods-paper headline. Shows the discipline operating in real time. Conveys "we ran 5 tracks; 3 closed under the cascade rule; 2 scoped-closed before compute saved by the discipline; 1 qualified under reframed questions." A reader sees the discipline in action.

**Source.** Manually compose from the LEDGERs + SCOPING / CLOSE_OUT docs. Tool: TikZ, Mermaid, or a hand-drawn diagram.

---

## §2 — Supporting figures (5 figures, "and here's the evidence")

### F4 — P_F(k) per class at M=64 (the "what the classifier sees")

**Form.** Log-log plot. X-axis: k in s/km, range ~10⁻³ to ~1.2. Y-axis: k × P_F(k) or P_F(k). Four curves, one per feedback class, color-coded. Bands for bootstrap CI. Vertical shaded region marking the [0.04, 0.15] s/km "mid-k peak" zone. Vertical line at the k=0.3 s/km expected thermal-cutoff.

**Why.** Shows what the actual signal looks like — C4 sits apart from C1+C2+C3 across the mid-k regime, C2 and C3 are visually indistinguishable. Anchors the classification result in a recognizable Lyα/IGM figure.

**Source.** Either re-extract from the existing P_F(k) feature arrays (in-memory inside the Stage 1 run), OR re-run a small script reading `flux.npy` and computing the M=64 stacked P(k) per class. Estimated 5-10 minute compute.

**Variant (MP4, ~10 sec).** P(k) curves sharpen/converge as M sweeps 1 → 256. The C4 separation emerges visually with stacking depth.

---

### F5 — The R9 ⟨F⟩-removal control (honesty-test figure)

**Form.** Two-panel comparison. Left panel: binary {C4 vs rest} balanced_acc curve under existing per_sightline norm (R1 baseline). Right panel: same curve under per_sightline_mean_removed norm (R9). Identical-within-noise — the figure shows visually that R9 PASSES (curves overlap). Include the SCOPING-§5-anticipated multiplicative-vs-additive degeneracy as an inset annotation: "⟨F⟩ ≈ 0.97 → multiplicative/additive removals indistinguishable; non-claim of total ⟨F⟩-independence holds."

**Why.** The honesty hurdle is visible. Conveys "we ran the control, it passed, and we honestly disclose the caveat that the control was weaker-than-spec'd due to the small-δ_F regime." Disarms the obvious reviewer attack.

**Source.** `results/reframe_suite/phase2/cv_partial_meanremoved.csv` + `results/reframe_suite/phase1/reframe1_binary_c4_vs_rest.csv`.

---

### F6 — Confusion matrix at M=256 (the headline-with-decomposition)

**Form.** Standard 4×4 row-normalized confusion. Annotated cells with percentages. Highlights: C4 row near-diagonal (~0.96 recall); C1 row near-diagonal but lower; C2 and C3 rows ENTANGLED (off-diagonal ~0.4-0.5 between C2 and C3).

**Why.** Shows the multiclass result decomposes EXACTLY as the lattice predicts. The C2-C3 entanglement is visually obvious; the C4 perfection is visually obvious. The lattice (F1) is the abstract statement; this figure is the concrete evidence.

**Source.** `results/pk_feedback_classifier/stage1/cv_partial.csv` aggregated (already produced for Phase 1 source).

---

### F7 — K=5 cluster vs mean-flux (R7 result)

**Form.** Violin or box plot, 2-row × 5-column grid. Top row: wavelet_k5 clusters 0..4. Bottom row: raw_k5 clusters 0..4. Each violin shows the distribution of per-sightline ⟨F⟩_los for that cluster. η²=0.125 annotated on the wavelet panel.

**Why.** Shows the R7 informative-null: the clusters DO partially track mean flux (visible in the violin shifts) but the segregation is medium-effect not large — confirming [D-13]'s reading that the unsupervised structure picks up the dominant variance axis (⟨F⟩-ordering) when class-discrimination signal is weak.

**Source.** `results/reframe_suite/phase1/reframe7_cluster_axes.csv` + `data/feature_discovery/labels_{wavelet,raw}_k5.npy` + `data/preprocessed/Sherwood_z0.3_inf/{1..4}/flux.npy`.

---

### F8 — RF feature-importance bar chart (the mid-k peak that started the cascade)

**Form.** Bar chart of RF `feature_importances_` across the 20 log-spaced k-bins. X-axis: k-bin index (or physical k in s/km). Y-axis: importance. Shaded mid-k region (idx 10/11/12, k ≈ 0.06-0.13 s/km) showing the peak that drove [D-10] re-anchor + [D-16] ablation gate + [D-26] G3+G6 falsification. Annotation: "G6 ablation dropped top-3 mid-k bins; Δacc = 1.5 pp << 5.3 pp threshold → mechanism falsified."

**Why.** Visual evidence of the cascade event itself. Shows what the falsified prior actually looked like in the data. Useful for both science paper (mechanism narrative) and methods paper (the falsification gate's payload).

**Source.** Existing Stage 1 results — the `kbin_importance_*.csv` per regime/M from `cloud_runs/pkstage1-20260601-184636-76c1de/` or re-aggregate from cv_partial if those CSVs were not the kbin-importance ones.

---

## §3 — Methods-paper-only figures (2 figures, for the discipline-contribution narrative)

### F9 — Falsified-prior cascade verb-ceiling diagram

**Form.** Vertical hierarchy / tree. Top: initial verb-ceiling at signal-clustering-v2 start ("high confidence: P(k) basis breaks ceiling"). Each cascade event drops one level. Each downgrade shown as an arrow with the rule-2 invocation label ("[D-01] high-k falsified → +1 downgrade") and the resulting verb-ceiling-after ("'first test of mid-k regime'"). Bottom: reframe-suite verb-ceiling ("'first test of binary detection, gated on ⟨F⟩-control'"). Visual showing the discipline forces increasingly hedged verbs as evidence accumulates.

**Why.** Methods-paper headline. Shows the unique contribution — most preregistration discipline is single-experiment; this is chain-level discipline. A reader sees the verb-degradation forced by the cascade rule operating.

**Source.** Compose from the LEDGER trail (verbs from [D-01], [D-10]/[D-15], [D-26], reframe-suite SCOPING §1 verb ceilings).

---

### F10 — Scoping-pass-kills timeline (the cost-savings argument)

**Form.** Bar chart or timeline. For each track (fps-sbi, linewidth-cddf), show: (a) wallclock that would have been spent on full Stage-N compute if the scoping passes had not gated (estimated: fps-sbi → 12-24h on Juno; linewidth-cddf → 10-30h depending on Voigt-fit budget); (b) wallclock actually spent (zero on compute, <1h on audits); (c) the gate that fired (Q-S1, Q-C1, Q-C3) and the kill-path. Annotation: "two scoping passes saved >24h of cluster compute and at least one bad paper each."

**Why.** Concrete cost-savings evidence for the methods paper. Shows the discipline isn't just epistemically clean; it's *operationally efficient*. Reviewers care about wall-clock and cluster cost.

**Source.** Compose from `fps-sbi/Q_S1_PARAM_AUDIT.md`, `linewidth-cddf/Q_C1_PIPELINE_FEASIBILITY.md`, `linewidth-cddf/Q_C3_LIT_ANCHOR.md`, plus the pk-feedback-classifier Stage 1 wallclock (52m31s for 7940 fits, scaling for the closed tracks' expected fit counts).

---

## §4 — Pre-existing artifacts worth re-using directly

These exist on disk from the closed tracks; usable verbatim or with minor edit:

- **`results/signal_clustering_v2/figs/fig_drift_animation.mp4`** + `.gif` — existing MP4 from v0.4 showing UMAP-2D cluster drift. Could be re-purposed as a supplementary "what unsupervised separability vector space looks like" for the methods paper.
- **`results/signal_clustering_v2/figs/fig_umap_2d_{wavelet,raw}_k5.png`** — 2D UMAPs per representation; show the K=5 clusters spatially. Supplementary visual for F7.
- **`results/signal_clustering_v2/figs/fig_umap_3d_{wavelet,raw}_k5.html`** — interactive 3D UMAPs. Online-supplementary-only (HTML), not print-friendly.
- **`results/signal_clustering_v2/figs/fig_spatial_map_*.png`** — spatial cluster maps. Could anchor a "where in the sky" supplementary.

---

## §5 — Figure-narrative arc recommendation

If the user triggers paper authoring and wants a tight 5-figure science paper:

1. **F3 cascade timeline** (Introduction — "here's what we did and why we stopped where we stopped")
2. **F4 P_F(k) per class** (Methods/Data — "here's what the observable looks like")
3. **F2 accuracy-vs-M curves with lattice overlay** (Results — "the binary signal is real and stacking-driven, but feedback-vs-feedback is structured")
4. **F1 distinguishability lattice heatmap** (Results — "and here's exactly what the structure is")
5. **F5 ⟨F⟩-removal control panel** (Results — "and here's the honesty-control; the signal is shape-encoded subject to the documented caveat")

For an 8-figure methods+science combined paper, add F6 (confusion), F8 (importance bar with ablation), F9 (verb-ceiling diagram), F10 (scoping-kills timeline).

For supplementary (online-only): F7 (cluster-vs-mean-flux violins), the pre-existing UMAP HTMLs + drift MP4.

**MP4 candidates ranked by intuitive payoff:**
1. **F1 lattice-fills-in-as-M-sweeps** (~5 sec, highest payoff — shows structure emergence visually)
2. **F4 P(k)-curves-sharpen-as-M-sweeps** (~10 sec, shows what stacking does to the signal)
3. **Existing v0.4 drift animation** (use as-is, ~30 sec)

---

**Authored:** 2026-06-02, this session.
**For paper-author dispatch IF the user triggers.** No dispatch authorized by this list. This list is the visual-inventory prep recorded in the LEDGER trail for future-decision surface.
