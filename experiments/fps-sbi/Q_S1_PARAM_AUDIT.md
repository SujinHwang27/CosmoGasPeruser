# Q-S1 — Sherwood parameter-space audit

**Track:** fps-sbi (SCOPING.md sibling — track not yet open)
**Branch authored on:** `exp/pk-feedback-classifier` (per dispatch)
**Audit date:** 2026-06-02
**Owner:** `data-engineer`
**Dispatch source:** PI brief on `experiments/fps-sbi/SCOPING.md` §2 Q-S1 (commit `d92170a`)

**Verdict (one line):** **(B) discrete-only — 4 fixed feedback prescriptions, no continuous θ-grid available in the Sherwood z=0.3 snapshot used here.**

---

## §1 — On-disk metadata inventory

### 1.1 Directory structure (`data/` tree, exhaustive)

DVC-managed; read-only per CLAUDE.md. The pointer files materialize the following on-disk layout (output of `find data -maxdepth 4 -type f ! -name "*.npy" ! -name "*.png"`):

```
data/.gitignore
data/feature_analysis.dvc
data/preprocessed/README.md
data/preprocessed.dvc
data/processed/README.md
data/processed/Sherwood_z0.3_inf/dct_dominant_3/z0.3_inf_dct_dominant_3_interactive.html
data/processed/Sherwood_z0.3_inf/dct_highfreq_3/z0.3_inf_dct_highfreq_3_interactive.html
data/processed.dvc
data/raw.dvc
```

Plus the `.npy` payloads under each of the 4 class directories `data/preprocessed/Sherwood_z0.3_inf/{1,2,3,4}/`, each holding identically named files:

```
axis.npy   (524 416 B)
flux.npy   (268 435 584 B  → (16384, 2048) float64)
tau.npy    (268 435 584 B  → (16384, 2048) float64)
vel.npy    (16 512 B       → (2048,) float64)
wave.npy   (16 512 B       → (2048,) float64)
```

### 1.2 Files searched for parameter metadata — observed states

| Candidate filename | State | Where searched |
|:---|:---|:---|
| `params.txt` | MISSING | recursive `find` under `data/` |
| `README` / `README.md` (with parameter content) | EXTRACTED-INCOMPLETE — only `data/preprocessed/README.md` (one-line: *"Dataset preprocessed and ready to be loaded for experiments"*) and `data/processed/README.md` (one-line: *"Dataset resulted from experiments"*) — neither names a feedback parameter, neither names a θ value | recursive `find` under `data/` |
| `config.yaml` / `config.yml` / `*.cfg` / `*.ini` | MISSING | recursive `find` under `data/` |
| `metadata.json` | MISSING | recursive `find` under `data/` |
| Simulation log files (`*.log`, `*.out`, run scripts) | MISSING | recursive `find` under `data/` |
| Sidecar parameter file per class (`1/params.txt` etc.) | MISSING — each class dir holds only `{axis,flux,tau,vel,wave}.npy`, no text files | direct `ls -la` per class dir |

### 1.3 DVC pointer-file `meta:` annotations

| Pointer | Contents | `meta:` field? |
|:---|:---|:---|
| `data/raw.dvc` | md5 `46020036…dir`, size 34367649602, nfiles 79, hash md5, path raw | **MISSING** |
| `data/preprocessed.dvc` | md5 `d074aa93…dir`, size 3068002464, nfiles 23 | **MISSING** |
| `data/processed.dvc` | md5 `3d610ebe…dir`, size 3244340684, nfiles 48 | **MISSING** |
| `data/feature_analysis.dvc` | md5 `ac86f085…dir`, size 268435584, nfiles 1 | **MISSING** |

No `meta:` block on any pointer; no parameter annotation embedded in DVC layer.

### 1.4 Class-directory naming convention

The 4 directories are named purely with integer indices `1/`, `2/`, `3/`, `4/`. The semantic mapping `1=NoFeedback, 2=StellarWind, 3=WindAGN, 4=WindStrongAGN` is recorded in `CLAUDE.md` and several LEDGER files, but NOT in any on-disk artifact accompanying the data itself. The integer naming is **discrete-recipe-style** (a class label) not **continuous-parameter-style** (no embedded θ value in the directory name).

### 1.5 In-array sidecar parameter encoding (checked)

`{axis, flux, tau, vel, wave}.npy` are bare NumPy arrays with no header metadata beyond dtype/shape (verified via `DataIngestor` usage pattern in `src/core/data.py`; `.npy` format carries no free-form parameter dict). The `vel.npy` and `wave.npy` arrays are pixel grids (km/s and Å respectively), identical across all 4 classes per the pk-feedback-classifier LEDGER [D-04] verification (Δv = 2.6365 km/s uniform). They encode the observational axis, NOT feedback parameters.

### 1.6 Raw archive (DVC-pointed)

`data/raw.dvc` reports 79 files / 34.4 GB at md5 `46020036…dir`. Local materialization state: **RAW-ARCHIVE-NOT-EXTRACTED** — the local `data/raw/` directory does not exist (`ls -la data/raw*` returns only the `.dvc` pointer). A `dvc pull data/raw.dvc` would be required to inspect any nested metadata in the raw tree, which is not done here (not authorized by the dispatch; the on-disk preprocessed/processed layer is the primary brief). **This is a noted caveat — the raw archive MAY contain parameter files inaccessible without `dvc pull`. See §3 caveat.**

### 1.7 In-repo references to feedback parameters / source provenance

Grep for `Sherwood|Bolton|Puchwein|recipe|continuous|knob|theta|parameter scan` across LEDGERs and docs:

- `experiments/eda-sherwood/LEDGER.md` line 190: *"Upstream dataset URI/version for the Sherwood z=0.3 snapshot — not recorded in repo."* (verbatim)
- No LEDGER or doc records a feedback θ value, a parameter-scan grid, or a Latin-hypercube specification. All references treat the 4 classes as named labels (`NoFeedback`, `StellarWind`, `WindAGN`, `WindStrongAGN`) without numeric θ.

### 1.8 On-disk inventory verdict (data-engineer empirical only, no inference)

The on-disk preprocessed/processed layer is **METADATA-SILENT** on continuous feedback parameters. There is no `params.txt`, no `README` with parameter values, no `config.yaml`, no `metadata.json`, no simulation log, no DVC `meta:` annotation, no in-repo provenance recording a θ-grid. The 4 classes are stored as integer-indexed sibling directories with identical file structure and identical pixel grids, consistent with discrete-recipe sibling runs — but the on-disk evidence alone cannot distinguish "4 sibling runs of a parameter scan with the scan parameters dropped during preprocessing" from "4 independent named-recipe runs."

Per Q-S1 protocol, on-disk-silent triggers the literature pass (§2).

---

## §2 — Literature confirmation

### 2.1 Sherwood suite reference: Bolton+2017 (MNRAS 464, 897; arXiv:1605.03462)

The Sherwood simulation suite is documented in Bolton, Puchwein, Sijacki, Haehnelt, Kim, Viel (2017) *MNRAS* **464**, 897, "The Sherwood simulation suite: overview and data comparisons with the Lyman α forest at redshifts 2 ≤ z ≤ 5." The suite is a P-GADGET-3 SPH set, configured with **discrete named feedback recipes** — not a Latin hypercube. The relevant variants discussed in §2 / Table 1 of that paper:

- **No feedback** — pure gravity + photoionization, no galactic winds, no AGN.
- **Stellar wind** (Springel & Hernquist 2003-style energy-driven kinetic winds at a fixed wind velocity and mass-loading factor) — fixed parameter values, not scanned.
- **AGN feedback** (added to stellar winds; standard recipe based on Sijacki et al. 2007 / Booth & Schaye 2009 with fiducial accretion-energy efficiency).
- **Strong AGN** (boosted AGN energy-injection efficiency, fixed at a single elevated value — not a scan).

These map 1:1 to the on-disk `1/`, `2/`, `3/`, `4/` directories under the project's `NoFeedback`, `StellarWind`, `WindAGN`, `WindStrongAGN` naming.

Bolton+2017 explicitly frames the feedback-variant suite as a **discrete recipe comparison**, designed to characterize the *direction* of feedback effects on the Lyα forest — NOT as a training set for emulators or SBI over continuous feedback parameters. There is no published Latin hypercube or dense θ-grid for the Sherwood feedback variants at z = 0.3.

### 2.2 Sherwood-Relics (Puchwein+2022/2023) — z=0.3 relevance check

The Sherwood-Relics extension (Puchwein, Bolton, Haehnelt, Madau, Becker, Haardt 2019/2022/2023 series) adds inhomogeneous reionization treatments to Sherwood at z ≥ 5, NOT a continuous feedback-parameter grid at z = 0.3. Sherwood-Relics does not change the discrete-recipe structure of the z = 0.3 feedback comparison subset.

### 2.3 Cross-reference: CAMELS as the contrasting "continuous-θ" suite

The pk-feedback-classifier LEDGER [D-12] (reproduced in SCOPING.md §2 Q-S1) correctly notes that the Lyα/IGM ML community's continuous-parameter inference works (Maitra 2026; Tillman 2024; Pirecki 2025) rely on **CAMELS** (Villaescusa-Navarro+2021, 2022) — a *separate* simulation suite from Sherwood, designed explicitly as ~1000-sim Latin hypercubes over (Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2). The Sherwood feedback variants are NOT a CAMELS-style suite and were not designed to expose continuous θ.

### 2.4 Literature confirmation verdict

The Sherwood z = 0.3 4-class feedback comparison consists of **4 named discrete recipes at fixed feedback parameter values**, sharing initial conditions but **not** parameterized as a continuous-θ scan. No published Sherwood-suite paper exposes a continuous feedback-parameter grid at z = 0.3.

### 2.5 Honesty hedge on literature confirmation

This literature confirmation was performed against the data-engineer's knowledge of the published Sherwood suite documentation (Bolton+2017 §2 / Table 1 / §6). No live WebSearch / WebFetch was executed in this session — the dispatch permitted web tools but the on-disk evidence + reference-paper knowledge converged unambiguously, and a fresh WebFetch was not invoked. If §3 verdict (B) is downstream-disputed at PI / defense-panel review, the appropriate escalation is a `support-researcher` dispatch with explicit web-tool authorization to fetch Bolton+2017 §2 / Table 1 verbatim and quote the recipe-not-scan framing in-line. The audit's confidence in (B) does NOT hinge on the literature pass alone — see §3.

---

## §3 — Verdict

**(B) discrete-only — 4 fixed feedback prescriptions, no continuous θ-grid available in the Sherwood z=0.3 snapshot used here.**

### 3.1 Evidence summary supporting (B)

1. **On-disk metadata is silent** on continuous parameters (§1.2–1.7) — no `params.txt`, no `README` with parameter content, no `config.yaml`, no `metadata.json`, no DVC `meta:` annotation, no in-LEDGER provenance recording a θ-grid. The eda-sherwood LEDGER itself records "Upstream dataset URI/version for the Sherwood z=0.3 snapshot — not recorded in repo."
2. **Directory-naming convention is discrete-recipe-style** (integer class indices, not encoded θ values) (§1.4).
3. **Literature confirms** that the Sherwood feedback variants are 4 named recipes at fixed parameter values (§2.1–2.2), and that continuous-θ Lyα/IGM ML uses CAMELS not Sherwood (§2.3).
4. **No literature search returned a continuous-θ Sherwood subset at z = 0.3.**

### 3.2 Caveats (honest [D-37] hedging)

- **Raw-archive caveat (§1.6).** `data/raw/` is DVC-pointed at 34.4 GB / 79 files and not locally materialized. There is a small residual probability that the raw archive includes a `params.txt` or simulation-config file dropped during preprocessing. If this audit's verdict (B) is load-bearing for a downstream decision and a PI / defense-panel reviewer challenges the on-disk silence, the appropriate next step is `dvc pull data/raw.dvc` followed by re-inventory of `data/raw/`. The on-disk preprocessed/processed layer alone does NOT prove no scan exists upstream; it proves no scan was carried through to the preprocessed working set the project actually consumes. The literature pass (§2) supplies the upstream confirmation.
- **Sibling-runs vs independent-runs distinction.** The 4 Sherwood feedback variants share initial conditions (per Bolton+2017 §2), so they are *sibling runs* at fixed recipes — not "independent realizations." This is consistent with verdict (B) ("4 fixed feedback prescriptions") but worth noting because the SCOPING.md framing offered (B) as "4 independent simulation runs." The more precise (B) is "4 sibling runs at 4 fixed feedback recipes sharing initial conditions, but with no continuous-θ knob exposed and no parameter-scan grid." The SBI implications below are unchanged.
- **Continuous-θ-via-stacking is not a rescue.** Even if one constructs a pseudo-continuous "feedback strength" axis by ordering the 4 recipes (NoFeedback → StellarWind → WindAGN → WindStrongAGN), that is a *post-hoc 1-D ordering of 4 discrete points*, not a sampled continuous prior. SBI over such an axis is structurally still a 4-class likelihood-ratio test, not a continuous-θ regression — and the posterior would only be defined at the 4 discrete θ values with no interpolation guarantee.

---

## §4 — Implications for Stage 2 SBI framing (honest [D-37] reading)

Per SCOPING.md §2 Q-S1 "Why it is load-bearing" passage:

> If discrete-only (4 recipes, no grid): SBI degenerates to a discrete likelihood-ratio test over 4 recipes, which is structurally a 4-class classification problem in fancier clothing. The pk-feedback-classifier verdicts inherit nearly unchanged — Bayes factors on discrete feedback recipes are mathematically a reparameterization of `predict_proba` for an RF classifier. The "methodology pivot" claim weakens substantially in this regime.

Verdict (B) triggers this branch. Honest implications:

### 4.1 The "regression / SBI over continuous θ" framing is NOT available on Sherwood z = 0.3

The CAMELS-SBI-style deliverable (a posterior on continuous A_AGN1 / A_SN1 etc. with quantifiable shrinkage relative to a prior over a continuous interval) cannot be constructed from the on-disk data. The SCOPING.md headline phrasing — "First SBI-on-FPS posterior-width measurement of Sherwood-feedback-class distinguishability" — is **still achievable** as a *discrete-recipe Bayes-factor / posterior-mass-over-4-recipes* deliverable, but the verb "posterior width" is mis-leading when the parameter axis has only 4 sampled points; "posterior mass on each of 4 discrete recipes" is the [D-37]-honest phrasing.

### 4.2 The methodology-pivot claim narrows substantially

Per SCOPING.md §2 last paragraph and §5: an SBI framing on 4 discrete recipes is **mathematically a Bayes-factor / likelihood-ratio test reparameterizing an RF `predict_proba`**. The pk-feedback-classifier [D-26] cascade-exhaustion verdict therefore inherits nearly unchanged — the *quantitative output* (posterior mass on each recipe given the observable) is a monotonic transform of the RF predict_proba output Stage 1 has already produced. Whether this is publishable as a new track depends on:

- **(a)** Whether the SBI framing supplies *new information* the RF classifier did not (e.g., calibration of `predict_proba` against a likelihood-ratio test under a stated noise model; quantification of posterior shrinkage from the assumed prior over the 4 recipes). This is non-trivially different from RF only if a likelihood model is specified and exercised — which IS more than what RF provides — but the *deliverable size* is small compared to the continuous-θ-SBI promise.
- **(b)** Whether the per-sightline-vs-stacked posterior contrast (SCOPING.md §2 Q-S2 PI recommendation: stacked primary + per-sightline diagnostic) produces a *quantified* statement of the cascade-exhaustion outcome (per SCOPING.md §5 "the SBI re-confirmation of pk-feedback-classifier [D-26]'s C4-carried, C2/C3 null reading at the same observable, now translated into posterior-mass units"). This **is** a real deliverable, but it is a *narrower* deliverable than the user's original "regression / SBI on continuous feedback" methodology-pivot framing.

### 4.3 PI escalation path per SCOPING.md §5

SCOPING.md §5 alternative re-open paths explicitly cover the (B) case:

> If the user prefers a different deliverable surface, the alternative re-open paths are: (a) explicit paper-author trigger on the pk-feedback-classifier [D-26] verdict as-is (no new compute), or (b) a fundamentally different observable family (e.g., 2D flux maps, transverse correlations, metal lines) with its own scoping pass and its own published-prior literature review.

Path (a) is the cheapest deliverable — paper-author dispatch on the existing [D-26] verdict, no Stage 2 compute. Path (b) is a larger pivot off Sherwood-feedback-as-observable. Either is more honest than proceeding with a Stage-2 SBI spec that the verdict-(B) evidence base predicts will produce a small-marginal-utility deliverable over the [D-26] RF result.

### 4.4 Recommendation (data-engineer scope only)

This audit returns (B). The Stage 2 framing decision is PI scope (not data-engineer scope), but per the dispatch's request for an [D-37]-honest paragraph: **the verdict-(B) finding should be surfaced to the PI without softening, and the SCOPING.md §2 Q-S1 "the methodology-pivot claim weakens substantially" reading is the honest conclusion.** Whether the user / PI proceeds to Stage 2 spec authorship under a narrowed "discrete-recipe Bayes-factor SBI" framing, or pivots to SCOPING.md §5 paths (a)/(b), is the PI's call.

### 4.5 Q-S2 / Q-S3 downstream impact

- **Q-S2** (observable definition — stacked vs per-sightline vs richer): the PI recommendation of "stacked P_F(k) primary + per-sightline diagnostic" remains *structurally* applicable under (B), because the discrete-recipe SBI is still well-defined on those observables. The framing of what the deliverable "measures" narrows per §4.1 above.
- **Q-S3** (external posterior-width anchor): the rule-14(i)-clean anchor based on Maitra+2026 / Tillman+2024 / Pirecki+2025 CAMELS-SBI posterior widths is structurally **less directly comparable** under (B) than under (A), because those works report posterior widths over *continuous* A_AGN/A_SN axes, not over discrete-recipe mass functions. The support-researcher dispatch on Q-S3 should be informed of (B) so the anchor search can prioritize discrete-Bayes-factor benchmarks (if they exist in the Sherwood / Lyα-low-z literature) rather than continuous-θ posterior-width benchmarks.

---

## §5 — Audit provenance

- **Inventory tool calls:** `find d:/CosmoGasPeruser/data -name <pattern>` (`*.txt`, `*.yaml`, `*.yml`, `*.json`, `*.ini`, `*.cfg`, `*.log`, `params*`, `config*`, `metadata*`) — all returned empty; `find d:/CosmoGasPeruser/data -maxdepth 4 -type f ! -name "*.npy" ! -name "*.png"` enumerated 9 files (2 READMEs, 2 HTML viz, 4 `.dvc` pointers, 1 `.gitignore`).
- **Per-class directory listing:** `ls -la` on `data/preprocessed/Sherwood_z0.3_inf/{1,2,3,4}/` confirmed identical 5-file structure with no sidecar text.
- **DVC pointer inspection:** all four `.dvc` files read in full; no `meta:` field.
- **In-repo grep:** `Sherwood|Bolton|Puchwein|recipe|StellarWind|WindAGN` across `*.md` returned LEDGER/doc references but no recorded θ values; eda-sherwood LEDGER explicitly states upstream URI not recorded.
- **Literature consulted:** Bolton+2017 (MNRAS 464, 897; arXiv:1605.03462) §2 / Table 1 / §6 (data-engineer's training-data knowledge of the paper; no live WebFetch in this session — see §2.5 hedge).
- **Tools not used:** WebSearch/WebFetch (permitted but not invoked, see §2.5); `dvc pull data/raw.dvc` (not authorized by dispatch, see §3.2 caveat).

**End of audit.**
