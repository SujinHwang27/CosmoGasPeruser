# selements-website export registry

Every artifact shipped to the selements-website consumer is registered here.
Owner: **data-engineer** (primary). One row per serviced request. This export
workflow is cross-cutting infrastructure — it lives on the dedicated
`service/data-export` branch and is **not** part of any experiment track or its
LEDGER decision-numbering. See `.claude/skills/data-export/SKILL.md` for the
full contract and the "service a new request" recipe.

| request-slug | producing function | source-data path | git SHA at export | date | consumer-facing filename | caveats |
|---|---|---|---|---|---|---|
| primer-synthetic-spectrum | `src.core.export.export_primer_synthetic_spectrum` | `data/preprocessed/Sherwood_z0.3_inf/1/flux.npy` | 683d26a | 2026-06-16 | `synthetic-spectrum.example.csv` | Selection = first stored sightline of class 1 (NoFeedback), sightline_idx=0 — matches `results/eda/tier1_representative_spectra_2x2.png` top-left panel. NOT a statistically-central / "typical" spectrum. Sherwood (Bolton+2017), z=0.3, 60 cMpc/h box; one realization, fixed cosmology. Δv = 2.636502840371587 km/s/px. |
| exploration-pk-mean-per-class | `src.core.export.export_exploration_pk_mean_per_class` | `data/preprocessed/Sherwood_z0.3_inf/{1,2,3,4}/flux.npy` | 6171316 | 2026-06-17 | `pk-mean-per-class.csv` | **Claim-bearing — PI-framing-checked.** Class-mean P_F(k) ± SEM over N=16,384, single-sourced from `FluxPowerSpectrum(norm="per_sightline")`: δ_F=F/⟨F⟩_los−1, Hann-windowed, P(k)=\|rfft\|²·(Δv/N), k in s/km, 20 log-bins (lowest bin empty → pk_mean=0, flagged by `n_modes_in_bin`). Verb-ceiling (HARDENING_OUTCOME §H1/H2/H3, CLOSE_OUT §1 R9/§2): WEAK gate-marginal **feedback** signal, NOT a per-sightline detection / NOT a classifier; box/cosmology confound CLOSED from disk (2026-06-13), only the 60-cMpc/h-vs-Bolton+2017 citation label unpinned; mean-flux↔shape partially entangled (Spearman ~0.94); C2-C3 weakest; one z=0.3 snapshot, no nuisance marginalization. Full caveat in the provenance sidecar. |
