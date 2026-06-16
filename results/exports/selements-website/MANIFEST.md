# selements-website export registry

[D-31] data-export contract — every artifact shipped to the selements-website
consumer is registered here. Owner: **data-engineer** (primary). One row per
serviced request. See `.claude/skills/data-export/SKILL.md` for the full
contract and the "service a new request" recipe.

| request-slug | producing function | source-data path | git SHA at export | date | consumer-facing filename | caveats |
|---|---|---|---|---|---|---|
| primer-synthetic-spectrum | `src.core.export.export_primer_synthetic_spectrum` | `data/preprocessed/Sherwood_z0.3_inf/1/flux.npy` | 683d26a | 2026-06-16 | `synthetic-spectrum.example.csv` | Selection = first stored sightline of class 1 (NoFeedback), sightline_idx=0 — matches `results/eda/tier1_representative_spectra_2x2.png` top-left panel. NOT a statistically-central / "typical" spectrum. Sherwood (Bolton+2017), z=0.3, 60 cMpc/h box; one realization, fixed cosmology. Δv = 2.636502840371587 km/s/px. |
