---
name: data-export
description: The project's canonical outbound data-export contract — how CosmoGasPeruser data leaves the project for external consumers (currently the selements-website primer panels). Cross-cutting infrastructure, NOT part of any experiment track: it lives on the dedicated service/data-export branch and is not assigned experiment LEDGER D-XX numbers. Codifies ownership (data-engineer primary; support-researcher secondary only for new-statistic derivations), the canonical landing path under results/exports/<consumer>/, the src/core/export.py + scripts/export/ + MANIFEST.md pattern, the git-stamped provenance-sidecar requirement, the honest-reporting verb-ceiling gate on any claim-bearing export, and the step-by-step recipe to service a new request. Trigger when the user asks to "export X for the website / for <external consumer>", "ship a CSV / figure / dataset out", "fulfill a selements-website data request", or when wiring a new src/core/export.py function or scripts/export/ wrapper. Do NOT trigger for internal pipeline outputs (those are dvc-track), MLflow runs (mlflow-run), or in-repo result CSVs/plots.
---

# Data-export contract (selements-website)

CosmoGasPeruser data leaves the project through ONE audited boundary:
`src/core/export.py`. Every export carries a no-bit-lost integrity guarantee —
canonical-loader ingestion, shape/finiteness/bounds validation, full-float64
precision (no silent rounding), and a git-stamped provenance sidecar.

## Branch & governance discipline
This export workflow is **cross-cutting infrastructure, not an experiment**.
- All export work is committed on the dedicated **`service/data-export`** branch
  — NOT on `main`, NOT on any `exp/<name>` experiment branch. (Servicing a
  request: `git checkout service/data-export`, do the work, commit, push;
  rebase onto a newer base only if an export needs a loader that postdates the
  branch point.)
- Exports are **not** assigned experiment LEDGER `D-XX` decision numbers and do
  not touch `experiments/<track>/LEDGER.md`. **This skill is the contract of
  record.** Decisions about the export workflow itself are recorded here.

## Ownership
- **Primary owner: `data-engineer`.** All export functions, landing paths, the
  MANIFEST, and this skill are data-engineer-owned.
- **Secondary: `support-researcher`** — ONLY when an export requires deriving a
  *new statistic* (a quantity not already produced by an existing pipeline
  stage). support-researcher derives + validates the statistic; data-engineer
  still owns the export function, validation, landing, and MANIFEST row.

## Canonical landing path
```
results/exports/<consumer>/<request-slug>/
```
The constant lives in code: `src.core.export.SELEMENTS_WEBSITE_ROOT`
(`results/exports/selements-website`). **selements-website is the first (and
currently only) registered consumer.** New consumers get a new subdir + their
own MANIFEST.

## The pattern (always all three)
1. **Logic** — a named, deterministic, re-runnable function in
   `src/core/export.py` (docstring stating scientific purpose; type hints with
   explicitly-imported annotations; np.float64; a `_validate_*` bounds/finiteness
   guard; loads via the canonical `SignalClusteringData` loader, never raw
   `np.load`).
2. **Thin wrapper** — `scripts/export/<request-slug>.py`: argparse (with an
   `--out-dir` defaulting to the canonical landing dir) + a single call into the
   `src/core/export.py` function. NO logic in the script.
3. **Registry row** — append to `results/exports/<consumer>/MANIFEST.md`:
   `request-slug | producing function | source-data path | git SHA at export |
   date | consumer-facing filename | caveats`.

## Provenance-sidecar requirement (mandatory)
Every export writes a sidecar JSON next to the artifact, stamped via
`src.core.provenance.get_git_info()` (git SHA + branch + dirty + timestamp),
plus: source-data path, producing-function name, export timestamp, semantic
labels (class id/label, sightline_idx, Δv, units), source lineage, and an
HONEST selection/limitation caveat. The sidecar makes the exact source array
and code state recoverable from the shipped file alone.

## Honest-reporting verb-ceiling gate (claim-bearing exports)
If an export carries or implies a *scientific claim* (not just raw spectra), the
caveat string and any consumer-facing copy MUST respect the project's
honest-reporting rule (CLAUDE.md) and the scientific verb-ceiling recorded in
`experiments/reframe-suite/CLOSE_OUT.md` §2 /
`experiments/reframe-suite/HARDENING_OUTCOME.md` (cited as the source of what the
project can and cannot claim, not as export-workflow decision coupling):
- C2–C3 is a **"weak, gate-marginal P(k)-shape signal"** — NEVER "feedback
  detected" / "feedback classifier" / "we detect AGN feedback".
- Always flag the **one-realization / fixed-cosmology** limitation.
- Always flag the **60 cMpc/h box vs. Bolton+2017** provenance caveat.
Lead with the empirical observation; the honest framing wins on first pass
(the project honest-reporting rule). A purely descriptive export (e.g. "first
stored sightline, not a typical spectrum") still carries its honest selection
caveat.

## How to service a new selements-website request
1. **Enumerate the source data this session** (data-engineer mandatory trigger):
   `ls -la` the source dir; confirm path, shape, dtype, finiteness, physical
   bounds. Lead with the empirical observation, not a derived conclusion.
2. **Pick a request-slug** (kebab-case, e.g. `primer-synthetic-spectrum`).
3. **Add the logic function** to `src/core/export.py` — canonical loader,
   `_validate_*` guard, float64, full precision, deterministic, type-hinted.
4. **Add the thin wrapper** `scripts/export/<slug>.py` (argparse + one call).
5. **Run it** from repo root: `PYTHONPATH=. uv run python scripts/export/<slug>.py`.
6. **Verify** the artifact lands at the canonical path, row/shape counts match,
   bounds hold, and the sidecar JSON has a real (non-`unknown`) git SHA.
7. **Append the MANIFEST row** with the git SHA at export + the honest caveat.
8. **Add tests** to `tests/test_export.py` (synthetic fixtures for write/validate
   logic; mark any real-`data/` test `@pytest.mark.slow`).
9. If the export is claim-bearing, run it past the **honest-reporting
   verb-ceiling gate** above before shipping copy.

## Registered consumers
- **selements-website** — primer panels. First registered consumer.
  Landing: `results/exports/selements-website/`. Registry:
  `results/exports/selements-website/MANIFEST.md`.

## Anti-patterns
- Logic in `scripts/export/` instead of `src/core/export.py`.
- Raw `np.load()` in the export path (bypasses the canonical loader + validation).
- Shipping without a provenance sidecar, or with `"commit": "unknown"`.
- Rounding floats on write (information loss — use full float64 precision).
- Comment lines in a CSV destined for a comment-unaware website parser.
- A claim-bearing caveat that exceeds the honest-reporting verb-ceiling.
- Forgetting the MANIFEST row (orphan export, no lineage).
