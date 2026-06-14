# papers/ — master-source paper tree

This directory holds the multi-venue paper authoring tree for CosmoGasPeruser,
following the master-source architecture in `CLAUDE.md` ("Master-source
architecture for paper authoring").

## Layout

```
papers/
  README.md                  <- this file
  shared/                    <- venue-independent canonical content
    numbers.tex              <- ALL numbers-of-record as \newcommand macros
    main.bib                 <- bibliography (corrected citations)
    sec/                     <- the section atoms (\input'd by venue manifests)
      0_abstract.tex
      1_intro.tex
      2_data.tex
      3_methods.tex
      4_results.tex
      5_discussion.tex
      6_conclusions.tex
    figures/                 <- figure assets (rendered separately by support-researcher)
  mnras/
    main.tex                 <- MNRAS venue manifest (\input's the shared atoms)
```

## Source-of-truth order (resolve conflicts up-chain)

1. `experiments/<track>/LEDGER.md` (+ `HARDENING_OUTCOME.md` for reframe-suite)
2. `papers/shared/numbers.tex`
3. `papers/shared/sec/*.tex`
4. `papers/<venue>/main.tex`

The authoritative current state is
`experiments/reframe-suite/HARDENING_OUTCOME.md` (decision date 2026-06-12);
it SUPERSEDES the older `experiments/reframe-suite/CLOSE_OUT.md` numbers and the
4-class sub-block lattice estimates. All lattice / detection / M-scaling numbers
in `numbers.tex` are the **dedicated-binary hardening numbers**
(`results/reframe_suite/hardening/*.csv`, 10-seed x 5-fold, per_sightline,
p16 = 16th percentile of the 50 seed-fold values).

## Numbers discipline

Bare numerals in result-claim sentences are forbidden. Every quantitative claim
cites a `\newcommand` macro from `numbers.tex`, which in turn names its source
CSV in a trailing comment. No number in this tree is invented; where a number is
not yet in-repo it is a visible `% TODO` placeholder.

## Compiling

The MNRAS manifest expects the `mnras.cls` class file. It is **not bundled** in
this repo. Until it is added, `papers/mnras/main.tex` falls back to the standard
`article` class via a documented `\documentclass` switch comment at the top of
the manifest; the section atoms compile under either class.

## OPEN TODOs (must resolve before submission)

- **`% TODO(box-citation)` — the load-bearing open item.** The four feedback
  classes are the packaged Sherwood `Physics1-4` feedback-variant set, with
  on-disk-confirmed matched box (60 cMpc/h), cosmology
  (Omega_m=0.308, Omega_b=0.0482, h=0.678), redshift (z=0.3), and IC geometry.
  But the measured 60 cMpc/h box does NOT match Bolton et al. (2017)'s canonical
  40 / 80 / 160 cMpc/h Sherwood boxes. The exact Sherwood run + the citation for
  it must be reconciled before submission. The data section
  (`sec/2_data.tex`) carries this as a marked placeholder rather than asserting a
  specific run. See `experiments/reframe-suite/HARDENING_OUTCOME.md` H2 caveat +
  `experiments/pk-feedback-classifier/LEDGER.md` Section 2 ("CONFIRMED FROM DISK
  2026-06-13").
- **Figures.** `papers/shared/figures/` is empty; all `\includegraphics` are
  commented out behind `% TODO(figure)` placeholders. The figure PNGs/PDFs are
  rendered separately by support-researcher from the hardening CSVs (each
  caption names the source CSV). Do NOT `\includegraphics` a file that is not on
  disk.
- **`mnras.cls`** not present; `article` fallback documented in the manifest.

## Authoring provenance

First draft authored 2026-06-13 (paper-author), user-triggered. The
paper-trigger gate (user-triggered-only IFF solid/successful) was cleared by the
completed reframe-suite defense-panel hardening sprint
(`HARDENING_OUTCOME.md`, decision date 2026-06-12).
