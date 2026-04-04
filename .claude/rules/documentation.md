---
description: "Documentation standards to prevent drift and duplication"
globs: ["docs/**/*.md", "README.md", "CLAUDE.md"]
alwaysApply: false
---

# Documentation

## Structure
- Each experiment branch gets `docs/feature/<branch-name>/` with a README explaining what it did, what it found, and how it connects forward.
- `CLAUDE.md` is the single source of truth for architecture, conventions, and current state.
- `.claude/rules/` encode always-follow guidelines. `.claude/skills/` encode workflows.

## Anti-Duplication
- If a summary doc is a strict subset of a full analysis doc, delete the summary.
- Theory sections (e.g., A=1-F rationale, wavelet config) should exist in one place and be cross-referenced, not copy-pasted.
- When a principle is captured in `CLAUDE.md` or `.claude/rules/`, delete the original standalone doc.

## Freshness
- Docs must not reference deleted files or stale paths. After cleanup, grep docs for removed paths.
- Terminology must match code: "separability vector" not "fingerprint", "K=5" not "K=8" (unless describing v1 historical results).
- When moving files between DVC and Git, update any docs that reference the old location.

## Image Paths
- EDA plots: `results/eda/`
- Baseline RF plots: `results/baseline_rf/`
- Signal clustering v2: `results/signal_clustering_v2/figs/`
- Use relative paths from the doc's location.
