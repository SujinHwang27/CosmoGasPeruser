---
description: "Git and DVC workflow rules"
globs: ["dvc.yaml", "dvc.lock", "*.dvc", ".gitignore"]
alwaysApply: false
---

# Git & DVC Workflow

## Commits
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Reference pipeline stage in commit message when changing stage code.

## DVC Pipeline
- `dvc.yaml` dependency paths must exactly match file locations on disk. DVC does not error on missing deps — it silently skips re-execution.
- After moving/renaming source files, always update `dvc.yaml` deps AND `dvc.lock`.
- Run `dvc repro` to verify pipeline integrity after structural changes.
- Results CSVs use `cache: false` — they are small and should be Git-readable.
- `run_all.py` and `dvc.yaml` must have the same stages. When adding/removing a stage, update both.

## DVC↔Git Migrations
- When moving files from DVC to Git: `git rm <file>.dvc`, then `git add <actual files>`. Verify the actual files exist in Git — data was lost in this repo when `.dvc` was deleted but files never made it to Git or DVC.
- When moving files from Git to DVC: `dvc add <dir>`, then `git rm -r <dir>`, then `git add <dir>.dvc`. Verify with `dvc status`.
- Never leave orphaned `.dvc` pointer files — they show up in every `git status` and confuse the state.

## Branches
- `main` — stable baseline
- `feature/*` — active development
- Merge via PR after pipeline passes `dvc repro` cleanly.

## Secrets
- Never commit `.env`, credentials, or API keys.
- Use `.env.example` as a template for required environment variables.
