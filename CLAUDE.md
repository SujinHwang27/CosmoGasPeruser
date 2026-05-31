# CLAUDE.md

> **At the start of every session, dispatch the `project-architect` (PI) agent for orientation before any other work.** The PI reads the active track's `experiments/<track>/LEDGER.md` (§1 Pulse + §3 current plan-of-record decision), the binding rules in `.claude/agents/project-architect.md`, and reports the current state + next-step authorization gate. Skipping this step risks acting on stale context — sprint state changes per-session; the LEDGER is the only source of truth that survives compaction.

**CosmoGasPeruser**: computational-astrophysics + ML research project that discovers physically meaningful spectral features in diffuse gas from the Sherwood cosmological simulation (z=0.3), by unsupervised clustering of per-sightline **separability vectors** across 4 physics feedback classes (1=NoFeedback, 2=StellarWind, 3=WindAGN, 4=WindStrongAGN). Active track is `signal-clustering-v2` — RBF-SVM micro-probing → 24-dim separability vectors → K-Means (K=5) → UMAP/audit/RF, run in parallel over wavelet and raw representations. Plan-of-record at `experiments/signal-clustering-v2/LEDGER.md` §3.

**Before starting work, read the active branch's LEDGER first**: `experiments/<branch_basename>/LEDGER.md`. It is the single source of truth for stage status, decisions (D-XX), data lineage, and next steps.

## Tooling

- Package manager: **`uv`** (Python 3.12+). Stay on `uv.lock`; install with `uv sync`, run with `uv run python <script>`. Quote version specifiers when adding packages so the shell does not create junk redirect files.
- Run scripts with `PYTHONPATH=.` from the repo root so `src.core.` imports resolve. The pipeline scripts set this; `run_all.py` injects it into subprocesses.
- Upstream raw data under `data/` is DVC-managed and read-only — **never modify files in `data/` directly**. Pull/push through DVC (`data/raw.dvc`, `data/preprocessed.dvc`, `data/processed.dvc`, `data/feature_analysis.dvc`).

## Source layout

All reusable logic lives in `src/core/` and below — **never place modules at `src/` level** (a stray `src/viz.py` once shadowed `src/core/viz.py` and broke imports). Scripts in `scripts/` are thin orchestration wrappers — no logic in scripts.

- `src/core/data.py` — data ingestion (`DataIngestor`, `SignalClusteringData`). Load via `SignalClusteringData`, never raw `np.load()` in scripts.
- `src/core/probe.py` — RBF SVM micro-probing → 24-dim separability vectors.
- `src/core/cluster.py` — K-Means with k-sweep + silhouette analysis.
- `src/core/viz.py` — UMAP, spatial maps, comparison plots.
- `src/core/audit.py`, `src/core/drift_animation.py` — cross-run overlap, variance, attribution, drift animation.
- `src/core/models/rf_classifier.py` — per-cluster Random Forest with GridSearchCV.
- `src/core/base.py` — ABCs (`BaseTransformer`, `BaseModel`). `src/core/transforms.py` — PCA/DCT/Wavelet/Fisher. `src/core/provenance.py` — git-metadata stamping. `src/core/utils.py` — MLflow cleanup, shape checks.
- `scripts/run_{prep,probe,cluster,viz,audit,rf}.py` — stages 1–6. `scripts/run_all.py` — always-re-runs orchestrator.
- `experiments/<name>/LEDGER.md` — command center per track (7-section schema below). The active pipeline entry-point for this project is the DVC DAG (`dvc.yaml`) + `scripts/`, not a per-track `pipeline.py` (project-specific adaptation of the Research-OS template).
- `dvc.yaml` — 6-stage reproducible pipeline. `results/<track>/` — CSVs, plots, figures (git-tracked via `cache: false`).

### Legacy (already removed — do not resurrect)
- An earlier config-driven orchestrator (`src/main.py`), standalone scripts (`scripts/baseline|clustering|utils/`), and `src/core/models.py` (`BaselineRFClassifier`, `MicroProbingClassifier`, `SimpleTransformerClassifier`) were removed in prior cleanup and are **absent on disk** (verified 2026-05-26). Their history is captured in the per-track LEDGERs. The current RF lives at `src/core/models/rf_classifier.py` (a package, not the old `models.py`).
- `configs/*.yaml` and `local/utils.py` are dangling remnants of the removed config-driven flow — no current code references them (candidate orphans; left in place pending owner confirmation).

## Coding conventions

- **ABC pattern**: transforms inherit `BaseTransformer` (`fit_transform(X, y)`); models inherit `BaseModel` (`train(...)`, `predict(X)`). New `src/core/` modules need a docstring stating scientific purpose.
- **Type hints on all public functions.** Every annotated type must be explicitly imported — Python only crashes when the function is *called*, not at import, so CI import-checks alone won't catch a missing `Optional`/`Dict`/`Tuple`. This has bitten the repo; verify typing imports.
- NumPy arrays carry shape comments (`# shape: (n_sightlines, 24)`). Use `np.float64` for scientific computation. Random seeds are explicit parameters, never hardcoded without a default.
- Validate data shapes and NaN/Inf before computation; no silent NaN passthrough — replace with a documented, physically defensible default.

## Terminology (standardized)
- **Separability vector** — the canonical term for the 24-dim feature per sightline (6 one-vs-one class pairs × 4 class decision distances). File outputs use `fingerprints_*.npy` for backward compatibility, but variable names and prose say `separability_vectors`.
- **Sightline** — a single line-of-sight spectrum (2048 pixels).
- Physics classes: `1`=NoFeedback, `2`=StellarWind, `3`=WindAGN, `4`=WindStrongAGN.

## Domain conventions
- Data shapes (Sherwood z=0.3): raw flux `(16384, 2048)` per class; wavelet features `(16384, 2048)` per class (db8, 6 levels, 12 detail; per-level z-score so D5/D6/A6 don't dominate); separability vectors `(16384, 24)`; cluster labels `(16384,)` ∈ `0..K-1`.
- The micro-probing algorithm: per sightline, for each of the 6 one-vs-one class pairs, train an RBF SVM on the 2 class vectors and record signed decision distances from all 4 classes → 24-dim vector. Encodes the per-position separability landscape.
- Every loader/preprocessor includes a `_validate_data` bounds/type/NaN check.

## Experiment workflow (mandatory)
- Each methodology is isolated on its own branch (`exp/<name>` for new tracks; existing tracks keep their `feature/<name>` names) with `experiments/<name>/{LEDGER.md, artifacts/}` and its results under `results/<name>/`.
- MLflow experiment names are branch-aware and hierarchical via `src.core.provenance.mlflow_experiment_name()` → `CosmoGasPeruser/<branch>`. Run names are stage-prefixed `Stage<N>-<ShortDescription>`. Mandatory tags: `model_type`, `stage`, `run` (wavelet/raw), `k`. See `.claude/skills/mlflow-run/SKILL.md`.
- Version-track any artifact > 10 MB or matching `.npy/.pt/.ckpt/.dat/.html/.mp4/.h5/.hdf5`. See `.claude/skills/dvc-track/SKILL.md`.

## Testing
- `uv run pytest tests/`. Test files `tests/test_<module>.py` mirror `src/core/<module>.py`. Use small synthetic arrays with a fixed-seed `np.random.default_rng(seed)` — tests must not depend on real data in `data/`.
- Cover: data loading (shape/dtype/NaN), transforms (shape/range/determinism), clustering (label count == K, all assigned), probe (24-dim, finite, deterministic). Mark slow tests `@pytest.mark.slow`.

## Git conventions
- Branches: `main` (stable), `exp/<name>` (new methodologies), `feat/<name>`, `refactor/<name>`. Existing tracks: `feature/<name>` (kept as-is). Hyphenated names.
- Conventional commits: `feat:`, `fix:`, `chore:`, `paper:`, `docs:`, `refactor:`, `test:`. Reference the pipeline stage when changing stage code. Commit in logical groups.
- **Commit and push proactively, whenever appropriate per git best practice — do NOT wait to be explicitly asked** (overrides the default "commit only when asked" behavior). Commit each logical unit as it completes with a conventional-commit message; push to the remote tracking branch once a coherent unit is done and any relevant tests/lints pass. Guardrails that still hold: never commit secrets (`.env`) or `data/` binaries (DVC-managed); on `main` create a branch first; don't bundle unrelated changes into one commit; don't `--no-verify` or force-push without explicit instruction.
- Tag stable checkpoints: `git tag -a v0.X-name -m "..." <commit>`. Existing: `v0.1-eda`, `v0.2-baseline-rf`, `v0.3-clustering-v1`, `v0.4-clustering-v2`.

### DVC ↔ Git discipline
- `dvc.yaml` dep paths must match disk exactly — DVC silently skips re-execution on a missing dep, it does not error. After moving/renaming source, update `dvc.yaml` deps AND `dvc.lock` (via `dvc repro`). `run_all.py` and `dvc.yaml` must list the same stages.
- Results CSVs/plots use `cache: false` so they are Git-readable without `dvc pull`.
- Migrations: Git→DVC (`dvc add` → `git rm -r --cached` → `git add *.dvc .gitignore`); DVC→Git (`git rm *.dvc` → `git add <actual files>` — **verify the files land in Git**; data was lost here once when a `.dvc` pointer was deleted but files never made it to Git or DVC). Never leave orphaned `.dvc` pointers.

## Documentation
- Each track gets `docs/feature/<branch-name>/` (historical analysis archives) and a live `experiments/<track>/LEDGER.md` (command center). This CLAUDE.md is the single source of truth for architecture/conventions.
- Anti-duplication: if a doc is a strict subset of another, delete the subset. Theory sections live in one place and are cross-referenced. After cleanup, grep docs for removed paths — docs must not reference deleted files or stale paths, and terminology must match code ("separability vector" not "fingerprint"; "K=5" not "K=8" unless describing v1 history).
- Image paths: EDA `results/eda/`, baseline RF `results/baseline_rf/`, clustering-v2 `results/signal_clustering_v2/figs/`.

## Security
- Never commit `.env` or hardcode credentials. Load secrets via `python-dotenv` at runtime. `.env` is gitignored; `.env.example` documents required vars without real values. DVC remote credentials via `dvc remote modify --local`. MLflow tracking URI is configurable (`MLFLOW_TRACKING_URI`), not hardcoded.

## Failure handling
If the same command fails 3 times with no progress, **stop and surface to the user** with exact commands, observed output, hypothesized cause, proposed fix. Don't keep retrying. Don't fabricate an "Error Report" file unless asked.

## Reporting findings (honest-reporting rule, the [D-37] discipline)
Lead with the empirical observation as observed. Framing-for-paper is a separate, downstream call. When a finding could either strengthen or weaken a current paper claim, the first-pass report favors the **honest** framing — the claim narrows to match the evidence unless extra evidence justifies the broader claim. Null results are scientific outcomes, not problems to spin. See `.claude/agents/project-architect.md` for the full [D-37]-extension rules.

## Never recommend unverified external-tool behavior
Never present MLflow URIs, DVC commands, or library APIs as fact without verifying — test it or check official docs first. (The MLflow backend was mis-stated as S3-capable once; it is file/sqlite/postgres/mysql only — see `mlflow-run` skill.)

## Subagents, commands, skills
- `.claude/agents/` — `project-architect` (PI), `data-engineer`, `core-implementer`, `infrastructure-manager`, `support-researcher`, `paper-author`, `defense-panel`. Dispatched by description match.
- `.claude/commands/` — `/new-experiment <name>`, `/update-ledger`.
- `.claude/skills/` — `ledger-update` (LEDGER write contract), `mlflow-run` (MLflow run contract + sync), `dvc-track` (DVC pipeline + heavy-artifact contract), `skill-transplant` (graft a capability from another agentic repo), `juno-hpc` (UTD Juno HPC submission contract — SSH/SLURM/rsync for jobs that exceed the local laptop budget, e.g. the `exp/pk-feedback-classifier` P_F(k) probe).

## Master-source architecture for paper authoring (when a paper track starts)
Multi-venue authoring = one decision-log + one set of atoms + N venue manifests. Source-of-truth order (resolve conflicts up-chain): `experiments/<name>/LEDGER.md` → `papers/shared/numbers.tex` → `papers/shared/sec/*.tex` → `papers/<venue>/main.tex`. The `papers/` tree is not yet created; the `paper-author` agent scaffolds it when paper work begins.
