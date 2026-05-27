---
name: dvc-track
description: The project's canonical DVC contract — covers BOTH the 6-stage reproducible pipeline (dvc.yaml / dvc repro) AND heavy-artifact versioning (dvc add / push of binaries >10 MB or matching the tracked-extension set). Trigger when adding/modifying a pipeline stage, debugging reproducibility, or version-tracking a new binary output (.npy, .pt, .ckpt, .dat, .html, .mp4, .h5, .hdf5) — model checkpoints, separability-vector arrays, label/centroid arrays, visualization HTMLs, the drift animation. Also covers DVC↔Git migrations and the mlruns/ sync handshake. Do NOT trigger for source code, configs (.yaml/.toml), markdown, or small CSVs (<1 MB) — those are git-tracked. This is the localized form of the generic artifact-versioning contract.
---

# DVC contract (pipeline + heavy artifacts)

DVC plays two roles in this repo. Both live here so the contract stays in one place.
Run every `dvc` command from the repo root so `.dvc/config` is picked up.

---

## Role 1 — The reproducible pipeline (`dvc.yaml`)

The signal-clustering pipeline is a 6-stage DAG defined in `dvc.yaml`:

| Stage | `cmd` script | Core module | Produces |
|---|---|---|---|
| **prepare** | `scripts/run_prep.py` | `src/core/data.py` | `data/feature_discovery/wavelet_class{1..4}.npy` |
| **probe** | `scripts/run_probe.py --run both` | `src/core/probe.py` | `data/feature_discovery/fingerprints_{wavelet,raw}.npy` (24-dim separability vectors) |
| **cluster** | `scripts/run_cluster.py --run both --k 5` | `src/core/cluster.py` | `labels_*_k5.npy`, `centroids_*_k5.npy` + K-sweep CSVs |
| **viz** | `scripts/run_viz.py --run both --view all --k 5` | `src/core/viz.py` | UMAP 2D/3D, spatial maps, comparison plots |
| **audit** | `scripts/run_audit.py --k 5` | `src/core/audit.py`, `src/core/drift_animation.py` | cross-run contingency/overlap CSVs, drift animation |
| **rf** | `scripts/run_rf.py --run both --k 5` | `src/core/models/rf_classifier.py` | per-cluster RF summary CSVs |

```bash
dvc repro            # reproduce full pipeline (skips unchanged stages)
dvc repro <stage>    # run one stage (e.g. dvc repro cluster)
dvc status           # what is stale / out of sync
dvc dag              # visualize the DAG
```

### Adding or modifying a stage
1. Put the logic in `src/core/<module>.py` (never in the script — scripts are thin wrappers).
2. Add a thin `scripts/run_<stage>.py`.
3. Add the stage to `dvc.yaml` with exact `deps`, `outs`, `metrics`/`plots`.
4. **`deps` paths must match disk exactly** — `src/cluster.py` ≠ `src/core/cluster.py`. DVC does *not* error on a missing dep; it silently skips re-execution.
5. Mirror the stage in `scripts/run_all.py` (the always-re-runs orchestrator). `run_all.py` and `dvc.yaml` must list the same stages.
6. `dvc repro <stage>` to regenerate `dvc.lock`.

### cache:false rule
Small result CSVs and plots use `cache: false` in `dvc.yaml` so they are **readable in Git without `dvc pull`** (the `results/signal_clustering_v2/*.csv` metrics and the `figs/*` plots). Large `.npy`/`.html` outputs use default caching.

---

## Role 2 — Heavy-artifact versioning (`dvc add`)

Track an artifact under DVC if **either**: size > 10 MB, OR extension ∈ {`.npy`, `.pt`, `.ckpt`, `.dat`, `.html`, `.mp4`, `.h5`, `.hdf5`}.

Do **not** version-track: source code, configs (`.yaml`/`.toml`), markdown, small CSVs (<1 MB), or MLflow's `mlruns/` store via this manual path (it has its own sync handshake below).

```bash
uv run dvc add <relative/path/to/artifact>      # 1. produces <path>.dvc pointer (small, git-tracked)
git add <relative/path/to/artifact>.dvc .gitignore  # 2. stage pointer + ignore update
uv run dvc push <relative/path/to/artifact>.dvc  # 3. push blob to remote
uv run dvc status                                # 4. "Cache and remote are in sync"
```

**Project data exception**: the upstream raw datasets under `data/` are managed by directory-level `.dvc` pointers (`data/raw.dvc`, `data/preprocessed.dvc`, `data/processed.dvc`, `data/feature_analysis.dvc`). Never modify files in `data/` directly — pull/push through DVC.

After tracking a new artifact, record its lineage with the `ledger-update` skill: append a row to the active track's `experiments/<track>/LEDGER.md` §4 (shape / setting / run_id / version hash).

---

## DVC ↔ Git migrations (data was lost here once — follow exactly)

- **Git → DVC**: `dvc add <dir>` → `git rm -r --cached <dir>` → `git add <dir>.dvc .gitignore`. Verify with `dvc status` and that the blob pushed.
- **DVC → Git**: `git rm <file>.dvc` → `git add <actual files>`. **Verify the actual files exist in Git** — this repo lost data once when a `.dvc` pointer was deleted but the files never landed in Git or DVC.
- Never leave orphaned `.dvc` pointer files — they show up in every `git status`.

## mlruns/ multi-machine sync handshake
`mlruns/` is gitignored but DVC-tracked via `mlruns.dvc`. See the `mlflow-run` skill for the full push/pull handshake — keep MLflow sync logic there, not duplicated here.

## Anti-patterns
- Committing a >10 MB binary directly to git → bloats history.
- `deps` path in `dvc.yaml` not matching disk → silent skip, stale outputs.
- `dvc.yaml` and `run_all.py` stage lists drifting apart.
- Forgetting `dvc push` after `dvc add` → pointer in git, blob only on your machine; collaborators get cache-miss.
- Version-tracking a small CSV/config → wasted indirection; just commit it.
