---
name: mlflow-run
description: The project's canonical MLflow run contract — file-based mlruns/ backend, branch-aware hierarchical experiment naming via src.core.provenance, stage-prefixed run names, mandatory tag set, dotenv credentials, no-op fallback when the tracker is unreachable, and the multi-machine mlruns/ sync handshake (DVC). Trigger when wiring a new tracked run, reviewing one for tag/name compliance, or setting up/syncing MLflow across machines. Do NOT trigger for small hyperparameter tweaks inside an already-compliant run. This is the localized form of the generic experiment-tracking contract (merged with the former mlflow-sync-guide).
---

# MLflow run contract (CosmoGasPeruser)

Every tracked run follows the same shape so the store stays queryable across tracks and stages.

## Backend (settled — do not re-litigate)
- Backend store is the **file-based `mlruns/` directory**. MLflow does NOT support S3 as a *backend* store (only file / sqlite / postgresql / mysql); S3 is only valid for the *artifact* store. The repo tried SQLite (`mlflow.db`) and S3-backend variants and reverted to file-based `mlruns/` — keep everything in one place.
- `.env` sets `MLFLOW_TRACKING_URI=mlruns`. `mlflow.db` in the tree is a stale remnant of the abandoned SQLite backend.
- View UI: `uv run mlflow ui --backend-store-uri mlruns` → http://localhost:5000

## Naming
- **Experiment name** (hierarchical, branch-aware): use `src.core.provenance.mlflow_experiment_name()`, which returns `CosmoGasPeruser/<branch>` (e.g. `CosmoGasPeruser/feature/signal-clustering-v2`). Never hardcode the experiment name.
- **Run name** (stage-prefixed): `Stage<N>-<ShortDescription>`, PascalCase, no spaces (e.g. `Stage2-MicroProbing`, `Stage6-RandomForest`).

## Mandatory tags
Set at run start. The repo's mandatory set:

| Tag | Type | Example |
|---|---|---|
| `model_type` | string | `rbf_svm_probe`, `kmeans`, `random_forest` |
| `stage` | string | `1`–`6` |
| `run` | string | `wavelet` / `raw` (the dual-representation axis) |
| `k` | string | `5` (cluster count, where applicable) |

## Boilerplate
```python
import os
from contextlib import nullcontext
from dotenv import load_dotenv
from src.core.provenance import mlflow_experiment_name

load_dotenv()  # loads MLFLOW_TRACKING_URI=mlruns and any credentials

try:
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(mlflow_experiment_name())   # CosmoGasPeruser/<branch>
    run_ctx = mlflow.start_run(run_name=f"Stage{STAGE}-{DESCRIPTION}")
    tracker_active = True
except Exception as e:
    print(f"[mlflow] unreachable, falling back to nullcontext: {e}")
    run_ctx, tracker_active = nullcontext(), False

with run_ctx as run:
    if tracker_active and hasattr(run, "info"):
        mlflow.set_tags({"model_type": MODEL_TYPE, "stage": STAGE, "run": RUN, "k": K})
    # ... training / evaluation; mlflow.log_metric(...), mlflow.log_artifact(...)
```

## Provenance stamp
Stamp results with git metadata via `src.core.provenance.provenance_header(params)` (commit SHA, branch, dirty flag, timestamp + params) so every artifact is traceable to the run that produced it.

## Post-run
Capture the `run_id` and append it to the active track's `experiments/<track>/LEDGER.md` §6 (Visualization & Artifacts) via the `ledger-update` skill — this is what makes a run discoverable later.

## Multi-machine sync (mlruns/ via DVC)
`mlruns/` is gitignored but DVC-tracked through `mlruns.dvc`.

```bash
# After running experiments (Machine A):
dvc add mlruns && git add mlruns.dvc
git commit -m "chore: sync mlflow experiment metadata"
dvc push && git push

# Before reviewing (Machine B):
git pull && dvc pull mlruns
uv run mlflow ui --backend-store-uri mlruns
```

## Anti-patterns
- Hardcoding the experiment name instead of `mlflow_experiment_name()` → breaks branch isolation.
- Run name without a `Stage<N>-` prefix → blocks chronological filtering.
- Skipping any mandatory tag → orphans the run from LEDGER §6 cross-referencing.
- Letting an unreachable tracker crash the script → always wrap in the try/except + `nullcontext` fallback (silent code-0 exits in this repo trace to an import-time hang or unreachable tracker).
- Re-introducing a SQLite/S3 *backend* store → settled as file-based `mlruns/`.
