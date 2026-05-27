"""Experiment-runner skeleton (CosmoGasPeruser).

Replace this docstring and the placeholders with your track's specifics.
Honors the project's MLflow run contract — see `.claude/skills/mlflow-run/SKILL.md`.

Note: established tracks drive computation through the DVC DAG (`dvc.yaml`) +
`scripts/run_<stage>.py`. This skeleton is for a new exploratory track before
its stages are wired into `dvc.yaml`. Run with `PYTHONPATH=. uv run python ...`.
"""
import argparse
import os
from contextlib import nullcontext

from dotenv import load_dotenv

load_dotenv()  # loads MLFLOW_TRACKING_URI=mlruns and any credentials


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, required=True, help="Stage tag, e.g. '1', '2', '3'.")
    p.add_argument("--description", type=str, required=True, help="Short PascalCase run name.")
    p.add_argument("--seed", type=int, default=42)
    # Mandatory tag axes (see the mlflow-run contract):
    p.add_argument("--run", type=str, default="wavelet", choices=["wavelet", "raw", "both"],
                   help="Data-representation axis.")
    p.add_argument("--k", type=str, default="5", help="Cluster count, where applicable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ----- Tracker setup with no-op fallback ----------------------------------
    try:
        import mlflow
        from src.core.provenance import mlflow_experiment_name
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment(mlflow_experiment_name())  # CosmoGasPeruser/<branch>
        run_ctx = mlflow.start_run(run_name=f"Stage{args.stage}-{args.description}")
        tracker_active = True
    except Exception as e:
        print(f"[mlflow] unreachable, falling back to nullcontext: {e}")
        run_ctx = nullcontext()
        tracker_active = False

    with run_ctx as run:
        if tracker_active and hasattr(run, "info"):
            import mlflow
            mlflow.set_tags({
                "model_type": "<model-shortname>",  # e.g. kmeans, random_forest
                "stage": args.stage,
                "run": args.run,
                "k": args.k,
            })
            mlflow.log_params({"seed": args.seed})

        # ----- Replace the body below with the actual experiment loop ---------
        print(f"Stage{args.stage}-{args.description} starting (run={args.run}, k={args.k})")
        # ... data loading via src.core.data.SignalClusteringData, computation, evaluation
        # ... log metrics: mlflow.log_metric("silhouette", value)


if __name__ == "__main__":
    main()
