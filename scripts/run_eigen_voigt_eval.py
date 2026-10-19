"""Stage EV-2: eigenvector-embedding evaluation for the eigen-voigt-embedding track.

Smoke:  PYTHONPATH=. uv run python scripts/run_eigen_voigt_eval.py --mode smoke --variant fiducial --seeds 2
Full:   PYTHONPATH=. uv run python scripts/run_eigen_voigt_eval.py --mode full --variant fiducial
        PYTHONPATH=. uv run python scripts/run_eigen_voigt_eval.py --mode full --variant taurescaled
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from src.core.eigen_eval import cluster_diagnostics
from src.core.eigen_voigt import VARIANTS, build_representations, run_evaluation, summarise
from src.core.provenance import provenance_header

ART = Path("results/eigen_voigt_embedding/artifacts")
RES = Path("results/eigen_voigt_embedding")
V2_LABELS = Path("data/feature_discovery/labels_raw_k5.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-clusters", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    cat_dir = ART / f"catalog_{args.variant}_{args.mode}"
    rows = np.load(cat_dir / "rows.npy") if args.mode == "smoke" else None
    factors = None
    if args.variant == "taurescaled":
        factors = json.loads((ART / "tau_rescale_factors.json").read_text())["A"]
    out_dir = RES / f"ev2_{args.variant}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"args": vars(args), "provenance": provenance_header(),
                                                   "tau_rescale_A": factors}, indent=2, default=str))

    log(f"building representations ({args.variant}, {args.mode})")
    data = build_representations(args.variant, cat_dir, rows, factors, log=log)
    seeds = list(range(args.seeds))
    run_evaluation(data, out_dir, seeds, n_folds=args.folds, n_jobs=args.n_jobs, log=log)
    summarise(data, out_dir, seeds, log=log)

    if not args.skip_clusters:
        v2 = np.load(V2_LABELS)
        if rows is not None:
            v2 = v2[rows]
        crow, extra = cluster_diagnostics(data["reps"], data["scalars"], data["y"], data["skewers"],
                                          data["blocks"], v2, log=log)
        pd.DataFrame(crow).to_csv(out_dir / "cluster_diagnostics.csv", index=False)
        (out_dir / "cluster_extra.json").write_text(json.dumps(extra, indent=2))
    log("done")


if __name__ == "__main__":
    main()
