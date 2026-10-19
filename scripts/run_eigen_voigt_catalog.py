"""Stage EV-1: Voigt-profile catalogs for the eigen-voigt-embedding track.

Smoke:  PYTHONPATH=. uv run python scripts/run_eigen_voigt_catalog.py --mode smoke --variant fiducial
Full:   PYTHONPATH=. uv run python scripts/run_eigen_voigt_catalog.py --mode full --variant fiducial
        PYTHONPATH=. uv run python scripts/run_eigen_voigt_catalog.py --mode full --variant taurescaled
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")                       # one BLAS thread per fitting worker

import numpy as np

from src.core.eigen_voigt import N_CLASSES, VARIANTS, load_flux_variant, load_grid, tau_rescale_factors
from src.core.provenance import provenance_header
from src.core.voigt import VoigtFitConfig, fit_catalog

ART = Path("results/eigen_voigt_embedding/artifacts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--n-smoke", type=int, default=256, help="sightline positions per class in smoke mode")
    ap.add_argument("--n-jobs", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wave, vel = load_grid()
    n_total = 16384
    rows = (np.sort(np.random.default_rng(args.seed).choice(n_total, args.n_smoke, replace=False))
            if args.mode == "smoke" else None)
    factors = None
    meta = {"mode": args.mode, "variant": args.variant, "seed": args.seed,
            "config": VoigtFitConfig().__dict__, "provenance": provenance_header()}
    if args.variant == "taurescaled":
        fac_path = ART / "tau_rescale_factors.json"
        if fac_path.exists():
            resc = json.loads(fac_path.read_text())
        else:
            resc = tau_rescale_factors()
            ART.mkdir(parents=True, exist_ok=True)
            fac_path.write_text(json.dumps(resc, indent=2))
        factors = resc["A"]
        meta["tau_rescale"] = resc
        print(f"[ev-1] tau rescale factors A_c = {np.round(factors, 5).tolist()} "
              f"(target <F> = {resc['target_mean_flux']:.6f})", flush=True)

    out_dir = ART / f"catalog_{args.variant}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        np.save(out_dir / "rows.npy", rows)
    for c in range(1, N_CLASSES + 1):
        out = out_dir / f"class{c}.npz"
        if out.exists():
            print(f"[ev-1] {out} exists, skipping", flush=True)
            continue
        t0 = time.time()
        flux = load_flux_variant(args.variant, c, rows=rows, factors=factors)   # shape: (n, 2048)
        cat = fit_catalog(flux, wave, vel, out, n_jobs=args.n_jobs)
        diag = cat["diag"]
        keys = list(cat["diag_keys"])
        conv = diag[:, keys.index("converged")].mean()
        print(f"[ev-1] class {c}: {flux.shape[0]} sightlines, {cat['logN'].size} components, "
              f"converged {conv:.4f}, {time.time() - t0:.0f}s", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
