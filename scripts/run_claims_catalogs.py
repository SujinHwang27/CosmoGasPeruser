"""lit-claims-audit: Voigt refits of tau-rescaled spectra (T1/T3 matched variants, nulls, tol sweep).

PYTHONPATH=. uv run python scripts/run_claims_catalogs.py --name dndz12 --factors-key dndz12
PYTHONPATH=. uv run python scripts/run_claims_catalogs.py --name null2_dndz12 --A 1,0.8204,1,1 --classes 2
PYTHONPATH=. uv run python scripts/run_claims_catalogs.py --name tol005 --A 1,1,1,1 --classes 2,3 --tol 0.005
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from src.core.eigen_voigt import load_flux_variant, load_grid
from src.core.provenance import provenance_header
from src.core.voigt import VoigtFitConfig, fit_catalog

OUT = Path("results/lit_claims_audit/artifacts")
FID = Path("results/eigen_voigt_embedding/artifacts/catalog_fiducial_full")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--factors-key", default=None, help="key in results/lit_claims_audit/matching_factors.json")
    ap.add_argument("--A", default=None, help="comma-separated per-class tau multipliers (classes 1-4)")
    ap.add_argument("--classes", default="1,2,3,4")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--n-jobs", type=int, default=9)
    args = ap.parse_args()
    if args.factors_key:
        A = json.loads(Path("results/lit_claims_audit/matching_factors.json").read_text())[args.factors_key]["A"]
    else:
        A = [float(a) for a in args.A.split(",")]
    cfg = VoigtFitConfig(tol=args.tol, region_thresh=min(0.005, args.tol / 2))
    wave, vel = load_grid()
    out_dir = OUT / f"catalog_{args.name}_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"A": A, "classes": args.classes, "config": cfg.__dict__,
                                                   "provenance": provenance_header()}, indent=2, default=str))
    for c in (int(x) for x in args.classes.split(",")):
        out = out_dir / f"class{c}.npz"
        if out.exists():
            print(f"[claims-cat] {out} exists, skipping", flush=True)
            continue
        if abs(A[c - 1] - 1.0) < 1e-12 and args.tol == 0.01:
            shutil.copy(FID / f"class{c}.npz", out)          # identical to the fiducial fit
            print(f"[claims-cat] {args.name} class {c}: A=1, copied fiducial", flush=True)
            continue
        t0 = time.time()
        flux = load_flux_variant("taurescaled", c, factors=A)
        cat = fit_catalog(flux, wave, vel, out, config=cfg, n_jobs=args.n_jobs)
        conv = cat["diag"][:, list(cat["diag_keys"]).index("converged")].mean()
        print(f"[claims-cat] {args.name} class {c}: A={A[c - 1]:.4f}, {cat['logN'].size} comps, "
              f"converged {conv:.4f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
