"""lit-claims-audit T2: forward-modelled (COS-like) Voigt catalogs + injection-recovery.

PYTHONPATH=. uv run python scripts/run_claims_t2.py --name dndz12_snr30_s0 --A-key dndz12 --snr 30 --seed 0
PYTHONPATH=. uv run python scripts/run_claims_t2.py --inject --snrs inf,10,20,30,50
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from src.core.eigen_voigt import load_flux_variant, load_grid
from src.core.forward import (ForwardModel, NoisyFitConfig, NoisyVoigtFitter, ObsConfig, fit_noisy_catalog,
                              gaussian_lsf, tabulated_lsf)
from src.core.provenance import provenance_header
from src.core.voigt import LYA_REST_A, tau_voigt

OUT = Path("results/lit_claims_audit/artifacts")
COS_LSF = Path("data/raw/Sherwood/COS_LSF1_cw1327_75_z0.100.dat")
FID = Path("results/eigen_voigt_embedding/artifacts/catalog_fiducial_full")


def get_lsf(kind: str, dv: float) -> np.ndarray:
    return tabulated_lsf(COS_LSF, dv) if kind == "cos" else gaussian_lsf(17.0, dv)


def run_catalog(args, wave, dv) -> None:
    A = json.loads(Path("results/lit_claims_audit/matching_factors.json").read_text())[args.A_key]["A"]
    if args.A_from_class:                                   # rescaling-only null: class c refitted at another class's A
        A = [A[args.A_from_class - 1]] * 4
    ocfg = ObsConfig(snr=float(args.snr), noise=args.noise, b_turb_max=args.bturb,
                     cont_sigma=0.0 if not np.isfinite(float(args.snr)) else args.cont)
    fcfg = NoisyFitConfig(d_aicc=args.daicc)
    lsf = get_lsf(args.lsf, dv)
    out_dir = OUT / f"noisy_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"A_key": args.A_key, "A": A, "obs": ocfg.__dict__,
                                                   "fit": fcfg.__dict__, "lsf": args.lsf, "seed": args.seed,
                                                   "lsf_kernel_px": int(lsf.size),
                                                   "provenance": provenance_header()}, indent=2, default=str))
    for c in (int(x) for x in args.classes.split(",")):
        out = out_dir / f"class{c}.npz"
        if out.exists():
            continue
        t0 = time.time()
        flux = load_flux_variant("taurescaled", c, factors=A)
        fit_noisy_catalog(flux, wave, dv, lsf, ocfg, fcfg, noise_seed=1000 * args.seed + c, out_path=out,
                          n_jobs=args.n_jobs)
        print(f"[t2] {args.name} class {c} done in {time.time() - t0:.0f}s", flush=True)


def run_injection(args, wave, dv) -> None:
    """Single lines drawn from each class's own fiducial (b, log N), 12.5 <= log N <= 14.5,
    forward-modelled and fitted; b bias and completeness per class and S/N."""
    rng = np.random.default_rng(7)
    dlnw = np.log(wave[1] / wave[0])
    lsf = get_lsf(args.lsf, dv)
    res = {}
    for snr_s in args.snrs.split(","):
        snr = float(snr_s)
        fm = ForwardModel(wave, dv, lsf, ObsConfig(snr=snr, cont_sigma=0.0 if not np.isfinite(snr) else args.cont))
        fit = NoisyVoigtFitter(fm)
        blank = [fit.fit(fm.observe(np.ones(wave.size), rng))[0] for _ in range(args.n_inject)]
        bl = np.vstack(blank)
        res[f"snr{snr_s}|blank"] = {"false_per_sightline": float(len(bl) / len(blank)),
                                    "false_per_sightline_logN13": float((bl[:, 2] >= 13.0).sum() / len(blank)),
                                    "false_per_sightline_logN12p5": float((bl[:, 2] >= 12.5).sum() / len(blank)),
                                    "median_b": float(np.median(bl[:, 1])) if len(bl) else None}
        print(f"[t2-inject] snr {snr_s} blank: {res[f'snr{snr_s}|blank']}", flush=True)
        for c in range(1, 5):
            cat = np.load(FID / f"class{c}.npz")
            m = (cat["logN"] >= 12.5) & (cat["logN"] <= 14.5)
            pick = rng.choice(np.flatnonzero(m), args.n_inject, replace=False)
            rows = []
            for i in pick:
                b, N = float(cat["b_kms"][i]), float(cat["logN"][i])
                v = rng.uniform(300, 5000)
                z = wave[0] * np.exp(v / dv * dlnw) / LYA_REST_A - 1
                comps, _ = fit.fit(fm.observe(np.exp(-tau_voigt(wave, z, b, N)), rng))
                if len(comps):
                    d = np.abs((comps[:, 0] - v + 2700) % (2048 * dv) - 2700)
                    k = int(np.argmin(d))
                    rows.append((b, N, comps[k, 1] if d[k] < 20 else np.nan, comps[k, 2] if d[k] < 20 else np.nan))
                else:
                    rows.append((b, N, np.nan, np.nan))
            r = np.array(rows)
            ok = np.isfinite(r[:, 2])
            entry = {"completeness": float(ok.mean()), "median_db": float(np.median(r[ok, 2] - r[ok, 0])),
                     "mad_db": float(np.median(np.abs(r[ok, 2] - r[ok, 0] - np.median(r[ok, 2] - r[ok, 0])))),
                     "median_dlogN": float(np.median(r[ok, 3] - r[ok, 1]))}
            for lo, hi in ((12.5, 13.0), (13.0, 13.5), (13.5, 14.5)):
                mm = (r[:, 1] >= lo) & (r[:, 1] < hi)
                if mm.sum() >= 10:
                    entry[f"N{lo}-{hi}"] = {"completeness": float(np.mean(ok[mm])),
                                            "median_db": float(np.nanmedian(r[mm, 2] - r[mm, 0]))}
            res[f"snr{snr_s}|class{c}"] = entry
            print(f"[t2-inject] snr {snr_s} class {c}: {entry['completeness']:.3f} complete, "
                  f"db {entry['median_db']:+.2f}", flush=True)
    Path("results/lit_claims_audit").mkdir(parents=True, exist_ok=True)
    Path(f"results/lit_claims_audit/t2_injection_{args.lsf}.json").write_text(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--A-key", dest="A_key", default="dndz12")
    ap.add_argument("--snr", default="30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lsf", choices=("cos", "gauss"), default="cos")
    ap.add_argument("--noise", choices=("const", "sqrtF"), default="const")
    ap.add_argument("--cont", type=float, default=0.01)
    ap.add_argument("--daicc", type=float, default=-2.0)
    ap.add_argument("--bturb", type=float, default=0.0)
    ap.add_argument("--A-from-class", dest="A_from_class", type=int, default=0)
    ap.add_argument("--classes", default="1,2,3,4")
    ap.add_argument("--n-jobs", type=int, default=9)
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--snrs", default="inf,10,20,30,50")
    ap.add_argument("--n-inject", type=int, default=400)
    args = ap.parse_args()
    wave, vel = load_grid()
    dv = float(vel[1] - vel[0])
    if args.inject:
        run_injection(args, wave, dv)
    else:
        run_catalog(args, wave, dv)


if __name__ == "__main__":
    main()
