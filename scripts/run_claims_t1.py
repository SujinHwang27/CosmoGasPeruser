"""lit-claims-audit T1 (+T3 factors): matched noiseless characterisation of the 2-3 and 4-1 width signal.

PYTHONPATH=. uv run python scripts/run_claims_t1.py
"""

import json
import time
from pathlib import Path

import numpy as np

from src.core.claims import (N_BINS_T1, BlockBootstrap, binary_cv, binned_dmedb, load_catalogs, predicted_matched_db,
                             skewer_blocks)
from src.core.eigen_voigt import FLUX_BASE, load_flux_variant, scalar_baselines
from src.core.provenance import provenance_header

OUT = Path("results/lit_claims_audit")
FID = Path("results/eigen_voigt_embedding/artifacts/catalog_fiducial_full")
TAU = Path("results/eigen_voigt_embedding/artifacts/catalog_taurescaled_full")
ART = OUT / "artifacts"
SEEDS = list(range(10))


def cat_dir(name: str) -> Path:
    return {"fiducial": FID, "meanF": TAU}.get(name, ART / f"catalog_{name}_full")


def main() -> None:
    t0 = time.time()
    log = lambda m: print(f"[t1 {time.time() - t0:6.0f}s] {m}", flush=True)
    factors = json.loads((OUT / "matching_factors.json").read_text())
    blocks = skewer_blocks()
    bb = BlockBootstrap(blocks, n_boot=500, seed=0)
    fid = load_catalogs(FID)
    res: dict = {"provenance": provenance_header(), "A_ratios": {}, "ensemble": {}, "binary": {}}
    for k, f in factors.items():
        A = f["A"]
        res["A_ratios"][k] = {"A": A, "A3/A2": A[2] / A[1], "A4/A1": A[3] / A[0], "A4/A3": A[3] / A[2]}

    # ensemble primaries: binned median-b differences, measured vs predicted from the fiducial catalog
    for name in ("fiducial", "meanF", "dndz12", "dndz13", "dndz13p5", "cddf"):
        cats = load_catalogs(cat_dir(name))
        A = factors[name]["A"] if name in factors else [1.0] * 4
        ent = {}
        for a, b in ((2, 3), (1, 4)):
            ent[f"{b}-{a}"] = {"measured": binned_dmedb(cats, bb, a, b),
                               "predicted_from_fiducial_shift": predicted_matched_db(fid, A, a, b, N_BINS_T1)}
        res["ensemble"][name] = ent
        log(f"ensemble {name} done")

    def run_bin(key, ca, cb, **kw):
        r = binary_cv(ca, cb, blocks, SEEDS, **kw)
        res["binary"][key] = {k: v for k, v in r.items() if k != "per_split"}
        res["binary"][key]["per_split"] = r["per_split"]
        log(f"binary {key}: RF {r['ba_rf']['value']:.4f} [{r['ba_rf']['ci95_lo']:.4f},{r['ba_rf']['ci95_hi']:.4f}] "
            f"LR {r['ba_logreg']['value']:.4f}")
        (OUT / "t1_results.json").write_text(json.dumps(res, indent=2, default=float))

    for name in ("fiducial", "dndz12"):
        cats = load_catalogs(cat_dir(name))
        for fs in ("all", "b_only", "n_only", "b_resid"):
            run_bin(f"{name}|23|{fs}", cats[1], cats[2], feature_set=fs)
        run_bin(f"{name}|14|all", cats[0], cats[3], feature_set="all")
    for name in ("meanF", "dndz13", "dndz13p5", "cddf"):
        cats = load_catalogs(cat_dir(name))
        run_bin(f"{name}|23|all", cats[1], cats[2], feature_set="all")
        run_bin(f"{name}|14|all", cats[0], cats[3], feature_set="all")
    d12 = load_catalogs(cat_dir("dndz12"))
    for det in (12.5, 13.0):
        run_bin(f"dndz12|23|all|detect{det}", d12[1], d12[2], feature_set="all", logN_detect=det)
    # residualised on [<F>, n_lines(>=13), <tau>_clip] of the dndz12-matched spectra
    A = factors["dndz12"]["A"]
    sc = []
    for c, cat in ((2, d12[1]), (3, d12[2])):
        F = load_flux_variant("taurescaled", c, factors=A)
        s = scalar_baselines(F)
        n13 = np.bincount(cat["sightline"][cat["logN"] >= 13.0], minlength=F.shape[0]).astype(float)
        sc.append(np.column_stack([s[:, 0], n13, s[:, 1]]))
    run_bin("dndz12|23|all|resid_F_n13_tau", d12[1], d12[2], feature_set="all", scalars=np.vstack(sc))
    axis = np.load(Path(FLUX_BASE) / "1" / "axis.npy")[:, 0].astype(int)
    run_bin("dndz12|23|all|axis_held_out", d12[1], d12[2], feature_set="all", axis_ids=axis)
    # rescaling-only null: class 2 at A2 vs class 2 refitted at A3 (dndz12 factors)
    null2 = load_catalogs_single(ART / "catalog_null2_dndz12_full", 2)
    run_bin("null|class2@A2_vs_class2@A3", d12[1], null2, feature_set="all")
    # fitter-depth sensitivity (fiducial spectra, tighter tol)
    for tol in ("tol005", "tol002"):
        while not all((ART / f"catalog_{tol}_full" / f"class{c}.npz").exists() for c in (2, 3)):
            time.sleep(120)                                 # tol refits run last in the refit queue
        t2c = load_catalogs_single(ART / f"catalog_{tol}_full", 2)
        t3c = load_catalogs_single(ART / f"catalog_{tol}_full", 3)
        run_bin(f"{tol}|23|all", t2c, t3c, feature_set="all")
        res.setdefault("components_per_sightline", {})[tol] = {
            "class2": float(t2c["logN"].size / 16384), "class3": float(t3c["logN"].size / 16384)}
    res.setdefault("components_per_sightline", {})["fiducial"] = {
        f"class{c}": float(fid[c - 1]["logN"].size / 16384) for c in range(1, 5)}
    (OUT / "t1_results.json").write_text(json.dumps(res, indent=2, default=float))
    log("T1 complete")


def load_catalogs_single(d: Path, c: int) -> dict:
    z = np.load(d / f"class{c}.npz")
    return {k: z[k] for k in ("sightline", "v_kms", "b_kms", "logN", "diag", "diag_keys")}


if __name__ == "__main__":
    main()
