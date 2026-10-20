"""lit-claims-audit EDGE_MAP checks (PI-authorised, EDGE_MAP.md §7):
  e2  fitter-jitter null for the T6 paired-absorber localisation (E2)
  e1  thermal-nuisance degeneracy: class 2 (1) + fixed extra broadening vs class 3 (4), forward-modelled S/N 30 (E1)
  e3  cool-phase T0 cut / weighting robustness from the LOS fields (E3)
  e4  P_F(k) shape residual after the COS-like forward model (E4)

PYTHONPATH=. uv run python scripts/run_claims_edges.py --checks e2,e3,e4,e1
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from src.core.claims import (BlockBootstrap, binary_cv, binned_dmedb, forward_pk_stats, load_catalogs, profile_chi2,
                             skewer_blocks, t0_cut_sweep, t6_paired)
from src.core.provenance import provenance_header

OUT = Path("results/lit_claims_audit")
ART = OUT / "artifacts"
FID = Path("results/eigen_voigt_embedding/artifacts/catalog_fiducial_full")
COS_LSF = Path("data/raw/Sherwood/COS_LSF1_cw1327_75_z0.100.dat")
B_EXTRA = (3, 6, 9, 12, 15, 18, 21)
KEEP = ("n_matched", "match_rate", "mean_db", "median_db", "frac_absdb_gt5", "frac_db_gt5", "frac_db_lt_m5",
        "share_of_positive_broadening_top20pct")


def single(d: Path, c: int) -> dict:
    z = np.load(d / f"class{c}.npz")
    return {k: z[k] for k in ("sightline", "v_kms", "b_kms", "logN", "diag", "diag_keys")}


def e2(bb: BlockBootstrap, n: int) -> dict:
    fid, d12 = load_catalogs(FID), load_catalogs(ART / "catalog_dndz12_full")
    comps = {
        "real_fiducial": t6_paired(fid, bb, n, pairs=((2, 3), (1, 4)), logN_min=12.0),
        "real_dndz12": t6_paired(d12, bb, n, pairs=((2, 3), (1, 4)), logN_min=12.0),
        "null_rescale_class2@A2_vs_class2@A3": t6_paired([d12[1], single(ART / "catalog_null2_dndz12_full", 2)], bb, n,
                                                         pairs=((1, 2),), logN_min=12.0),
        "null_depth_class2_fid_vs_tol005": t6_paired([fid[1], single(ART / "catalog_tol005_full", 2)], bb, n,
                                                     pairs=((1, 2),), logN_min=12.0),
    }
    res = {}
    for name, r in comps.items():
        for pk, v in r.items():
            if pk.startswith("pair_"):
                ent = {k: v[k] for k in KEEP}
                ent["tail_asymmetry_gt5_minus_ltm5"] = v["frac_db_gt5"] - v["frac_db_lt_m5"]
                res[f"{name}|{pk}"] = ent
    return res


def e1(blocks: np.ndarray, bb: BlockBootstrap) -> dict:
    base = ART / "noisy_dndz12_snr30_s0"
    tgt = {2: single(base, 3), 1: single(base, 4)}
    seeds = list(range(5))
    res = {"b_extra_kms": list(B_EXTRA), "seeds_cv": seeds, "pairs": {}}
    for src, lab in ((2, "class2+bx_vs_class3"), (1, "class1+bx_vs_class4")):
        rows = []
        for bx in (0,) + B_EXTRA:
            d = base if bx == 0 else ART / f"noisy_e1_bx{bx}"
            if not (d / f"class{src}.npz").exists():
                continue
            a = single(d, src)
            prof = binned_dmedb([a, tgt[src]], bb, 1, 2)
            row = {"b_extra": bx, "dmedb": prof, "profile_chi2": profile_chi2(prof),
                   "ba_all": binary_cv(a, tgt[src], blocks, seeds, feature_set="all")["ba_rf"],
                   "ba_b_only": binary_cv(a, tgt[src], blocks, seeds, feature_set="b_only")["ba_rf"],
                   "comps_per_sightline": float(a["logN"].size / blocks.size)}
            rows.append(row)
            print(f"[edges e1] {lab} bx={bx}: chi2 {row['profile_chi2']['chi2']:.1f}/{row['profile_chi2']['dof']} "
                  f"p={row['profile_chi2']['p']:.2g} | BA all {row['ba_all']['value']:.4f} b-only {row['ba_b_only']['value']:.4f}",
                  flush=True)
        res["pairs"][lab] = rows
    return res


def e4() -> dict:
    from src.core.forward import tabulated_lsf
    from src.core.eigen_voigt import load_grid
    _, vel = load_grid()
    lsf = tabulated_lsf(COS_LSF, float(vel[1] - vel[0]))
    A = json.loads((OUT / "matching_factors.json").read_text())["cddf"]["A"]
    return {f"snr{s}": forward_pk_stats(A, lsf, snr=s) for s in (np.inf, 50.0, 30.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="e2,e3,e4,e1")
    args = ap.parse_args()
    t0 = time.time()
    blocks = skewer_blocks()
    bb = BlockBootstrap(blocks, n_boot=500, seed=0)
    path = OUT / "edge_checks.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["provenance"] = provenance_header()
    for chk in args.checks.split(","):
        if chk == "e2":
            res["e2_jitter_null"] = e2(bb, blocks.size)
        elif chk == "e3":
            res["e3_t0_cut_sweep"] = t0_cut_sweep()
        elif chk == "e4":
            res["e4_forward_pk"] = e4()
        elif chk == "e1":
            res["e1_nuisance"] = e1(blocks, bb)
        path.write_text(json.dumps(res, indent=2, default=float))
        print(f"[edges {time.time() - t0:6.0f}s] {chk} done", flush=True)
    print("[edges] complete", flush=True)


if __name__ == "__main__":
    main()
