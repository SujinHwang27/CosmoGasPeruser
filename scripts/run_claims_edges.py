"""lit-claims-audit EDGE_MAP checks (PI-authorised, EDGE_MAP.md §7):
  e2  fitter-jitter null for the T6 paired-absorber localisation (E2)
  e1  thermal-nuisance degeneracy: class 2 (1) + fixed extra broadening vs class 3 (4), forward-modelled S/N 30 (E1)
  e3  cool-phase T0 cut / weighting robustness from the LOS fields (E3)
  e4  P_F(k) shape residual after the COS-like forward model (E4)
  m1  mechanism (panel attack 1a): tau-weighted gas temperature per component; b_th vs b at fixed N; paired dT
  x2  per-class transfer function noiseless -> S/N 30 at fixed true (N, b) (panel attack 2)
  x7  suite identity: LOS headers and void-density correlation across classes (panel attack 7)
  x8  combined realistic instrument configuration (panel attack 8)

PYTHONPATH=. uv run python scripts/run_claims_edges.py --checks e2,e3,e4,e1
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from src.core.claims import (B_TH_COEF, BlockBootstrap, binary_cv, binned_dmedb, component_temperatures,
                             forward_pk_stats, load_catalogs, match_lines, profile_chi2, read_los_fields,
                             skewer_blocks, t0_cut_sweep, t6_paired, tau_mapping_lag)
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


N_TB = ((12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 14.5))
B_TB = ((0.0, 20.0), (20.0, 35.0), (35.0, 300.0))


def _sub(cats: list, fields: list, key: str) -> list:
    return [dict(c, b_kms=f[key]) for c, f in zip(cats, fields)]


def m1(bb: BlockBootstrap, n: int) -> dict:
    res = {"tau_mapping": {f"class{c}": tau_mapping_lag(c, n_skewers=32) for c in (1, 4)}}
    lag = res["tau_mapping"]["class1"]["best_lag"]
    for label, d in (("noiseless_dndz12", ART / "catalog_dndz12_full"), ("snr30_dndz12_s0", ART / "noisy_dndz12_snr30_s0")):
        cats = load_catalogs(d)
        der = []
        for c, cat in enumerate(cats, 1):
            T = component_temperatures(c, cat, lag_px=lag)
            n_nan = int(np.isnan(T).sum())
            T = np.where(np.isnan(T), np.nanmedian(T), T)
            bth = B_TH_COEF * np.sqrt(T)
            der.append({"T": T, "log10T": np.log10(T), "bth": bth, "bnt2": cat["b_kms"] ** 2 - bth ** 2, "n_nan": n_nan})
            print(f"[edges m1] {label} class {c}: T done ({n_nan} NaN)", flush=True)
        ent: dict = {"per_class": {}, "deltas": {}}
        for c, (cat, dr) in enumerate(zip(cats, der), 1):
            pc = {"n_T_nan": dr["n_nan"]}
            for lo, hi in N_TB:
                m = (cat["logN"] >= lo) & (cat["logN"] < hi)
                pc[f"{lo}-{hi}"] = {"median_b": float(np.median(cat["b_kms"][m])), "median_b_th": float(np.median(dr["bth"][m])),
                                    "median_T": float(np.median(dr["T"][m])), "frac_T_gt_10^4.5": float(np.mean(dr["T"][m] > 10 ** 4.5)),
                                    "frac_T_gt_10^5": float(np.mean(dr["T"][m] > 1e5)),
                                    "median_b_nt2": float(np.median(dr["bnt2"][m])), "frac_b_lt_b_th": float(np.mean(dr["bnt2"][m] < 0))}
            ent["per_class"][f"class{c}"] = pc
        for a, b in ((2, 3), (1, 4), (2, 4), (3, 4)):
            ent["deltas"][f"{b}-{a}"] = {k: binned_dmedb(cats if k == "b" else _sub(cats, der, k), bb, a, b, bins=N_TB)
                                         for k in ("b", "bth", "log10T", "bnt2")}
        if label.startswith("noiseless"):
            paired = {}
            for a, b in ((2, 3), (1, 4)):
                M = match_lines(cats[a - 1], cats[b - 1], n, 12.0, 10.0)
                MT = match_lines(_sub(cats, der, "T")[a - 1], _sub(cats, der, "T")[b - 1], n, 12.0, 10.0)
                db = M["b_b"] - M["b_a"]
                dbth = B_TH_COEF * (np.sqrt(MT["b_b"]) - np.sqrt(MT["b_a"]))
                dT = MT["b_b"] - MT["b_a"]
                up, dn = db > 5, db < -5
                paired[f"{a}->{b}"] = {"n": int(db.size), "mean_db": float(db.mean()), "mean_db_th": float(dbth.mean()),
                                      "corr_db_dbth": float(np.corrcoef(db, dbth)[0, 1]),
                                      "frac_var_db_explained_by_dbth": float(1 - np.var(db - dbth) / np.var(db)),
                                      "P_dT_gt0_given_db_gt5": float(np.mean(dT[up] > 0)), "P_dT_lt0_given_db_lt_m5": float(np.mean(dT[dn] < 0)),
                                      "P_dT_gt0_all": float(np.mean(dT > 0)),
                                      "median_dlog10T_given_db_gt5": float(np.median(np.log10(MT["b_b"][up] / MT["b_a"][up]))),
                                      "median_dlog10T_given_absdb_lt1": float(np.median(np.log10(MT["b_b"] / MT["b_a"])[np.abs(db) < 1]))}
            ent["paired"] = paired
        res[label] = ent
    return res


def x2(n: int) -> dict:
    true, obs = load_catalogs(ART / "catalog_dndz12_full"), load_catalogs(ART / "noisy_dndz12_snr30_s0")
    res: dict = {}
    for c in range(1, 5):
        M = match_lines(true[c - 1], obs[c - 1], n, 12.0, 10.0)
        t = true[c - 1]
        ent = {}
        for lo, hi in N_TB:
            for blo, bhi in B_TB:
                mt = (t["logN"] >= lo) & (t["logN"] < hi) & (t["b_kms"] >= blo) & (t["b_kms"] < bhi)
                mm = (M["logN_a"] >= lo) & (M["logN_a"] < hi) & (M["b_a"] >= blo) & (M["b_a"] < bhi)
                ent[f"N{lo}-{hi}|b{blo:.0f}-{bhi:.0f}"] = {
                    "n_true": int(mt.sum()), "detected_frac": float(mm.sum() / max(mt.sum(), 1)),
                    "median_db_obs_minus_true": float(np.median(M["b_b"][mm] - M["b_a"][mm])) if mm.sum() >= 30 else None}
        res[f"class{c}"] = ent
    for a, b in ((2, 3), (1, 4)):
        res[f"transfer_diff_{b}-{a}"] = {k: {"d_detected_frac": res[f"class{b}"][k]["detected_frac"] - res[f"class{a}"][k]["detected_frac"],
                                             "d_median_db": (res[f"class{b}"][k]["median_db_obs_minus_true"] - res[f"class{a}"][k]["median_db_obs_minus_true"])
                                             if res[f"class{a}"][k]["median_db_obs_minus_true"] is not None
                                             and res[f"class{b}"][k]["median_db_obs_minus_true"] is not None else None}
                                         for k in res["class1"]}
    return res


def x7() -> dict:
    rows = np.sort(np.random.default_rng(0).choice(16384, 256, replace=False))
    f = {c: read_los_fields(c) for c in range(1, 5)}
    ld = {c: np.log10(np.asarray(f[c]["density"][rows])).ravel() for c in range(1, 5)}
    void = ld[1] < np.log10(0.3)
    res = {"headers": {f"class{c}": f[c]["header"] for c in range(1, 5)},
           "headers_identical": all(f[c]["header"] == f[1]["header"] for c in range(2, 5))}
    for c in range(2, 5):
        res[f"class{c}_vs_1"] = {"r_log_density_all": float(np.corrcoef(ld[c], ld[1])[0, 1]),
                                 "r_log_density_void": float(np.corrcoef(ld[c][void], ld[1][void])[0, 1]),
                                 "median_dlog_density_void": float(np.median(ld[c][void] - ld[1][void]))}
    return res


def x8(blocks: np.ndarray, bb: BlockBootstrap) -> dict:
    d = ART / "noisy_dndz12_realistic_s0"
    cats = load_catalogs(d)
    seeds = list(range(5))
    res = {"meta": json.loads((d / "meta.json").read_text())}
    for a, b in ((2, 3), (1, 4)):
        res[f"{b}-{a}"] = {"dmedb": binned_dmedb(cats, bb, a, b),
                           "ba_all": binary_cv(cats[a - 1], cats[b - 1], blocks, seeds, feature_set="all")["ba_rf"],
                           "ba_b_only": binary_cv(cats[a - 1], cats[b - 1], blocks, seeds, feature_set="b_only")["ba_rf"]}
        print(f"[edges x8] {b}-{a}: BA {res[f'{b}-{a}']['ba_all']}", flush=True)
    res["median_b"] = [float(np.median(c["b_kms"][(c["logN"] >= 12.5) & (c["logN"] < 14.5)])) for c in cats]
    return res


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
        elif chk == "m1":
            res["m1_mechanism"] = m1(bb, blocks.size)
        elif chk == "x2":
            res["x2_transfer"] = x2(blocks.size)
        elif chk == "x7":
            res["x7_suite"] = x7()
        elif chk == "x8":
            res["x8_realistic"] = x8(blocks, bb)
        path.write_text(json.dumps(res, indent=2, default=float))
        print(f"[edges {time.time() - t0:6.0f}s] {chk} done", flush=True)
    print("[edges] complete", flush=True)


if __name__ == "__main__":
    main()
