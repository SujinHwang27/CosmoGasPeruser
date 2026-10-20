"""lit-claims-audit T2 analysis: forward-modelled (COS-like) catalogs -> per-sightline 2-vs-3 / 1-vs-4
classification, binned Delta median b, noiseless->observed b shift, n_3sigma forecast (CLAIM_TEST_PLAN T2 +
amendment 5).

PYTHONPATH=. uv run python scripts/run_claims_t2_analysis.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.core.claims import (N_BINS_T1, BlockBootstrap, binary_cv, binned_dmedb, load_catalogs, skewer_blocks,
                             weighted_quantile)
from src.core.provenance import provenance_header

OUT = Path("results/lit_claims_audit")
ART = OUT / "artifacts"
SEEDS = list(range(10))
PRIMARY = "dndz12_snr30_s0"
RUNS = ([f"dndz12_snr30_s{s}" for s in range(5)] + [f"dndz12_snr{x}_s0" for x in ("inf", "10", "20", "50")]
        + ["meanF_snr30_s0"] + [f"dndz12_snr30_s0_{v}" for v in ("gauss", "sqrtF", "aicc10", "bturb20", "cont0")])
N_SKEWERS = 16384
DZ_PER_SIGHTLINE = 5399.56 / 299792.458 * 1.3               # box velocity span -> redshift path at z = 0.3


def single(d: Path, c: int) -> Dict[str, np.ndarray]:
    z = np.load(d / f"class{c}.npz")
    return {k: z[k] for k in ("sightline", "v_kms", "b_kms", "logN", "diag", "diag_keys")}


def qc(cats: List[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for c, cat in enumerate(cats, 1):
        keys = [str(k) for k in cat["diag_keys"]]
        d = cat["diag"]
        col = lambda k: d[:, keys.index(k)] if k in keys else np.full(d.shape[0], np.nan)
        err = col("n_fit_errors")
        m = (cat["logN"] >= 12.5) & (cat["logN"] < 14.5)
        out[f"class{c}"] = {"comps_per_sightline": float(cat["logN"].size / N_SKEWERS),
                            "comps_per_sightline_12p5_14p5": float(m.sum() / N_SKEWERS),
                            "frac_sightlines_crashed": float(np.mean(err < 0)),
                            "frac_sightlines_with_fit_errors": float(np.mean(err > 0)),
                            "median_chi2_nu": float(np.nanmedian(col("chi2_nu"))) if "chi2_nu" in keys else None,
                            "median_b_12p5_14p5": float(np.median(cat["b_kms"][m])),
                            "median_b_by_bin": {f"{lo}-{hi}": float(np.median(cat["b_kms"][(cat["logN"] >= lo)
                                                                                            & (cat["logN"] < hi)]))
                                                for lo, hi in N_BINS_T1}}
    return out


def _combine(deltas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    ok = (ses > 0) & np.isfinite(deltas)
    if not ok.any():
        return {}
    w = 1.0 / ses[ok] ** 2
    d, se = float(np.sum(w * deltas[ok]) / w.sum()), float(w.sum() ** -0.5)
    n3 = float(N_SKEWERS * (3 * se / max(abs(d), 1e-9)) ** 2)
    return {"delta": d, "se": se, "z": d / se, "n3sig_sightlines": n3, "dz_path": n3 * DZ_PER_SIGHTLINE}


def forecast(cats: List[Dict[str, np.ndarray]], bb: BlockBootstrap, a: int, b: int,
             dmedb: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Sightlines for a 3-sigma ensemble detection of Delta median b in log N bins (192-block bootstrap SE,
    cosmic variance included; SE ~ n^-1/2 so n_3sig = N_SKEWERS (3 SE / |Delta|)^2, inverse-variance over bins).
    observational: one observed sample vs a model prediction -> single-sample SE of median b (class a);
    paired_sim_vs_sim: shared-IC paired SE of the difference (lower bound, not observationally available)."""
    deltas, se_pair, se_obs, per = [], [], [], {}
    for lo, hi in N_BINS_T1:
        k = f"{lo}-{hi}"
        v = dmedb[k]
        c = cats[a - 1]
        m = (c["logN"] >= lo) & (c["logN"] < hi)
        boot = weighted_quantile(c["b_kms"][m], bb.line_weights(c["sightline"][m]), 0.5)
        deltas.append(v["value"])
        se_pair.append((v["ci95_hi"] - v["ci95_lo"]) / (2 * 1.96))
        se_obs.append(float(np.std(boot, ddof=1)))
        per[k] = {"delta": v["value"], "se_paired": se_pair[-1], "se_single_sample": se_obs[-1]}
    deltas, se_pair, se_obs = np.array(deltas), np.array(se_pair), np.array(se_obs)
    return {"per_bin": per, "observational": _combine(deltas, se_obs), "paired_sim_vs_sim": _combine(deltas, se_pair)}


def z_vs(ba: Dict[str, float], ref: float) -> float:
    se = (ba["ci95_hi"] - ba["ci95_lo"]) / (2 * 1.96)
    return float((ba["value"] - ref) / se) if se > 0 else float("nan")


def main() -> None:
    t0 = time.time()
    log = lambda m: print(f"[t2a {time.time() - t0:6.0f}s] {m}", flush=True)
    blocks = skewer_blocks()
    bb = BlockBootstrap(blocks, n_boot=500, seed=0)
    res: Dict[str, object] = {"provenance": provenance_header(), "runs": {}, "nulls": {}, "primary_extras": {}}
    path = OUT / "t2_results.json"

    def save() -> None:
        path.write_text(json.dumps(res, indent=2, default=float))

    def strip(r: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in r.items() if k != "per_split"} | {"per_split": r["per_split"]}

    noiseless = load_catalogs(ART / "catalog_dndz12_full")
    res["noiseless_dndz12"] = qc(noiseless)
    for name in RUNS:
        d = ART / f"noisy_{name}"
        if not all((d / f"class{c}.npz").exists() for c in range(1, 5)):
            res["runs"][name] = {"missing": True}
            log(f"{name}: MISSING")
            continue
        cats = load_catalogs(d)
        ent: Dict[str, object] = {"meta": json.loads((d / "meta.json").read_text()), "qc": qc(cats)}
        ent["dmedb"] = {f"{b}-{a}": binned_dmedb(cats, bb, a, b) for a, b in ((2, 3), (1, 4))}
        ent["forecast"] = {"3-2": forecast(cats, bb, 2, 3, ent["dmedb"]["3-2"]),
                           "4-1": forecast(cats, bb, 1, 4, ent["dmedb"]["4-1"])}
        ent["b_shift_vs_noiseless"] = {f"class{c}": ent["qc"][f"class{c}"]["median_b_12p5_14p5"]
                                       - res["noiseless_dndz12"][f"class{c}"]["median_b_12p5_14p5"] for c in range(1, 5)}
        ent["binary_23"] = strip(binary_cv(cats[1], cats[2], blocks, SEEDS, feature_set="all"))
        ent["binary_14"] = strip(binary_cv(cats[0], cats[3], blocks, SEEDS, feature_set="all"))
        res["runs"][name] = ent
        log(f"{name}: 2v3 RF {ent['binary_23']['ba_rf']['value']:.4f} [{ent['binary_23']['ba_rf']['ci95_lo']:.4f},"
            f"{ent['binary_23']['ba_rf']['ci95_hi']:.4f}] | 1v4 RF {ent['binary_14']['ba_rf']['value']:.4f} | "
            f"medb {[round(ent['qc'][f'class{c}']['median_b_12p5_14p5'], 1) for c in range(1, 5)]} | "
            f"n3sig obs (3-2) {ent['forecast']['3-2']['observational'].get('n3sig_sightlines', float('nan')):.0f} "
            f"(4-1) {ent['forecast']['4-1']['observational'].get('n3sig_sightlines', float('nan')):.0f}")
        save()

    p = ART / f"noisy_{PRIMARY}"
    if (p / "class2.npz").exists():
        cats = load_catalogs(p)
        for fs in ("b_only", "n_only"):
            res["primary_extras"][f"23|{fs}"] = strip(binary_cv(cats[1], cats[2], blocks, SEEDS, feature_set=fs))
            log(f"primary 23|{fs}: RF {res['primary_extras'][f'23|{fs}']['ba_rf']['value']:.4f}")
        for det in (12.5, 13.0):
            r = strip(binary_cv(cats[1], cats[2], blocks, SEEDS, feature_set="all", logN_detect=det))
            res["primary_extras"][f"23|detect{det}"] = r
            log(f"primary 23|detect{det}: RF {r['ba_rf']['value']:.4f}")
        save()
        s1 = ART / "noisy_dndz12_snr30_s1" / "class2.npz"
        if s1.exists():
            r = strip(binary_cv(cats[1], single(s1.parent, 2), blocks, SEEDS, feature_set="all"))
            res["nulls"]["label|class2_s0_vs_class2_s1"] = r
            log(f"label null: RF {r['ba_rf']['value']:.4f} [{r['ba_rf']['ci95_lo']:.4f},{r['ba_rf']['ci95_hi']:.4f}]")
        nr = ART / "noisy_null2_dndz12_snr30_s7" / "class2.npz"
        if nr.exists():
            r = strip(binary_cv(cats[1], single(nr.parent, 2), blocks, SEEDS, feature_set="all"))
            res["nulls"]["rescaling|class2@A2_s0_vs_class2@A3_s7"] = r
            log(f"rescaling null: RF {r['ba_rf']['value']:.4f} [{r['ba_rf']['ci95_lo']:.4f},{r['ba_rf']['ci95_hi']:.4f}]")
        save()

    # pre-registered verdicts (CLAIM_TEST_PLAN T2 + amendments 4-5)
    v: Dict[str, object] = {}
    seeds_ba = [res["runs"][f"dndz12_snr30_s{s}"]["binary_23"]["ba_rf"] for s in range(5)
                if "binary_23" in res["runs"].get(f"dndz12_snr30_s{s}", {})]
    if seeds_ba:
        v["primary_rf_ba_23_per_seed"] = [b["value"] for b in seeds_ba]
        v["primary_rf_ba_23_min_ci_lo"] = min(b["ci95_lo"] for b in seeds_ba)
        v["primary_rf_ba_23_mean"] = float(np.mean([b["value"] for b in seeds_ba]))
        null_ref = max([0.5] + [r["ba_rf"]["value"] for r in res["nulls"].values()])
        v["null_reference_ba"] = null_ref
        v["z_vs_0p5_seed0"] = z_vs(seeds_ba[0], 0.5)
        lo = v["primary_rf_ba_23_min_ci_lo"]
        v["per_sightline_23"] = ("EXTENDS (distinguishable per sightline)" if lo > 0.52 and lo > null_ref + 0.02
                                 else "VANISHES" if lo <= null_ref + 0.01 else "INCONCLUSIVE per sightline")
    if "binary_14" in res["runs"].get(PRIMARY, {}):
        b14 = res["runs"][PRIMARY]["binary_14"]["ba_rf"]
        v["per_sightline_14"] = "EXTENDS (distinguishable per sightline)" if b14["ci95_lo"] > 0.52 else "INCONCLUSIVE"
        q = res["runs"][PRIMARY]["qc"]
        v["narrow_line_tension_all_median_b_lt_30"] = bool(all(q[f"class{c}"]["median_b_12p5_14p5"] < 30 for c in range(1, 5)))
        v["b_shift_ge_4_consistent_with_C08"] = {k: bool(x >= 4) for k, x in res["runs"][PRIMARY]["b_shift_vs_noiseless"].items()}
    res["verdicts"] = v
    save()
    log("T2 analysis complete")


if __name__ == "__main__":
    main()
