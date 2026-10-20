"""
lit-claims-audit: literature-claim tests on the Sherwood z=0.3 Voigt catalogs.

Scientific purpose
------------------
Tests published claims about how feedback imprints on the low-z Lyman-alpha forest
(experiments/lit-claims-audit/CLAIM_TEST_PLAN.md) on this suite's noiseless Voigt
catalogs, to locate where the literature leaves room for a per-sightline / line-by-line
feedback signature:

  T5  winds vs AGN on b and the CDDF (Viel+2017 vs Nasir+2017),
  T4a thermal lower envelope b_env(N) vs the bulk of the b distribution (Bolton+2022,
      Hu+2024): does feedback move the thermal floor or only the bulk?
  T6  paired-absorber response: the same skewer exists in every class (shared initial
      conditions), so absorbers can be matched line by line across classes.

All uncertainties are block bootstraps over the 192 spatial blocks (whole skewers by
projection axis x 7.5 Mpc/h transverse cell). The classes share skewers, so a single
block resample is applied to every class — differences are paired.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.eigen_eval import spatial_blocks
from src.core.eigen_voigt import BOX_KPC, FLUX_BASE, N_BLOCK_CELLS, N_CLASSES

SPAN_KMS = 2048 * 2.6365028403447024        # periodic sightline length (km/s)


def skewer_blocks() -> np.ndarray:
    """Spatial block id per skewer index (16384,), shared by every class."""
    axis = np.load(Path(FLUX_BASE) / "1" / "axis.npy")
    return spatial_blocks(axis, BOX_KPC, N_BLOCK_CELLS)


def load_catalogs(cat_dir: Path) -> List[Dict[str, np.ndarray]]:
    """Per-class flattened catalogs (sightline, v_kms, b_kms, logN) plus diag."""
    out = []
    for c in range(1, N_CLASSES + 1):
        z = np.load(Path(cat_dir) / f"class{c}.npz")
        out.append({k: z[k] for k in ("sightline", "v_kms", "b_kms", "logN", "diag", "diag_keys")})
    return out


class BlockBootstrap:
    """Shared block-resampling weights; weighted statistics of line-level samples."""

    def __init__(self, blocks: np.ndarray, n_boot: int = 500, seed: int = 0) -> None:
        self.blocks = blocks                                 # (n_skewers,) block id
        ub, self.inv = np.unique(blocks, return_inverse=True)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, ub.size, ub.size), minlength=ub.size)
                           for _ in range(n_boot)]).astype(np.float32)   # (n_boot, n_blocks)

    def line_weights(self, sightline: np.ndarray) -> np.ndarray:
        """(n_boot, n_lines) weight of every line under every resample."""
        return self.W[:, self.inv[sightline]]


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Quantile q of x under each row of weights w (n_rep, n) -> (n_rep,)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    cw = np.cumsum(w[:, order], axis=1)
    tot = cw[:, -1:]
    idx = np.array([np.searchsorted(cw[i], q * tot[i, 0]) for i in range(cw.shape[0])])
    return xs[np.clip(idx, 0, xs.size - 1)]


def _summ(point: float, boot: np.ndarray) -> Dict[str, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return {"value": float(point), "ci95_lo": float(lo), "ci95_hi": float(hi), "p_two_sided_vs0": p_two}


# ---------------------------------------------------------------------------
# T5 — winds vs AGN on b and the CDDF
# ---------------------------------------------------------------------------
def t5_winds_vs_agn(cats: List[Dict[str, np.ndarray]], bb: BlockBootstrap,
                    logN_cut: float = 13.0) -> Dict[str, object]:
    """Median-b differences between classes, f(b>50) ratio 3/2, CDDF count ratio 2/1 at
    log N >= 14, all with paired block-bootstrap CIs."""
    med_pt, med_bt, f50_pt, f50_bt, n14_pt, n14_bt = {}, {}, {}, {}, {}, {}
    for c, cat in enumerate(cats, start=1):
        m = cat["logN"] >= logN_cut
        b, sl = cat["b_kms"][m], cat["sightline"][m]
        w = bb.line_weights(sl)
        med_pt[c] = float(np.median(b))
        med_bt[c] = weighted_quantile(b, w, 0.5)
        f50_pt[c] = float(np.mean(b > 50.0))
        f50_bt[c] = (w * (b > 50.0)).sum(axis=1) / w.sum(axis=1)
        m14 = cat["logN"] >= 14.0
        n14_pt[c] = float(m14.sum())
        n14_bt[c] = bb.line_weights(cat["sightline"][m14]).sum(axis=1)
    out: Dict[str, object] = {"logN_cut": logN_cut, "median_b": med_pt, "f_b_gt50": f50_pt}
    for a, b_ in ((2, 1), (3, 2), (4, 3), (3, 1), (4, 1)):
        out[f"dmedb_{a}{b_}"] = _summ(med_pt[a] - med_pt[b_], med_bt[a] - med_bt[b_])
    r = f50_pt[3] / f50_pt[2]
    out["f50_ratio_32"] = _summ(r, f50_bt[3] / f50_bt[2])
    out["n14_ratio_21"] = _summ(n14_pt[2] / n14_pt[1], n14_bt[2] / n14_bt[1])
    out["n14_ratio_31"] = _summ(n14_pt[3] / n14_pt[1], n14_bt[3] / n14_bt[1])
    out["n14_ratio_41"] = _summ(n14_pt[4] / n14_pt[1], n14_bt[4] / n14_bt[1])
    return out


# ---------------------------------------------------------------------------
# T4a — thermal lower envelope vs bulk
# ---------------------------------------------------------------------------
ENV_EDGES = np.arange(12.75, 14.5 + 1e-9, 0.25)


def _envelope(b: np.ndarray, N: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Per resample: power-law fit of the q-quantile of b in ENV_EDGES bins, evaluated at
    log N = 13.5. Returns (n_rep,)."""
    centers, vals = [], []
    for lo, hi in zip(ENV_EDGES[:-1], ENV_EDGES[1:]):
        m = (N >= lo) & (N < hi)
        if m.sum() < 50:
            continue
        centers.append(0.5 * (lo + hi) - 13.5)
        vals.append(np.log10(weighted_quantile(b[m], w[:, m], q)))
    X = np.column_stack([np.ones(len(centers)), centers])
    Y = np.stack(vals, axis=0)                               # (n_bins, n_rep)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return 10 ** coef[0]


def t4a_envelope(cats: List[Dict[str, np.ndarray]], bb: BlockBootstrap, q_env: float = 0.10
                 ) -> Dict[str, object]:
    """b_env(13.5) (10th-percentile envelope) and bulk median b(13.5) per class; class
    differences with paired CIs. Point estimates use unit weights."""
    env_pt, env_bt, bulk_pt, bulk_bt = {}, {}, {}, {}
    for c, cat in enumerate(cats, start=1):
        m = (cat["logN"] >= ENV_EDGES[0]) & (cat["logN"] < ENV_EDGES[-1])
        b, N, sl = cat["b_kms"][m], cat["logN"][m], cat["sightline"][m]
        w = bb.line_weights(sl)
        one = np.ones((1, b.size))
        env_pt[c] = float(_envelope(b, N, one, q_env)[0])
        env_bt[c] = _envelope(b, N, w, q_env)
        bulk_pt[c] = float(_envelope(b, N, one, 0.5)[0])
        bulk_bt[c] = _envelope(b, N, w, 0.5)
    out: Dict[str, object] = {"q_env": q_env, "b_env_13p5": env_pt, "b_med_13p5": bulk_pt}
    for a, b_ in ((4, 1), (3, 2), (2, 1), (3, 1)):
        out[f"denv_{a}{b_}"] = _summ(env_pt[a] - env_pt[b_], env_bt[a] - env_bt[b_])
        out[f"dbulk_{a}{b_}"] = _summ(bulk_pt[a] - bulk_pt[b_], bulk_bt[a] - bulk_bt[b_])
    return out


# ---------------------------------------------------------------------------
# T6 — paired absorbers across classes (shared initial conditions)
# ---------------------------------------------------------------------------
def match_lines(cat_a: Dict[str, np.ndarray], cat_b: Dict[str, np.ndarray], n_skewers: int,
                logN_min: float = 12.5) -> Dict[str, np.ndarray]:
    """Mutual-nearest-in-velocity matching of class-a and class-b components on the same
    skewer, |dv| < max(0.5 b_a, 5 km/s), periodic in v."""
    def split(cat):
        m = cat["logN"] >= logN_min
        sl, v, b, N = cat["sightline"][m], cat["v_kms"][m], cat["b_kms"][m], cat["logN"][m]
        o = np.argsort(sl, kind="stable")
        sl, v, b, N = sl[o], v[o], b[o], N[o]
        bounds = np.searchsorted(sl, np.arange(n_skewers + 1))
        return v, b, N, bounds

    va, ba, Na, ka = split(cat_a)
    vb, bbb, Nb, kb = split(cat_b)
    rows = []
    n_a_total = 0
    for i in range(n_skewers):
        a0, a1, b0, b1 = ka[i], ka[i + 1], kb[i], kb[i + 1]
        n_a_total += a1 - a0
        if a1 == a0 or b1 == b0:
            continue
        d = vb[b0:b1][None, :] - va[a0:a1][:, None]
        d = (d + SPAN_KMS / 2) % SPAN_KMS - SPAN_KMS / 2          # periodic separation
        ad = np.abs(d)
        ia = np.argmin(ad, axis=1)                           # nearest b for each a
        ib = np.argmin(ad, axis=0)                           # nearest a for each b
        for j in range(a1 - a0):
            k = ia[j]
            if ib[k] != j:
                continue
            tol = max(0.5 * ba[a0 + j], 5.0)
            if ad[j, k] < tol:
                rows.append((i, va[a0 + j], ba[a0 + j], Na[a0 + j], bbb[b0 + k], Nb[b0 + k], d[j, k]))
    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 7))
    return {"sightline": arr[:, 0].astype(np.int64), "v": arr[:, 1], "b_a": arr[:, 2], "logN_a": arr[:, 3],
            "b_b": arr[:, 4], "logN_b": arr[:, 5], "dv": arr[:, 6], "n_a_total": n_a_total}


def t6_paired(cats: List[Dict[str, np.ndarray]], bb: BlockBootstrap, n_skewers: int,
              pairs: Sequence[Tuple[int, int]] = ((2, 3), (1, 4), (1, 2)),
              logN_min: float = 12.5) -> Dict[str, object]:
    """Paired line-by-line response: median Δb (b_class_b − b_class_a), Δb by log N bin,
    fraction |Δb| > 5 km/s, localisation of the broadening, pair-order accuracy, match rate."""
    out: Dict[str, object] = {"logN_min": logN_min}
    for a, b_ in pairs:
        M = match_lines(cats[a - 1], cats[b_ - 1], n_skewers, logN_min)
        db = M["b_b"] - M["b_a"]
        w = bb.line_weights(M["sightline"])
        res: Dict[str, object] = {"n_matched": int(db.size), "match_rate": float(db.size / max(M["n_a_total"], 1))}
        res["median_db"] = _summ(float(np.median(db)), weighted_quantile(db, w, 0.5))
        res["mean_db"] = _summ(float(db.mean()), (w * db).sum(axis=1) / w.sum(axis=1))
        res["median_dlogN"] = float(np.median(M["logN_b"] - M["logN_a"]))
        res["frac_absdb_gt5"] = float(np.mean(np.abs(db) > 5.0))
        res["frac_db_gt5"] = float(np.mean(db > 5.0))
        res["frac_db_lt_m5"] = float(np.mean(db < -5.0))
        pos = np.sort(np.clip(db, 0, None))[::-1]
        top = max(1, int(0.2 * pos.size))
        res["share_of_positive_broadening_top20pct"] = float(pos[:top].sum() / max(pos.sum(), 1e-12))
        res["pair_order_acc_b_larger_in_class_b"] = float(np.mean(db > 0) + 0.5 * np.mean(db == 0))
        by_bin = {}
        for lo, hi in ((12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 14.5), (14.5, 16.0)):
            m = (M["logN_a"] >= lo) & (M["logN_a"] < hi)
            if m.sum() >= 100:
                by_bin[f"{lo}-{hi}"] = _summ(float(np.median(db[m])), weighted_quantile(db[m], w[:, m], 0.5))
        res["median_db_by_logN"] = by_bin
        out[f"pair_{a}{b_}"] = res
    return out
