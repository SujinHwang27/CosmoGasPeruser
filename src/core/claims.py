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
                logN_min: float = 12.5, dv_max: float = 10.0) -> Dict[str, np.ndarray]:
    """Mutual-nearest-in-velocity matching of class-a and class-b components on the same
    skewer, |dv| < dv_max km/s (fixed, so the matching does not depend on b), periodic in v."""
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
            if ad[j, k] < dv_max:
                rows.append((i, va[a0 + j], ba[a0 + j], Na[a0 + j], bbb[b0 + k], Nb[b0 + k], d[j, k]))
    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 7))
    return {"sightline": arr[:, 0].astype(np.int64), "v": arr[:, 1], "b_a": arr[:, 2], "logN_a": arr[:, 3],
            "b_b": arr[:, 4], "logN_b": arr[:, 5], "dv": arr[:, 6], "n_a_total": n_a_total}


def t6_paired(cats: List[Dict[str, np.ndarray]], bb: BlockBootstrap, n_skewers: int,
              pairs: Sequence[Tuple[int, int]] = ((2, 3), (1, 4), (1, 2)),
              logN_min: float = 12.5, dv_max: float = 10.0) -> Dict[str, object]:
    """Paired line-by-line response: median Δb (b_class_b − b_class_a), Δb by log N bin,
    fraction |Δb| > 5 km/s, localisation of the broadening, pair-order accuracy, match rate."""
    out: Dict[str, object] = {"logN_min": logN_min, "dv_max": dv_max}
    for a, b_ in pairs:
        M = match_lines(cats[a - 1], cats[b_ - 1], n_skewers, logN_min, dv_max)
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
        for lo, hi in ((12.0, 12.5), (12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 14.5), (14.5, 16.0)):
            m = (M["logN_a"] >= lo) & (M["logN_a"] < hi)
            if m.sum() >= 100:
                by_bin[f"{lo}-{hi}"] = _summ(float(np.median(db[m])), weighted_quantile(db[m], w[:, m], 0.5))
        res["median_db_by_logN"] = by_bin
        out[f"pair_{a}{b_}"] = res
    return out


# ---------------------------------------------------------------------------
# T1 / T3 — matching factors (analytic, on the fiducial catalog: tau -> A tau shifts every
# component's log N by log10 A and leaves its b unchanged)
# ---------------------------------------------------------------------------
_S_GRID = np.arange(-0.6, 0.6 + 1e-9, 0.0005)


def dndz_match_factors(cats: List[Dict[str, np.ndarray]], lo: float, hi: float) -> Dict[str, object]:
    """A_c so that every class has the pooled-mean number of components with lo <= log N < hi."""
    counts = [int(((c["logN"] >= lo) & (c["logN"] < hi)).sum()) for c in cats]
    target = float(np.mean(counts))
    A = []
    for c in cats:
        N = np.sort(c["logN"])
        n_s = np.searchsorted(N, hi - _S_GRID) - np.searchsorted(N, lo - _S_GRID)
        A.append(float(10 ** _S_GRID[int(np.argmin(np.abs(n_s - target)))]))
    return {"window": [lo, hi], "counts_fiducial": counts, "target": target, "A": A}


def cddf_match_factors(cats: List[Dict[str, np.ndarray]], lo: float = 13.3, hi: float = 14.5,
                       dbin: float = 0.1) -> Dict[str, object]:
    """A_c minimising chi^2 between class c's CDDF and class 1's over [lo, hi] (Poisson errors)."""
    edges = np.arange(lo, hi + 1e-9, dbin)
    ref = np.histogram(cats[0]["logN"], edges)[0].astype(float)
    A = [1.0]
    for c in cats[1:]:
        chi = [np.sum((np.histogram(c["logN"] + s, edges)[0] - ref) ** 2 / np.maximum(ref, 1.0)) for s in _S_GRID[::4]]
        A.append(float(10 ** _S_GRID[::4][int(np.argmin(chi))]))
    return {"window": [lo, hi], "A": A}


def predicted_matched_db(cats: List[Dict[str, np.ndarray]], A: Sequence[float], a: int, b: int,
                         bins: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """Prediction for the matched catalog: median b(class b) - median b(class a) in log N bins
    after shifting every component's log N by log10 A_c (b unchanged)."""
    out = {}
    for lo, hi in bins:
        meds = []
        for k in (a, b):
            c = cats[k - 1]
            N = c["logN"] + np.log10(A[k - 1])
            m = (N >= lo) & (N < hi)
            meds.append(float(np.median(c["b_kms"][m])))
        out[f"{lo}-{hi}"] = meds[1] - meds[0]
    return out


# ---------------------------------------------------------------------------
# T1 — matched characterisation: ensemble Δb in log N bins + binary 2-vs-3 classification
# ---------------------------------------------------------------------------
N_BINS_T1: Tuple[Tuple[float, float], ...] = ((12.0, 12.5), (12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 14.5))
B_FEATS = ("b_q10", "b_q25", "b_q50", "b_q75", "b_q90", "b_Nweighted", "frac_b_gt40", "frac_b_lt15", "bN_slope")
N_FEATS = ("n_lines_12", "n_lines_13", "n_lines_14", "logNtot", "logN_q25", "logN_q50", "logN_q75", "logN_q90",
           "dv_nn_median", "has_lines", "has_bN_slope", "has_dv_nn")


def binned_dmedb(cats: List[Dict[str, np.ndarray]], bb: BlockBootstrap, a: int, b: int,
                 bins: Sequence[Tuple[float, float]] = N_BINS_T1) -> Dict[str, Dict[str, float]]:
    """median b(class b) - median b(class a) within log N bins, paired block bootstrap."""
    out = {}
    for lo, hi in bins:
        pts, bts = [], []
        for k in (a, b):
            c = cats[k - 1]
            m = (c["logN"] >= lo) & (c["logN"] < hi)
            pts.append(float(np.median(c["b_kms"][m])))
            bts.append(weighted_quantile(c["b_kms"][m], bb.line_weights(c["sightline"][m]), 0.5))
        out[f"{lo}-{hi}"] = _summ(pts[1] - pts[0], bts[1] - bts[0])
    return out


def _b_residual_features(cat: Dict[str, np.ndarray], n: int, train_rows: np.ndarray,
                         ref_cats: List[Dict[str, np.ndarray]], ref_train: List[np.ndarray],
                         logN_detect: float = 12.0) -> np.ndarray:
    """Per-sightline aggregates of b - median_b(log N), the median curve fit in 0.1-dex bins on
    the TRAINING sightlines of the reference classes only (fold-internal)."""
    edges = np.arange(logN_detect, 17.0 + 1e-9, 0.1)
    bs, Ns = [], []
    for rc, rt in zip(ref_cats, ref_train):
        keep = np.isin(rc["sightline"], rt) & (rc["logN"] >= logN_detect)
        bs.append(rc["b_kms"][keep])
        Ns.append(rc["logN"][keep])
    b_all, N_all = np.concatenate(bs), np.concatenate(Ns)
    idx = np.clip(np.digitize(N_all, edges) - 1, 0, edges.size - 2)
    med = np.array([np.median(b_all[idx == i]) if np.any(idx == i) else np.nan for i in range(edges.size - 1)])
    good = np.isfinite(med)
    med = np.interp(np.arange(med.size), np.flatnonzero(good), med[good])
    m = cat["logN"] >= logN_detect
    sl = cat["sightline"][m]
    r = cat["b_kms"][m] - med[np.clip(np.digitize(cat["logN"][m], edges) - 1, 0, edges.size - 2)]
    X = np.full((n, 5), np.nan)
    order = np.argsort(sl, kind="stable")
    sl, r = sl[order], r[order]
    bounds = np.searchsorted(sl, np.arange(n + 1))
    for i in range(n):
        ri = r[bounds[i]:bounds[i + 1]]
        if ri.size:
            X[i] = (np.median(ri), np.percentile(ri, 75), np.percentile(ri, 90), np.mean(ri > 10.0), np.mean(ri))
    return X


def binary_cv(cats_a: Dict[str, np.ndarray], cats_b: Dict[str, np.ndarray], blocks: np.ndarray,
              seeds: Sequence[int], feature_set: str = "all", logN_detect: float = 12.0,
              scalars: Optional[np.ndarray] = None, axis_ids: Optional[np.ndarray] = None,
              n_jobs: int = -1) -> Dict[str, object]:
    """Binary per-sightline classification of catalog A (label 0) vs catalog B (label 1).

    feature_set: all | b_only | n_only | b_resid (fold-internal b(N) residual aggregates).
    scalars (2n, k): if given, every feature column is spline-residualised on them per fold.
    axis_ids (n,): if given, folds are the 3 projection axes (axis-held-out sensitivity).
    Returns per-split BA (RF, logreg) and seed-averaged block-bootstrap BA.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from src.core.eigen_eval import _SplineResidualiser, _impute_train_median, block_bootstrap, group_splits
    from src.core.voigt import line_feature_names, line_statistics_features

    n = blocks.size
    names = line_feature_names()
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    sk = np.r_[np.arange(n), np.arange(n)]
    bl = np.r_[blocks, blocks]
    if feature_set != "b_resid":
        X = np.vstack([line_statistics_features(c["sightline"], c["v_kms"], c["b_kms"], c["logN"], n_sightlines=n,
                                                span_kms=SPAN_KMS, logN_detect=logN_detect, impute=False)
                       for c in (cats_a, cats_b)])
        if feature_set == "b_only":
            X = X[:, [names.index(f) for f in B_FEATS]]
        elif feature_set == "n_only":
            X = X[:, [names.index(f) for f in N_FEATS]]
    splits_by_seed = {}
    for s in seeds:
        if axis_ids is not None:
            ax = np.r_[axis_ids, axis_ids]
            splits_by_seed[s] = [(np.flatnonzero(ax != k), np.flatnonzero(ax == k)) for k in (1, 2, 3)]
        else:
            splits_by_seed[s] = group_splits(bl, 5, s)
    correct = {"rf": np.zeros((len(seeds), 2 * n)), "logreg": np.zeros((len(seeds), 2 * n))}
    per_split = []
    for si, s in enumerate(seeds):
        for fold, (tr, te) in enumerate(splits_by_seed[s]):
            if feature_set == "b_resid":
                tr_sk = [np.unique(sk[tr][y[tr] == 0]), np.unique(sk[tr][y[tr] == 1])]
                X = np.vstack([_b_residual_features(c, n, None, [cats_a, cats_b], tr_sk, logN_detect)
                               for c in (cats_a, cats_b)])
            Xtr, Xte = _impute_train_median(X[tr], X[te])
            if scalars is not None:
                r = _SplineResidualiser().fit(scalars[tr], Xtr)
                Xtr, Xte = r.residual(scalars[tr], Xtr), r.residual(scalars[te], Xte)
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            row = {"seed": s, "fold": fold}
            for name, clf in (("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_samples=0.5,
                                                             n_jobs=n_jobs, random_state=s)),
                              ("logreg", LogisticRegression(C=1.0, max_iter=3000))):
                clf.fit(Xtr, y[tr])
                p = clf.predict(Xte)
                correct[name][si, te] = (p == y[te])
                row[f"ba_{name}"] = float(balanced_accuracy_score(y[te], p))
            per_split.append(row)
    out: Dict[str, object] = {"feature_set": feature_set, "logN_detect": logN_detect, "per_split": per_split}
    for name in ("rf", "logreg"):
        point, boot = block_bootstrap(y + 1, bl, correct[name].mean(axis=0)[None, :], n_boot=2000, seed=0)
        out[f"ba_{name}"] = {"value": float(point[0]), "ci95_lo": float(np.percentile(boot[:, 0], 2.5)),
                             "ci95_hi": float(np.percentile(boot[:, 0], 97.5))}
    return out


# ---------------------------------------------------------------------------
# T4b — temperature-density relation from the Sherwood LOS files (C-11, C-13, C-24)
# ---------------------------------------------------------------------------
RAW_DIRS = {1: "Physics1_nofeedback", 2: "Physics2_stellarwind", 3: "Physics3_windAGN", 4: "Physics4_windstrongAGN"}
_LOS_HEADER_BYTES = 7 * 8 + 2 * 4                          # 7 doubles + nbins, numlos (int32)


def read_los_fields(class_id: int, z: float = 0.3, raw_root: str = "data/raw/Sherwood"
                    ) -> Dict[str, np.ndarray]:
    """Memory-mapped density (rho/<rho>), HI fraction, HI-weighted T [K] and v_pec [km/s],
    each (numlos, nbins), from the Sherwood los2048 binary (format: data/raw/Sherwood/src/utils.py)."""
    path = Path(raw_root) / RAW_DIRS[class_id] / f"los2048_n16384_z{z:.3f}.dat"
    with open(path, "rb") as fh:
        hdr_d = np.fromfile(fh, dtype=np.float64, count=7)
        nbins, numlos = (int(x) for x in np.fromfile(fh, dtype=np.int32, count=2))
    off = _LOS_HEADER_BYTES + numlos * 4 + 3 * numlos * 8 + 2 * nbins * 8
    size = nbins * numlos
    expect = off + 4 * size * 8
    if path.stat().st_size != expect:
        raise ValueError(f"read_los_fields: {path} size {path.stat().st_size} != expected {expect}")
    if abs(hdr_d[0] - z) > 1e-3:
        raise ValueError(f"read_los_fields: header redshift {hdr_d[0]} != {z}")
    names = ("density", "H1frac", "temp", "vpec")
    out = {n: np.memmap(path, dtype=np.float64, mode="r", offset=off + i * size * 8, shape=(numlos, nbins))
           for i, n in enumerate(names)}
    out["header"] = {"z": float(hdr_d[0]), "omegaM": float(hdr_d[1]), "omegaL": float(hdr_d[2]),
                     "omegab": float(hdr_d[3]), "h100": float(hdr_d[4]), "box_kpc_h": float(hdr_d[5]),
                     "Xh": float(hdr_d[6]), "nbins": nbins, "numlos": numlos}
    return out


def t4b_temperature_density(n_skewers_sample: int = 4096, seed: int = 0, logd_lo: float = -0.5,
                            logd_hi: float = 0.5) -> Dict[str, object]:
    """Per class: T0 and gamma from the median log T in 0.1-dex log-density bins over
    [logd_lo, logd_hi]; temperature-PDF fractions (pixels, i.e. volume-weighted along the sightline)
    above 1e5 K overall and at Delta > 10; block-bootstrap over the same skewer sample in every class.
    The HI-weighted temperature is what the LOS files carry (not a mass-weighted T)."""
    rng = np.random.default_rng(seed)
    blocks = skewer_blocks()
    rows = np.sort(rng.choice(16384, n_skewers_sample, replace=False))
    edges = np.arange(logd_lo, logd_hi + 1e-9, 0.1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ub, inv = np.unique(blocks[rows], return_inverse=True)
    W = np.stack([np.bincount(np.random.default_rng(seed + 1 + i).integers(0, ub.size, ub.size), minlength=ub.size)
                  for i in range(100)]).astype(np.float64)
    res: Dict[str, object] = {"n_skewers": int(rows.size), "fit_range_logDelta": [logd_lo, logd_hi],
                              "note": "HI-weighted T and density per pixel from the LOS files"}
    boot = {}
    for c in range(1, N_CLASSES + 1):
        f = read_los_fields(c)
        res.setdefault("header", f["header"])
        logd = np.log10(np.asarray(f["density"][rows]))      # (n, nbins)
        logT = np.log10(np.asarray(f["temp"][rows]))
        sk_w = np.ones(rows.size)

        def fit(weights_sk):
            w = np.repeat(weights_sk, logd.shape[1])
            ld, lt = logd.ravel(), logT.ravel()

            def tdr(sel):
                med = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (ld >= lo) & (ld < hi) & (w > 0) & sel
                    order = np.argsort(lt[m])
                    cw = np.cumsum(w[m][order])
                    med.append(lt[m][order][np.searchsorted(cw, 0.5 * cw[-1])])
                slope, icpt = np.polyfit(centers, med, 1)
                return 10 ** icpt, slope + 1.0

            T0, gam = tdr(np.ones_like(lt, dtype=bool))
            T0c, gamc = tdr(lt < 5.0)                     # cool phase only (T < 1e5 K)
            hot = float(np.sum(w * (lt > 5.0)) / np.sum(w))
            dense = ld > 1.0
            hot_dense = float(np.sum((w * (lt > 5.0))[dense]) / max(np.sum(w[dense]), 1e-12))
            mean_d = (ld >= -0.5) & (ld < 0.5)
            hot_mean = float(np.sum((w * (lt > 5.0))[mean_d]) / max(np.sum(w[mean_d]), 1e-12))
            return T0, gam, hot, hot_dense, T0c, gamc, hot_mean

        T0, gam, hot, hot_dense, T0c, gamc, hot_mean = fit(sk_w)
        bt = np.array([fit(Wi[inv]) for Wi in W])
        boot[c] = bt
        hist, he = np.histogram(logT.ravel(), bins=np.arange(3.0, 7.01, 0.1))
        res[f"class{c}"] = {"T0": float(T0), "gamma": float(gam), "frac_T_gt_1e5": hot,
                            "frac_T_gt_1e5_at_Delta_gt_10": hot_dense, "T0_cool": float(T0c),
                            "gamma_cool": float(gamc), "frac_T_gt_1e5_near_mean_density": hot_mean,
                            "logT_pdf": {"edges": he.tolist(), "frac": (hist / hist.sum()).tolist()}}
    for a, b in ((2, 1), (3, 1), (4, 1), (3, 2), (4, 3)):
        r = boot[a][:, 0] / boot[b][:, 0] - 1.0
        pt = res[f"class{a}"]["T0"] / res[f"class{b}"]["T0"] - 1.0
        res[f"dT0_rel_{a}{b}"] = {"value": pt, "ci95_lo": float(np.percentile(r, 2.5)), "ci95_hi": float(np.percentile(r, 97.5))}
        dg = boot[a][:, 1] - boot[b][:, 1]
        res[f"dgamma_{a}{b}"] = {"value": res[f"class{a}"]["gamma"] - res[f"class{b}"]["gamma"],
                                 "ci95_lo": float(np.percentile(dg, 2.5)), "ci95_hi": float(np.percentile(dg, 97.5))}
        rc = boot[a][:, 4] / boot[b][:, 4] - 1.0
        res[f"dT0cool_rel_{a}{b}"] = {"value": res[f"class{a}"]["T0_cool"] / res[f"class{b}"]["T0_cool"] - 1.0,
                                      "ci95_lo": float(np.percentile(rc, 2.5)), "ci95_hi": float(np.percentile(rc, 97.5))}
        dm = boot[a][:, 6] - boot[b][:, 6]
        res[f"dhot_mean_{a}{b}"] = {"value": res[f"class{a}"]["frac_T_gt_1e5_near_mean_density"] - res[f"class{b}"]["frac_T_gt_1e5_near_mean_density"],
                                    "ci95_lo": float(np.percentile(dm, 2.5)), "ci95_hi": float(np.percentile(dm, 97.5))}
        dh = boot[a][:, 3] - boot[b][:, 3]
        res[f"dhot_dense_{a}{b}"] = {"value": res[f"class{a}"]["frac_T_gt_1e5_at_Delta_gt_10"] - res[f"class{b}"]["frac_T_gt_1e5_at_Delta_gt_10"],
                                     "ci95_lo": float(np.percentile(dh, 2.5)), "ci95_hi": float(np.percentile(dh, 97.5))}
    return res


# ---------------------------------------------------------------------------
# T3 — UVB/Gamma degeneracy: CDDF slope and match, flux PDF, P_F(k) shape after CDDF matching
# ---------------------------------------------------------------------------
def cddf_stats(cats: List[Dict[str, np.ndarray]], lo: float = 12.5, hi: float = 14.5,
               dbin: float = 0.1) -> Dict[str, object]:
    """Per class: CDDF slope beta (dN/dlogN ∝ N^(1-beta), fit over [lo, hi]) and the CDDF ratio
    to class 1 over [13.3, 14.5] (chi^2 per bin, Poisson)."""
    edges = np.arange(lo, hi + 1e-9, dbin)
    cen = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    ref = np.histogram(cats[0]["logN"], np.arange(13.3, 14.5 + 1e-9, dbin))[0].astype(float)
    for c, cat in enumerate(cats, start=1):
        h = np.histogram(cat["logN"], edges)[0].astype(float)
        slope = np.polyfit(cen, np.log10(np.maximum(h, 1)), 1)[0]
        hm = np.histogram(cat["logN"], np.arange(13.3, 14.5 + 1e-9, dbin))[0].astype(float)
        out[f"class{c}"] = {"beta": float(1.0 - slope),
                            "chi2_per_bin_vs_class1_13.3-14.5": float(np.mean((hm - ref) ** 2 / np.maximum(ref, 1))),
                            "count_ratio_vs_class1_13.3-14.5": float(hm.sum() / ref.sum())}
    return out


def flux_shape_stats(A: Sequence[float], n_skewers: int = 4096, seed: int = 0, n_boot: int = 300
                     ) -> Dict[str, object]:
    """For tau-rescaled (factors A) flux: KS distance of the pixel flux PDF vs class 1, and the
    per-sightline P_F(k) ratio P_c/P_1 (each class normalised by its own <F>) in k bins with a
    paired block bootstrap over the same skewers."""
    from scipy.stats import ks_2samp
    from src.core.eigen_voigt import load_flux_variant
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(16384, n_skewers, replace=False))
    blocks = skewer_blocks()[rows]
    ub, inv = np.unique(blocks, return_inverse=True)
    W = np.stack([np.bincount(np.random.default_rng(seed + 1 + i).integers(0, ub.size, ub.size), minlength=ub.size)
                  for i in range(n_boot)]).astype(np.float64)[:, inv]          # (n_boot, n_skewers)
    dv = 2.6365028403447024
    k = 2 * np.pi * np.fft.rfftfreq(2048, d=dv)[1:]                           # s/km
    kedges = np.geomspace(k[0], k[-1], 13)
    kb = np.clip(np.digitize(k, kedges) - 1, 0, 11)
    P, F = {}, {}
    for c in range(1, N_CLASSES + 1):
        f = load_flux_variant("taurescaled", c, rows=rows, factors=list(A))
        F[c] = f
        d = f / f.mean() - 1.0
        p = np.abs(np.fft.rfft(d, axis=1)[:, 1:]) ** 2
        P[c] = np.stack([p[:, kb == j].mean(axis=1) for j in range(12)], axis=1)   # (n_skewers, 12)
    kc = np.array([k[kb == j].mean() for j in range(12)])
    out: Dict[str, object] = {"A": list(A), "k_centers_s_per_km": kc.tolist()}
    for c in range(2, N_CLASSES + 1):
        ks = ks_2samp(F[c].ravel()[::7], F[1].ravel()[::7]).statistic
        ratio = P[c].mean(axis=0) / P[1].mean(axis=0)
        boot = (W @ P[c]) / (W @ P[1])                                        # (n_boot, 12)
        hi_k = kc > 0.04
        dev_hi = (W @ P[c])[:, hi_k].sum(axis=1) / (W @ P[1])[:, hi_k].sum(axis=1) - 1.0
        out[f"class{c}_vs_1"] = {
            "flux_pdf_ks": float(ks), "pk_ratio": ratio.tolist(),
            "pk_ratio_ci95_lo": np.percentile(boot, 2.5, axis=0).tolist(),
            "pk_ratio_ci95_hi": np.percentile(boot, 97.5, axis=0).tolist(),
            "pk_ratio_k_gt_0.04_minus1": _summ(float(P[c].mean(axis=0)[hi_k].sum() / P[1].mean(axis=0)[hi_k].sum() - 1.0), dev_hi)}
    return out


# ---------------------------------------------------------------------------
# EDGE_MAP checks (E1 nuisance degeneracy, E3 T0 cut robustness, E4 forward-modelled P_F(k))
# ---------------------------------------------------------------------------
def profile_chi2(dmedb: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """chi^2 of a binned Delta median b profile against zero (paired block-bootstrap SE per bin)."""
    terms = []
    for v in dmedb.values():
        se = (v["ci95_hi"] - v["ci95_lo"]) / (2 * 1.96)
        if se > 0 and np.isfinite(v["value"]):
            terms.append((v["value"] / se) ** 2)
    from scipy.stats import chi2 as _chi2
    x = float(np.sum(terms))
    return {"chi2": x, "dof": len(terms), "p": float(_chi2.sf(x, len(terms)))}


def _weighted_median_boot(x: np.ndarray, w_base: np.ndarray, sk: np.ndarray, Wsk: np.ndarray,
                          q: float = 0.5) -> Tuple[float, np.ndarray]:
    """Quantile q of x with pixel weights w_base (point) and w_base * Wsk[i, sk] (bootstrap rows)."""
    o = np.argsort(x, kind="stable")
    xs, wo, so = x[o], w_base[o], sk[o]
    out = []
    for w in [wo] + [wo * Wi[so] for Wi in Wsk]:
        cw = np.cumsum(w)
        out.append(xs[min(np.searchsorted(cw, q * cw[-1]), xs.size - 1)])
    return float(out[0]), np.array(out[1:])


def t0_cut_sweep(n_skewers_sample: int = 2048, seed: int = 0, cuts: Sequence[float] = (4.5, 4.75, 5.0, 5.5),
                 n_boot: int = 100, logd_lo: float = -0.5, logd_hi: float = 0.5) -> Dict[str, object]:
    """Robustness of the cool-phase T0 shift (EDGE_MAP E3): T0/gamma from the median-T(rho) fit restricted to
    log T < cut, with volume (pixel) or density weighting, plus two cut-free estimators at mean density
    (|log Delta| < 0.05): the cool-phase mode of log T and the 10th percentile of log T. Ratios vs class 1 with
    a paired block bootstrap over the same skewers. HI-weighted T, as carried by the LOS files."""
    rng = np.random.default_rng(seed)
    blocks = skewer_blocks()
    rows = np.sort(rng.choice(16384, n_skewers_sample, replace=False))
    ub, inv = np.unique(blocks[rows], return_inverse=True)
    Wsk = np.stack([np.bincount(np.random.default_rng(seed + 1 + i).integers(0, ub.size, ub.size), minlength=ub.size)
                    for i in range(n_boot)]).astype(np.float64)[:, inv]           # (n_boot, n_skewers)
    edges = np.arange(logd_lo, logd_hi + 1e-9, 0.1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hbins = np.arange(3.0, 4.6 + 1e-9, 0.02)                                  # cool-phase window for the mode
    est: Dict[int, Dict[str, np.ndarray]] = {}
    for c in range(1, N_CLASSES + 1):
        f = read_los_fields(c)
        ld = np.log10(np.asarray(f["density"][rows]))
        lt = np.log10(np.asarray(f["temp"][rows]))
        sk = np.repeat(np.arange(rows.size), ld.shape[1])
        ld, lt = ld.ravel(), lt.ravel()
        inr = (ld >= logd_lo) & (ld < logd_hi)
        ld, lt, sk = ld[inr], lt[inr], sk[inr]
        e: Dict[str, np.ndarray] = {}
        for weighting in ("volume", "density"):
            wb = np.ones_like(ld) if weighting == "volume" else 10.0 ** ld
            for cut in cuts:
                sel = lt < cut
                meds = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = sel & (ld >= lo) & (ld < hi)
                    p, b = _weighted_median_boot(lt[m], wb[m], sk[m], Wsk)
                    meds.append(np.r_[p, b])
                meds = np.array(meds)                                             # (n_bins, 1 + n_boot)
                fits = np.polyfit(centers, meds, 1)                               # (2, 1 + n_boot)
                e[f"T0|{weighting}|cut{cut}"] = 10 ** fits[1]
                e[f"gamma|{weighting}|cut{cut}"] = fits[0] + 1.0
        md = np.abs(ld) < 0.05
        ltm, skm = lt[md], sk[md]
        idx = np.digitize(ltm, hbins) - 1
        inb = (idx >= 0) & (idx < hbins.size - 1)                                # cool-phase window only
        modes = []
        for w in [np.ones_like(ltm)] + [Wi[skm] for Wi in Wsk]:
            h = np.bincount(idx[inb], weights=w[inb], minlength=hbins.size - 1)
            h = np.convolve(h, np.ones(5) / 5, mode="same")
            modes.append(10 ** (hbins[np.argmax(h)] + 0.01))
        e["T_mode_mean_density"] = np.array(modes)
        p, b = _weighted_median_boot(ltm, np.ones_like(ltm), skm, Wsk, q=0.10)
        e["T_p10_mean_density"] = 10 ** np.r_[p, b]
        est[c] = e
    out: Dict[str, object] = {"n_skewers": int(rows.size), "cuts_logT": list(cuts), "per_class": {}, "rel_vs_class1": {}}
    for c in range(1, N_CLASSES + 1):
        out["per_class"][f"class{c}"] = {k: float(v[0]) for k, v in est[c].items()}
    for c in range(2, N_CLASSES + 1):
        rel = {}
        for k in est[c]:
            if k.startswith("gamma"):
                d = est[c][k] - est[1][k]
            else:
                d = est[c][k] / est[1][k] - 1.0
            rel[k] = {"value": float(d[0]), "ci95_lo": float(np.percentile(d[1:], 2.5)),
                      "ci95_hi": float(np.percentile(d[1:], 97.5))}
        out["rel_vs_class1"][f"class{c}"] = rel
    return out


def forward_pk_stats(A: Sequence[float], lsf: np.ndarray, snr: float = 30.0, cont_sigma: float = 0.01,
                     n_skewers: int = 4096, seed: int = 0, n_boot: int = 300) -> Dict[str, object]:
    """P_F(k) shape ratios P_c/P_1 after the T2 forward model (EDGE_MAP E4): LSF, rebin x3, continuum
    error, Gaussian noise (independent per class), each class normalised by its own <F>; the expected
    white-noise power n sigma^2 / <F>^2 is subtracted per mode. Paired block bootstrap over skewers."""
    from src.core.eigen_voigt import load_flux_variant, load_grid
    from src.core.forward import ForwardModel, ObsConfig
    wave, vel = load_grid()
    dv = float(vel[1] - vel[0])
    fm = ForwardModel(wave, dv, lsf, ObsConfig(snr=snr, cont_sigma=cont_sigma if np.isfinite(snr) else 0.0))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(16384, n_skewers, replace=False))
    blocks = skewer_blocks()[rows]
    ub, inv = np.unique(blocks, return_inverse=True)
    W = np.stack([np.bincount(np.random.default_rng(seed + 1 + i).integers(0, ub.size, ub.size), minlength=ub.size)
                  for i in range(n_boot)]).astype(np.float64)[:, inv]
    dv_obs = dv * fm.cfg.rebin
    n_obs = fm.obs_wave.size
    k = 2 * np.pi * np.fft.rfftfreq(n_obs, d=dv_obs)[1:]
    kedges = np.geomspace(k[0], k[-1], 13)
    kb = np.clip(np.digitize(k, kedges) - 1, 0, 11)
    P = {}
    for c in range(1, N_CLASSES + 1):
        f = load_flux_variant("taurescaled", c, rows=rows, factors=list(A))
        r = np.random.default_rng([seed, c])
        obs = [fm.observe(row, r) for row in f]
        fo = np.stack([o["flux"] for o in obs])                                   # shape: (n_skewers, 682)
        sg = np.stack([o["sigma"] for o in obs])
        mF = fo.mean()
        p = np.abs(np.fft.rfft(fo / mF - 1.0, axis=1)[:, 1:]) ** 2
        if np.isfinite(snr):
            p = p - (np.sum(sg ** 2, axis=1) / mF ** 2)[:, None]
        P[c] = np.stack([p[:, kb == j].mean(axis=1) for j in range(12)], axis=1)
    kc = np.array([k[kb == j].mean() for j in range(12)])
    out: Dict[str, object] = {"A": list(A), "snr": snr, "k_centers_s_per_km": kc.tolist()}
    bands = {"k_0.01-0.03": (kc >= 0.01) & (kc < 0.03), "k_0.03-0.04": (kc >= 0.03) & (kc < 0.04),
             "k_gt_0.04": kc > 0.04, "k_0.04-0.1": (kc > 0.04) & (kc < 0.1)}
    for c in range(2, N_CLASSES + 1):
        ent: Dict[str, object] = {"pk_ratio": (P[c].mean(axis=0) / P[1].mean(axis=0)).tolist()}
        for name, m in bands.items():
            if not m.any():
                continue
            pt = float(P[c].mean(axis=0)[m].sum() / P[1].mean(axis=0)[m].sum() - 1.0)
            bt = (W @ P[c])[:, m].sum(axis=1) / (W @ P[1])[:, m].sum(axis=1) - 1.0
            ent[name] = _summ(pt, bt)
        out[f"class{c}_vs_1"] = ent
    return out
