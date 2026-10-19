"""
eigen-voigt-embedding track: evaluation harness.

Scientific purpose
------------------
Measures whether graph-Laplacian eigenvector embeddings carry per-sightline
feedback-class information beyond PCA at matched dimension, on three inputs (raw flux,
Voigt line statistics, P_F(k)), under the controls the pre-registration names (LEDGER
§3 [D-04], amended by the defense-panel review [D-07]):

  * Leakage: sightline index i is the same skewer in all 4 classes, and skewers are
    randomly placed (~0.8 Mpc/h apart) so neighbours cross the same structures. CV
    folds are therefore SPATIAL BLOCKS (whole skewers, grouped by projection axis and a
    transverse cell); graphs, PCA bases, scalers, imputers and residualisers are fit on
    the training fold only. Training AND test rows are embedded by the same
    leave-self-out Nystrom map, so both see the same (masked) neighbourhood structure.
    Eigenmaps are built with and without same-skewer edges (masked = primary).
  * Mean-flux confound: scalar baselines <F> and [<F>, <tau>_clip]; mode resid=1 removes
    from every coordinate the part predictable by additive cubic splines of the ranked
    scalars (fold-internal, nonparametric: it also removes harmonics of rank <F>); mode
    resid=2 is the increment test, the classifier sees [scalars, embedding] so any gain
    over [scalars, PCA] is information beyond the scalars. The tau-rescaled data variant
    (equal class mean flux) is run by the caller.
  * Classifiers: multinomial logistic regression and a random forest (the RF delta is
    the geometry test; a logistic delta can come from nonlinearity alone). Metrics:
    4-class balanced accuracy and 6 pairwise BAs (pair decision = argmax of the 4-class
    probabilities restricted to the pair).

Unsupervised diagnostics (`cluster_diagnostics`): spectral clustering vs PCA+K-Means vs a
scalar-only (<F>, <tau>_clip) K-Means null — stability under block subsampling,
degeneracy, eta^2(<F>), AMI vs class and vs the v2 K=5 labels, cross-input AMI vs a
permutation null, eigenvector/<F> correlations, harmonic residuals, same-skewer
neighbour share.
"""

import itertools
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

from src.core.eigen_embedding import LaplacianEigenmap, harmonic_residuals, spectral_clusters

PAIRS: Tuple[Tuple[int, int], ...] = tuple(itertools.combinations((1, 2, 3, 4), 2))
DIMS: Tuple[int, ...] = (2, 4, 8, 16, 32)
LE_MODES: Tuple[str, ...] = ("le_masked", "le_unmasked")
RESID_MODES: Tuple[int, ...] = (0, 1, 2)      # none / spline-residualised / scalar-augmented
PRIMARY = {"le_mode": "le_masked", "clf": "rf", "d": 8}   # pre-declared primary cell [D-07]


def spatial_blocks(axis: np.ndarray, box_kpc: float, n_cells: int) -> np.ndarray:
    """Block id per skewer: projection axis x (n_cells x n_cells) transverse cell.

    `axis` rows are (axis id in {1,2,3}, x, y, z) in kpc/h; the two coordinates other
    than the projection axis are the skewer's transverse position.
    """
    ax = axis[:, 0].astype(np.int64)
    if not np.isin(ax, (1, 2, 3)).all():
        raise ValueError("spatial_blocks: axis ids must be 1, 2 or 3")
    pos = axis[:, 1:4]
    trans = np.stack([np.delete(pos[i], ax[i] - 1) for i in range(pos.shape[0])])   # shape: (n, 2)
    cell = np.clip((trans / box_kpc * n_cells).astype(np.int64), 0, n_cells - 1)
    return (ax - 1) * n_cells * n_cells + cell[:, 0] * n_cells + cell[:, 1]


def group_splits(groups: np.ndarray, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold over shuffled unique groups; returns (train_idx, test_idx) pairs."""
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(ug)
    fold_of = np.empty(ug.max() + 1, dtype=np.int64)
    fold_of[perm] = np.arange(perm.size) % n_folds
    f = fold_of[groups]
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(n_folds)]


def balanced_scores(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """4-class BA and the 6 pairwise BAs from 4-class probabilities."""
    out = {"ba4": float(balanced_accuracy_score(y_true, classes[np.argmax(proba, axis=1)]))}
    col = {c: i for i, c in enumerate(classes)}
    for a, b in PAIRS:
        m = (y_true == a) | (y_true == b)
        pick = np.where(proba[m, col[a]] >= proba[m, col[b]], a, b)
        out[f"ba_{a}{b}"] = float(balanced_accuracy_score(y_true[m], pick))
    return out


def _classifiers(seed: int, n_jobs: int) -> Dict[str, Callable[[], object]]:
    return {
        "logreg": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "rf": lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_samples=0.5,
                                             max_features="sqrt", n_jobs=n_jobs, random_state=seed),
    }


class _SplineResidualiser:
    """Fold-internal additive cubic-spline regression on the ranked confound scalars.

    Ranks (uniform quantile transform) make the basis density-adaptive; 48 knots resolve
    harmonics of rank <F> up to order ~20, beyond which resid=2 (augmentation) is the
    control.
    """

    def __init__(self, n_knots: int = 48) -> None:
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", subsample=10**6)
        self.sp = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
        self.reg = LinearRegression()

    def fit(self, S: np.ndarray, Z: np.ndarray) -> "_SplineResidualiser":
        self.reg.fit(self.sp.fit_transform(self.qt.fit_transform(S)), Z)
        return self

    def residual(self, S: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return Z - self.reg.predict(self.sp.transform(self.qt.transform(S)))


def _impute_train_median(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not (np.isnan(Xtr).any() or np.isnan(Xte).any()):
        return Xtr, Xte
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def evaluate_split(reps: Dict[str, Dict[str, np.ndarray]], scalars: np.ndarray, y: np.ndarray,
                   skewers: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int,
                   n_jobs: int = -1, raw_direct_rf: bool = False, keep_proba: bool = False,
                   log: Optional[Callable[[str], None]] = None
                   ) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, np.ndarray],
                              Dict[str, Dict[str, float]]]:
    """Every (input, method, d, resid mode, classifier) configuration on one split.

    `reps[name]` holds "X" (n, p; NaN = missing, imputed with the training-fold median),
    "knn_idx"/"knn_dist" tables and optionally "direct_only". Raw flux is PCA'd unscaled
    (pixels share units); the other inputs are z-scored first. `skewers` drives the
    same-skewer edge mask. Returns metric rows, out-of-fold labels (int8) and, if
    `keep_proba`, probabilities (float16) per configuration, and per-embedding diagnostics.
    """
    clfs = _classifiers(seed, n_jobs)
    S_tr, S_te = scalars[train], scalars[test]
    rows: List[Dict[str, object]] = []
    preds: Dict[str, np.ndarray] = {}
    probas: Dict[str, np.ndarray] = {}
    diags: Dict[str, Dict[str, float]] = {}

    def run(rep: str, method: str, Ztr: np.ndarray, Zte: np.ndarray, d: int, resid: int) -> None:
        if resid == 1:
            r = _SplineResidualiser().fit(S_tr, Ztr)
            Ztr, Zte = r.residual(S_tr, Ztr), r.residual(S_te, Zte)
        elif resid == 2:
            Ztr, Zte = np.column_stack([S_tr, Ztr]), np.column_stack([S_te, Zte])
        sc = StandardScaler().fit(Ztr)
        Ztr_s, Zte_s = sc.transform(Ztr), sc.transform(Zte)
        for cname, make in clfs.items():
            t0 = time.time()
            clf = make().fit(Ztr_s, y[train])
            proba = clf.predict_proba(Zte_s)
            key = f"{rep}|{method}|{d}|{resid}|{cname}"
            preds[key] = clf.classes_[np.argmax(proba, axis=1)].astype(np.int8)
            if keep_proba:
                probas[key] = proba.astype(np.float16)
            rows.append({"rep": rep, "method": method, "d": d, "resid": resid, "clf": cname,
                         "fit_s": round(time.time() - t0, 2), **balanced_scores(y[test], proba, clf.classes_)})

    for name, cols in (("meanF", [0]), ("meanF_meantau", [0, 1])):
        run("scalar", name, S_tr[:, cols], S_te[:, cols], len(cols), 0)

    for rep, R in reps.items():
        Xtr, Xte = _impute_train_median(R["X"][train], R["X"][test])
        if R.get("direct_only", False):
            for resid in RESID_MODES:
                run(rep, "direct", Xtr, Xte, Xtr.shape[1], resid)
            continue
        if rep == "raw":
            pca = PCA(n_components=max(DIMS), svd_solver="randomized", random_state=seed)
            Ptr = pca.fit_transform(Xtr)
            Pte = pca.transform(Xte)
        else:
            sc = StandardScaler().fit(Xtr)
            ncomp = min(max(DIMS), Xtr.shape[1])
            pca = PCA(n_components=ncomp, random_state=seed).fit(sc.transform(Xtr))
            Ptr, Pte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
        diags[f"{rep}|pca"] = {"evr_cum_" + str(d): float(np.sum(pca.explained_variance_ratio_[:d]))
                               for d in DIMS if d <= Ptr.shape[1]}
        embs = {"pca": (Ptr, Pte)}
        for mode in LE_MODES:
            le = LaplacianEigenmap(n_components=max(DIMS), n_neighbors=15, local_scale_k=7,
                                   mask_same_group=(mode == "le_masked"), random_state=seed)
            E_in = le.fit_from_knn(R["knn_idx"], R["knn_dist"], train, groups=skewers)
            Etr = le.transform_from_knn(R["knn_idx"], R["knn_dist"], train, groups=skewers)
            Ete = le.transform_from_knn(R["knn_idx"], R["knn_dist"], test, groups=skewers)
            embs[mode] = (Etr, Ete)
            dg = dict(le.diagnostics_)
            dg["spearman_phi1_meanF"] = float(spearmanr(Etr[:, 0], S_tr[:, 0])[0])
            dg["corr_insample_vs_nystrom_phi1"] = float(np.corrcoef(E_in[:, 0], Etr[:, 0])[0, 1])
            diags[f"{rep}|{mode}"] = dg
        for method, (Ztr, Zte) in embs.items():
            for d in DIMS:
                if d > Ztr.shape[1]:
                    continue
                for resid in RESID_MODES:
                    run(rep, method, Ztr[:, :d], Zte[:, :d], d, resid)
        if rep != "raw":
            for resid in RESID_MODES:
                run(rep, "direct", Xtr, Xte, Xtr.shape[1], resid)
        elif raw_direct_rf:
            t0 = time.time()
            rf = _classifiers(seed, n_jobs)["rf"]().fit(Xtr, y[train])
            proba = rf.predict_proba(Xte)
            key = f"raw|direct|{Xtr.shape[1]}|0|rf"
            preds[key] = rf.classes_[np.argmax(proba, axis=1)].astype(np.int8)
            if keep_proba:
                probas[key] = proba.astype(np.float16)
            rows.append({"rep": "raw", "method": "direct", "d": Xtr.shape[1], "resid": 0, "clf": "rf",
                         "fit_s": round(time.time() - t0, 2), **balanced_scores(y[test], proba, rf.classes_)})
        if log:
            log(f"  {rep}: done")
    return rows, preds, probas, diags


def block_bootstrap(y: np.ndarray, blocks: np.ndarray, correct: np.ndarray, n_boot: int = 2000,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap BA over spatial blocks for several configurations at once.

    `correct` (n_configs, n) holds per-sightline correctness in [0, 1] (seed-averaged if
    desired). Every block holds whole skewers, i.e. equal counts per class, so under block
    resampling BA = sum_b w_b k_b / (n_classes sum_b w_b n_b), with k_b the summed
    correctness in block b. All configurations share the same weights (for max-T bands).
    Returns (point BA per config, bootstrap BA matrix (n_boot, n_configs)).
    """
    ub, inv = np.unique(blocks, return_inverse=True)
    n_cls = np.unique(y).size
    per_block_cls = np.zeros((ub.size, n_cls))
    np.add.at(per_block_cls, (inv, np.searchsorted(np.unique(y), y)), 1.0)
    if not np.allclose(per_block_cls, per_block_cls[:, :1]):
        raise ValueError("block_bootstrap: every block must hold equal counts per class")
    n_b = per_block_cls[:, 0]                                # sightlines per class per block
    K = np.stack([np.bincount(inv, weights=c, minlength=ub.size) for c in np.atleast_2d(correct)])
    point = K.sum(axis=1) / (n_cls * n_b.sum())
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, K.shape[0]))
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, ub.size, ub.size), minlength=ub.size).astype(np.float64)
        boot[i] = (K @ w) / (n_cls * (w @ n_b))
    return point, boot


def max_t_band(point: np.ndarray, boot: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Simultaneous (1 - alpha) band over configurations from shared-weight bootstraps."""
    se = boot.std(axis=0, ddof=1)
    se = np.where(se > 0, se, np.inf)
    tmax = np.max(np.abs(boot - point[None, :]) / se[None, :], axis=1)
    c = np.quantile(tmax, 1 - alpha)
    return point - c * se, point + c * se


def corrected_resampled_t(diffs: np.ndarray, n_train: int, n_test: int) -> Dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on per-split score differences."""
    J = diffs.size
    m, v = float(np.mean(diffs)), float(np.var(diffs, ddof=1))
    se = np.sqrt((1.0 / J + n_test / n_train) * v)
    t = m / se if se > 0 else np.inf
    return {"mean": m, "t": float(t), "p_one_sided": float(student_t.sf(t, J - 1)), "n_splits": J}


def _eta2(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_tot = np.sum((values - grand) ** 2)
    ss_b = sum(np.sum(labels == k) * (values[labels == k].mean() - grand) ** 2 for k in np.unique(labels))
    return float(ss_b / ss_tot) if ss_tot > 0 else 0.0


def cluster_diagnostics(reps: Dict[str, Dict[str, np.ndarray]], scalars: np.ndarray, y: np.ndarray,
                        skewers: np.ndarray, blocks: np.ndarray, v2_labels: Optional[np.ndarray],
                        ks: Sequence[int] = (4, 5), d: int = 8, n_sub: int = 10, sub_frac: float = 0.8,
                        seed: int = 0, log: Optional[Callable[[str], None]] = None
                        ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Unsupervised diagnostics on the full (transductive, label-free) embeddings."""
    rng = np.random.default_rng(seed)
    n = y.size
    rows: List[Dict[str, object]] = []
    extra: Dict[str, object] = {}
    full_labels: Dict[Tuple[str, str, int], np.ndarray] = {}
    ub = np.unique(blocks)

    def score(rep: str, method: str, K: int, lab: np.ndarray, subs: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        aris = []
        for (ia, la), (ib, lb) in itertools.combinations(subs, 2):
            _, pa, pb = np.intersect1d(ia, ib, return_indices=True)
            aris.append(adjusted_rand_score(la[pa], lb[pb]))
        sizes = np.bincount(lab, minlength=K) / n
        rows.append({"rep": rep, "method": method, "K": K, "d": d,
                     "stability_ari_median": float(np.median(aris)),
                     "stability_ari_p16": float(np.percentile(aris, 16)),
                     "largest_cluster_frac": float(sizes.max()),
                     "eta2_meanF": _eta2(scalars[:, 0], lab), "eta2_meantau": _eta2(scalars[:, 1], lab),
                     "ami_class": float(adjusted_mutual_info_score(y, lab)),
                     "ami_v2_k5": (float(adjusted_mutual_info_score(v2_labels[skewers], lab))
                                   if v2_labels is not None else float("nan"))})

    def subsample() -> np.ndarray:
        keep = rng.choice(ub, size=int(sub_frac * ub.size), replace=False)
        return np.flatnonzero(np.isin(blocks, keep))

    Ssc = StandardScaler().fit_transform(scalars)
    for K in ks:                                              # <F>-only null
        lab = KMeans(K, n_init=10, random_state=seed).fit_predict(Ssc)
        subs = []
        for s in range(n_sub):
            idx = subsample()
            subs.append((idx, KMeans(K, n_init=10, random_state=seed + s + 1).fit_predict(Ssc[idx])))
        score("scalar", "kmeans_meanF_meantau", K, lab, subs)

    for rep, R in reps.items():
        if R.get("direct_only", False):
            continue
        extra[f"{rep}|same_skewer_neighbour_share_k15"] = float(
            (skewers[R["knn_idx"][:, :15]] == skewers[:, None]).mean())
        X = R["X"]
        if np.isnan(X).any():
            X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        Xs = X if rep == "raw" else StandardScaler().fit_transform(X)

        def embed(method: str, idx: np.ndarray) -> np.ndarray:
            if method == "pca":
                return PCA(n_components=d, svd_solver="randomized", random_state=seed).fit_transform(Xs[idx])
            le = LaplacianEigenmap(n_components=32, n_neighbors=15, local_scale_k=7,
                                   mask_same_group=True, random_state=seed)
            return le.fit_from_knn(R["knn_idx"], R["knn_dist"], idx, groups=skewers)[:, :d]

        for method in ("pca", "le_masked"):
            E = embed(method, np.arange(n))
            if method == "le_masked":
                le32 = LaplacianEigenmap(32, 15, 7, mask_same_group=True, random_state=seed)
                E32 = le32.fit_from_knn(R["knn_idx"], R["knn_dist"], np.arange(n), groups=skewers)
                extra[f"{rep}|le_masked|spearman_meanF"] = [float(spearmanr(E32[:, j], scalars[:, 0])[0]) for j in range(5)]
                extra[f"{rep}|le_masked|spearman_meantau"] = [float(spearmanr(E32[:, j], scalars[:, 1])[0]) for j in range(5)]
                hr = harmonic_residuals(E32, n_sample=1500, seed=seed)
                extra[f"{rep}|le_masked|harmonic_r"] = hr.tolist()
                extra[f"{rep}|le_masked|n_nonharmonic_r>0.5_first8"] = int(np.sum(hr[:8] > 0.5))
                extra[f"{rep}|le_masked|diag"] = dict(le32.diagnostics_)
            for K in ks:
                cl = (lambda Z, s: spectral_clusters(Z, K, s)) if method == "le_masked" else \
                     (lambda Z, s: KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                lab = cl(E, seed)
                full_labels[(rep, method, K)] = lab
                subs = []
                for s in range(n_sub):
                    idx = subsample()
                    subs.append((idx, cl(embed(method, idx), seed + s + 1)))
                score(rep, method, K, lab, subs)
            if log:
                log(f"  clusters {rep}/{method}: done")
    names = [r for r in reps if not reps[r].get("direct_only", False)]
    for K in ks:
        for a, b in itertools.combinations(names, 2):
            la, lb = full_labels[(a, "le_masked", K)], full_labels[(b, "le_masked", K)]
            obs = adjusted_mutual_info_score(la, lb)
            null = [adjusted_mutual_info_score(la, rng.permutation(lb)) for _ in range(200)]
            extra[f"cross_ami|{a}~{b}|K{K}"] = {"ami": float(obs), "null_p99": float(np.percentile(null, 99))}
    return rows, extra
