"""
eigen-voigt-embedding track: evaluation harness.

Scientific purpose
------------------
Measures whether graph-Laplacian eigenvector embeddings carry per-sightline
feedback-class information beyond PCA at matched dimension, on three inputs (raw flux,
Voigt line statistics, P_F(k)), under the controls the pre-registration names:

  * Leakage: sightline index i is the same skewer (shared initial conditions) in all 4
    classes, so every split is a group split on skewer index; graphs, PCA bases,
    scalers and residualisers are fit on the training fold only; held-out sightlines
    enter the eigenmap by Nystrom extension. Eigenmaps are built with and without
    same-skewer edges.
  * Mean-flux confound: (a) scalar baselines <F> and [<F>, <tau_eff>]; (b) every
    embedding re-tested after fold-internal residualisation on [<F>, <F>^2, <tau_eff>];
    (c) the whole pipeline re-run on the tau-rescaled variant (equal class mean flux),
    handled by the caller via the data `variant`.
  * Classifiers: multinomial logistic regression (linear probe) and a random forest.
    Metrics: 4-class balanced accuracy and the 6 pairwise balanced accuracies (the pair
    decision is the argmax of the 4-class probabilities restricted to the two classes).

Unsupervised diagnostics (`cluster_diagnostics`): spectral clustering vs PCA+K-Means
stability under skewer-group subsampling, cluster-size degeneracy, eta^2 of <F>, AMI
vs class and vs the signal-clustering-v2 K=5 labels, cross-input agreement against a
permutation null, eigenvector/<F> correlations, harmonic residuals, and the share of
graph neighbours that are copies of the same skewer.
"""

import itertools
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.core.eigen_embedding import LaplacianEigenmap, harmonic_residuals, spectral_clusters

PAIRS: Tuple[Tuple[int, int], ...] = tuple(itertools.combinations((1, 2, 3, 4), 2))
DIMS: Tuple[int, ...] = (2, 4, 8, 16, 32)
LE_MODES: Tuple[str, ...] = ("le_masked", "le_unmasked")


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


def _residualise(Z_tr: np.ndarray, Z_te: np.ndarray, C_tr: np.ndarray, C_te: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the part of each column linearly predictable from the confound basis C."""
    reg = LinearRegression().fit(C_tr, Z_tr)
    return Z_tr - reg.predict(C_tr), Z_te - reg.predict(C_te)


def _confound_basis(scalars: np.ndarray) -> np.ndarray:
    F, t = scalars[:, 0], scalars[:, 1]
    return np.column_stack([F, F ** 2, t])                   # shape: (n, 3)


def evaluate_split(reps: Dict[str, Dict[str, np.ndarray]], scalars: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int,
                   n_jobs: int = -1, raw_direct_rf: bool = False,
                   log: Optional[Callable[[str], None]] = None
                   ) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Every (input, method, d, residualised, classifier) configuration on one split.

    `reps[name]` holds "X" (n, p) features and "knn_idx"/"knn_dist" tables. Raw flux is
    PCA'd unscaled (pixels share units); the other inputs are z-scored first. Returns metric rows, out-of-fold predicted labels per configuration (int8, test order),
    and per-embedding diagnostics.
    """
    classes = np.arange(1, 5)
    clfs = _classifiers(seed, n_jobs)
    C = _confound_basis(scalars)
    rows: List[Dict[str, object]] = []
    preds: Dict[str, np.ndarray] = {}
    diags: Dict[str, Dict[str, float]] = {}

    def run(rep: str, method: str, Ztr: np.ndarray, Zte: np.ndarray, d: int, resid: bool) -> None:
        if resid:
            Ztr, Zte = _residualise(Ztr, Zte, C[train], C[test])
        sc = StandardScaler().fit(Ztr)
        Ztr_s, Zte_s = sc.transform(Ztr), sc.transform(Zte)
        for cname, make in clfs.items():
            t0 = time.time()
            clf = make().fit(Ztr_s, y[train])
            proba = clf.predict_proba(Zte_s)
            sc_ = balanced_scores(y[test], proba, clf.classes_)
            key = f"{rep}|{method}|{d}|{int(resid)}|{cname}"
            preds[key] = clf.classes_[np.argmax(proba, axis=1)].astype(np.int8)
            rows.append({"rep": rep, "method": method, "d": d, "resid": int(resid), "clf": cname,
                         "fit_s": round(time.time() - t0, 2), **sc_})

    # scalar confound baselines
    for name, cols in (("meanF", [0]), ("meanF_taueff", [0, 1])):
        run("scalar", name, scalars[train][:, cols], scalars[test][:, cols], len(cols), False)

    for rep, R in reps.items():
        X = R["X"]
        # PCA (fit on train only)
        if rep == "raw":
            pca = PCA(n_components=max(DIMS), svd_solver="randomized", random_state=seed)
            Ptr = pca.fit_transform(X[train])
            Pte = pca.transform(X[test])
        else:
            sc = StandardScaler().fit(X[train])
            ncomp = min(max(DIMS), X.shape[1])
            pca = PCA(n_components=ncomp, random_state=seed).fit(sc.transform(X[train]))
            Ptr, Pte = pca.transform(sc.transform(X[train])), pca.transform(sc.transform(X[test]))
        diags[f"{rep}|pca"] = {"evr_cum_" + str(d): float(np.sum(pca.explained_variance_ratio_[:d]))
                               for d in DIMS if d <= Ptr.shape[1]}
        embs = {"pca": (Ptr, Pte)}
        # Laplacian eigenmaps (graph on train only; Nystrom for test)
        for mode in LE_MODES:
            le = LaplacianEigenmap(n_components=max(DIMS), n_neighbors=15, local_scale_k=7,
                                   mask_same_group=(mode == "le_masked"), random_state=seed)
            Etr = le.fit_from_knn(R["knn_idx"], R["knn_dist"], train, groups=groups)
            Ete = le.transform_from_knn(R["knn_idx"], R["knn_dist"], test, groups=groups)
            embs[mode] = (Etr, Ete)
            diags[f"{rep}|{mode}"] = dict(le.diagnostics_)
            diags[f"{rep}|{mode}"]["spearman_phi1_meanF"] = float(spearmanr(Etr[:, 0], scalars[train, 0])[0])
        for method, (Ztr, Zte) in embs.items():
            for d in DIMS:
                if d > Ztr.shape[1]:
                    continue
                for resid in (False, True):
                    run(rep, method, Ztr[:, :d], Zte[:, :d], d, resid)
        if rep != "raw":
            for resid in (False, True):
                run(rep, "direct", X[train], X[test], X.shape[1], resid)
        elif raw_direct_rf:
            t0 = time.time()
            rf = _classifiers(seed, n_jobs)["rf"]().fit(X[train], y[train])
            proba = rf.predict_proba(X[test])
            preds[f"raw|direct|{X.shape[1]}|0|rf"] = rf.classes_[np.argmax(proba, axis=1)].astype(np.int8)
            rows.append({"rep": "raw", "method": "direct", "d": X.shape[1], "resid": 0, "clf": "rf",
                         "fit_s": round(time.time() - t0, 2), **balanced_scores(y[test], proba, rf.classes_)})
        if log:
            log(f"  {rep}: done")
    return rows, preds, diags


def group_bootstrap_delta(y: np.ndarray, groups: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over skewer groups of BA(a) - BA(b) from out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(ug.size)]
    sizes = np.array([m.size for m in members])
    flat = np.concatenate(members)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    point = balanced_accuracy_score(y, pred_a) - balanced_accuracy_score(y, pred_b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, ug.size, ug.size)
        idx = np.concatenate([flat[offs[g]:offs[g + 1]] for g in pick])
        deltas[i] = balanced_accuracy_score(y[idx], pred_a[idx]) - balanced_accuracy_score(y[idx], pred_b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}


def _eta2(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_tot = np.sum((values - grand) ** 2)
    ss_b = sum(np.sum(labels == k) * (values[labels == k].mean() - grand) ** 2 for k in np.unique(labels))
    return float(ss_b / ss_tot) if ss_tot > 0 else 0.0


def cluster_diagnostics(reps: Dict[str, Dict[str, np.ndarray]], scalars: np.ndarray, y: np.ndarray,
                        groups: np.ndarray, v2_labels: Optional[np.ndarray], ks: Sequence[int] = (4, 5),
                        d: int = 8, n_sub: int = 10, sub_frac: float = 0.8, seed: int = 0,
                        log: Optional[Callable[[str], None]] = None
                        ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Unsupervised diagnostics on the full (transductive) embeddings; see module doc."""
    rng = np.random.default_rng(seed)
    n = y.size
    rows: List[Dict[str, object]] = []
    extra: Dict[str, object] = {}
    full_labels: Dict[Tuple[str, str, int], np.ndarray] = {}
    ug = np.unique(groups)
    for rep, R in reps.items():
        same = (groups[R["knn_idx"][:, :15]] == groups[:, None]).mean()
        extra[f"{rep}|same_skewer_neighbour_share_k15"] = float(same)
        Xs = R["X"] if rep == "raw" else StandardScaler().fit_transform(R["X"])

        def embed(method: str, idx: np.ndarray) -> np.ndarray:
            if method == "pca":
                return PCA(n_components=d, svd_solver="randomized", random_state=seed).fit_transform(Xs[idx])
            le = LaplacianEigenmap(n_components=max(d, 32), n_neighbors=15, local_scale_k=7,
                                   mask_same_group=True, random_state=seed)
            return le.fit_from_knn(R["knn_idx"], R["knn_dist"], idx, groups=groups)[:, :d]

        all_idx = np.arange(n)
        for method in ("pca", "le_masked"):
            E = embed(method, all_idx)
            if method == "le_masked":
                le32 = LaplacianEigenmap(32, 15, 7, mask_same_group=True, random_state=seed)
                E32 = le32.fit_from_knn(R["knn_idx"], R["knn_dist"], all_idx, groups=groups)
                extra[f"{rep}|le_masked|spearman_meanF"] = [float(spearmanr(E32[:, j], scalars[:, 0])[0]) for j in range(5)]
                extra[f"{rep}|le_masked|spearman_taueff"] = [float(spearmanr(E32[:, j], scalars[:, 1])[0]) for j in range(5)]
                extra[f"{rep}|le_masked|harmonic_r"] = harmonic_residuals(E32, n_sample=1500, seed=seed).tolist()
                extra[f"{rep}|le_masked|diag"] = dict(le32.diagnostics_)
            for K in ks:
                lab = (spectral_clusters(E, K, seed) if method == "le_masked"
                       else KMeans(K, n_init=10, random_state=seed).fit_predict(E))
                full_labels[(rep, method, K)] = lab
                subs = []
                for s in range(n_sub):
                    keep_g = rng.choice(ug, size=int(sub_frac * ug.size), replace=False)
                    idx = np.flatnonzero(np.isin(groups, keep_g))
                    Es = embed(method, idx)
                    ls = (spectral_clusters(Es, K, seed + s + 1) if method == "le_masked"
                          else KMeans(K, n_init=10, random_state=seed + s + 1).fit_predict(Es))
                    subs.append((idx, ls))
                aris = []
                for (ia, la), (ib, lb) in itertools.combinations(subs, 2):
                    common, pa, pb = np.intersect1d(ia, ib, return_indices=True)
                    aris.append(adjusted_rand_score(la[pa], lb[pb]))
                sizes = np.bincount(lab, minlength=K) / n
                rows.append({"rep": rep, "method": method, "K": K, "d": d,
                             "stability_ari_median": float(np.median(aris)),
                             "stability_ari_p16": float(np.percentile(aris, 16)),
                             "largest_cluster_frac": float(sizes.max()),
                             "eta2_meanF": _eta2(scalars[:, 0], lab),
                             "eta2_taueff": _eta2(scalars[:, 1], lab),
                             "ami_class": float(adjusted_mutual_info_score(y, lab)),
                             "ami_v2_k5": (float(adjusted_mutual_info_score(v2_labels[groups], lab))
                                           if v2_labels is not None else float("nan"))})
            if log:
                log(f"  clusters {rep}/{method}: done")
    # cross-input agreement of spectral clusterings vs a permutation null
    names = [r for r in reps]
    for K in ks:
        for a, b in itertools.combinations(names, 2):
            la, lb = full_labels[(a, "le_masked", K)], full_labels[(b, "le_masked", K)]
            obs = adjusted_mutual_info_score(la, lb)
            null = [adjusted_mutual_info_score(la, rng.permutation(lb)) for _ in range(200)]
            extra[f"cross_ami|{a}~{b}|K{K}"] = {"ami": float(obs), "null_p99": float(np.percentile(null, 99))}
    return rows, extra
