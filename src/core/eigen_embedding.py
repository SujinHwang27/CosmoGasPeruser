"""
Graph-Laplacian eigenvector embeddings (Laplacian eigenmaps / spectral clustering).

Scientific purpose
------------------
PCA keeps the eigenvectors of the global covariance — linear directions of largest
variance. A Laplacian eigenmap keeps the slowest-varying eigenvectors of a k-nearest-
neighbour similarity graph — nonlinear coordinates that follow the local geometry of
the data (Belkin & Niyogi 2003), and K-Means on them is spectral clustering (Shi &
Malik 2000; Ng, Jordan & Weiss 2002; von Luxburg 2007). The eigen-voigt-embedding
track asks whether these coordinates, computed on raw sightlines or on Voigt-line
features, carry feedback-class information PCA at matched dimension does not.

Construction (all choices are parameters, recorded with every run):
  * kNN table computed once per representation on all points (label-free), then
    restricted per CV fold to TRAINING points, so the graph and eigenvectors are fit on
    the training fold only; held-out points are embedded by the Nystrom extension of
    the random-walk eigenvectors, phi_k(x) = (1/mu_k) sum_j p(x, j) phi_k(j).
  * Optional group mask: edges between points of the same group (the same Sherwood
    skewer in a different feedback class) are removed, so the eigenvectors cannot
    encode skewer identity.
  * Self-tuning Gaussian weights w_ij = exp(-d_ij^2 / (s_i s_j)), s_i = distance to the
    `local_scale_k`-th neighbour (Zelnik-Manor & Perona 2004); W symmetrised by max.
  * Eigenvectors of S = D^-1/2 W D^-1/2 (largest mu), mapped to random-walk eigenvectors
    psi = D^-1/2 u; the trivial constant eigenvector is dropped.
  * `harmonic_residuals` implements the local-linear-regression test of Dsilva et al.
    (2018): an eigenvector that is a smooth function of earlier ones (a harmonic) is not
    a new coordinate, so "d dimensions" counts only the non-harmonic ones.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from src.core.base import BaseTransformer


def knn_table(X: np.ndarray, k: int, n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Euclidean k nearest neighbours of every row of X, excluding the row itself.

    Returns idx (n, k) int64 and dist (n, k) float64, sorted by distance.
    """
    X = np.asarray(X, dtype=np.float32 if X.shape[1] > 256 else np.float64)
    if not np.isfinite(X).all():
        raise ValueError("knn_table: X contains NaN/Inf")
    algo = "brute" if X.shape[1] > 32 else "auto"
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm=algo, n_jobs=n_jobs).fit(X)
    dist, idx = nn.kneighbors(X)                            # shape: (n, k+1)
    n = X.shape[0]
    out_i = np.empty((n, k), dtype=np.int64)
    out_d = np.empty((n, k), dtype=np.float64)
    rows = np.arange(n)
    for r in range(n):                                      # drop self (usually column 0)
        keep = idx[r] != rows[r]
        out_i[r] = idx[r][keep][:k]
        out_d[r] = dist[r][keep][:k]
    return out_i, out_d


def _select_neighbours(knn_idx: np.ndarray, knn_dist: np.ndarray, query: np.ndarray,
                       allowed: np.ndarray, k: int, groups: Optional[np.ndarray]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First k neighbours of each query point that are `allowed` (and, if `groups` is
    given, not in the query's own group). Missing slots get index -1."""
    cand_i = knn_idx[query]                                 # shape: (n_q, K)
    cand_d = knn_dist[query]
    ok = allowed[cand_i]
    if groups is not None:
        ok &= groups[cand_i] != groups[query][:, None]
    order = np.argsort(~ok, axis=1, kind="stable")[:, :k]   # valid first, distance order kept
    sel_i = np.take_along_axis(cand_i, order, axis=1)
    sel_d = np.take_along_axis(cand_d, order, axis=1)
    sel_ok = np.take_along_axis(ok, order, axis=1)
    sel_i = np.where(sel_ok, sel_i, -1)
    n_valid = sel_ok.sum(axis=1)
    return sel_i, sel_d, n_valid


class LaplacianEigenmap(BaseTransformer):
    """Laplacian eigenmap fit on a training subset of a precomputed kNN table."""

    def __init__(self, n_components: int = 32, n_neighbors: int = 15, local_scale_k: int = 7,
                 mask_same_group: bool = False, random_state: int = 0) -> None:
        if local_scale_k > n_neighbors:
            raise ValueError("LaplacianEigenmap: local_scale_k must be <= n_neighbors")
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.local_scale_k = local_scale_k
        self.mask_same_group = mask_same_group
        self.random_state = random_state
        self.diagnostics_: Dict[str, float] = {}

    def _scale(self, sel_d: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
        kk = np.clip(np.minimum(n_valid, self.local_scale_k) - 1, 0, None)
        s = sel_d[np.arange(sel_d.shape[0]), kk]
        return np.maximum(s, 1e-12)

    def fit_from_knn(self, knn_idx: np.ndarray, knn_dist: np.ndarray, train: np.ndarray,
                     groups: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit on the points `train` (index array); returns their embedding (n_train, d)."""
        n_all = knn_idx.shape[0]
        train = np.asarray(train, dtype=np.int64)
        allowed = np.zeros(n_all, dtype=bool)
        allowed[train] = True
        g = groups if self.mask_same_group else None
        sel_i, sel_d, n_valid = _select_neighbours(knn_idx, knn_dist, train, allowed, self.n_neighbors, g)
        local = -np.ones(n_all, dtype=np.int64)
        local[train] = np.arange(train.size)
        s = self._scale(sel_d, n_valid)                     # shape: (n_train,)
        r_idx = np.repeat(np.arange(train.size), self.n_neighbors)
        c_glob = sel_i.ravel()
        valid = c_glob >= 0
        r_idx, c_loc, d = r_idx[valid], local[c_glob[valid]], sel_d.ravel()[valid]
        w = np.exp(-d ** 2 / (s[r_idx] * s[c_loc]))
        W = coo_matrix((w, (r_idx, c_loc)), shape=(train.size, train.size)).tocsr()
        W = W.maximum(W.T)                                  # symmetrise
        deg = np.asarray(W.sum(axis=1)).ravel()
        if np.any(deg <= 0):
            raise ValueError("LaplacianEigenmap: isolated vertex in the training graph")
        n_cc, _ = connected_components(W, directed=False)
        Dm = diags(1.0 / np.sqrt(deg))
        S = Dm @ W @ Dm
        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(train.size)
        mu, U = eigsh(S, k=self.n_components + 1, which="LA", v0=v0, tol=1e-8, maxiter=20000)
        order = np.argsort(mu)[::-1]
        mu, U = mu[order], U[:, order]
        psi = U / np.sqrt(deg)[:, None]                     # random-walk eigenvectors
        psi = psi / np.linalg.norm(psi, axis=0, keepdims=True) * np.sqrt(train.size)
        self.mu_ = mu[1:]                                   # shape: (d,)
        self.embedding_ = psi[:, 1:]                        # shape: (n_train, d)
        self.train_ = train
        self.local_ = local
        self.scale_train_ = s
        self.diagnostics_ = {"n_connected_components": float(n_cc),
                             "mu_trivial": float(mu[0]), "mu_last": float(mu[-1]),
                             "frac_short_neighbour_lists": float(np.mean(n_valid < self.n_neighbors)),
                             "n_edges": float(W.nnz)}
        return self.embedding_

    def transform_from_knn(self, knn_idx: np.ndarray, knn_dist: np.ndarray, query: np.ndarray,
                           groups: Optional[np.ndarray] = None) -> np.ndarray:
        """Nystrom-extend the fitted eigenvectors to `query` points (index array)."""
        n_all = knn_idx.shape[0]
        allowed = np.zeros(n_all, dtype=bool)
        allowed[self.train_] = True
        g = groups if self.mask_same_group else None
        query = np.asarray(query, dtype=np.int64)
        sel_i, sel_d, n_valid = _select_neighbours(knn_idx, knn_dist, query, allowed, self.n_neighbors, g)
        if np.any(n_valid == 0):
            raise ValueError("transform_from_knn: a query point has no training neighbour; "
                             "increase the kNN table width")
        s_q = self._scale(sel_d, n_valid)
        valid = sel_i >= 0
        loc = np.where(valid, self.local_[np.where(valid, sel_i, 0)], 0)
        w = np.where(valid, np.exp(-sel_d ** 2 / (s_q[:, None] * self.scale_train_[loc])), 0.0)
        p = w / w.sum(axis=1, keepdims=True)                # shape: (n_q, k)
        emb = np.einsum("qk,qkd->qd", p, self.embedding_[loc])
        return emb / self.mu_[None, :]                      # shape: (n_q, d)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        idx, dist = knn_table(X, k=self.n_neighbors)
        return self.fit_from_knn(idx, dist, np.arange(X.shape[0]))


def spectral_clusters(embedding: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Spectral clustering: K-Means on the first `n_clusters` non-trivial eigenvectors."""
    Z = np.asarray(embedding[:, :n_clusters], dtype=np.float64)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Z)


def harmonic_residuals(embedding: np.ndarray, n_sample: int = 1500, seed: int = 0) -> np.ndarray:
    """Dsilva et al. (2018) normalised leave-one-out local-linear-regression residual of
    each eigenvector on all earlier ones; r_1 = 1 by definition, r_j << 1 = harmonic.

    Kernel bandwidth is one third of the median pairwise distance in the regressor space,
    as in Dsilva et al.; a narrow (k-NN) kernel overfits and flags every vector harmonic.
    """
    rng = np.random.default_rng(seed)
    n, d = embedding.shape
    sub = rng.choice(n, size=min(n_sample, n), replace=False)
    E = np.asarray(embedding[sub], dtype=np.float64)        # shape: (m, d)
    m = E.shape[0]
    r = np.ones(d, dtype=np.float64)
    for j in range(1, d):
        A = E[:, :j]
        D2 = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1)   # shape: (m, m)
        eps = np.median(np.sqrt(D2[np.triu_indices(m, 1)])) / 3.0
        K = np.exp(-D2 / eps ** 2)
        np.fill_diagonal(K, 0.0)                            # leave-one-out
        pred = np.empty(m)
        for q in range(m):
            sw = np.sqrt(K[q])
            Xa = np.column_stack([np.ones(m), A - A[q]]) * sw[:, None]
            coef, *_ = np.linalg.lstsq(Xa, E[:, j] * sw, rcond=None)
            pred[q] = coef[0]
        r[j] = np.sqrt(np.sum((E[:, j] - pred) ** 2) / np.sum(E[:, j] ** 2))
    return r
