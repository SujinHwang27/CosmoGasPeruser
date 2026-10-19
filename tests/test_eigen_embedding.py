"""Tests for src/core/eigen_embedding.py and the split/score helpers in src/core/eigen_eval.py."""

import numpy as np
from scipy.stats import spearmanr

from src.core.eigen_embedding import LaplacianEigenmap, harmonic_residuals, knn_table, spectral_clusters
from src.core.eigen_eval import balanced_scores, group_splits


def _swiss_roll(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])     # shape: (n, 3)
    return X, t, h


def test_eigenmap_recovers_roll_coordinate_and_nystrom_is_consistent():
    X, t, _ = _swiss_roll(2500)
    idx, dist = knn_table(X, 40)
    rng = np.random.default_rng(1)
    perm = rng.permutation(X.shape[0])
    test, train = perm[:500], perm[500:]
    le = LaplacianEigenmap(n_components=6, n_neighbors=15, random_state=0)
    E_tr = le.fit_from_knn(idx, dist, train)
    E_te = le.transform_from_knn(idx, dist, test)
    assert E_tr.shape == (train.size, 6) and E_te.shape == (test.size, 6)
    assert np.isfinite(E_tr).all() and np.isfinite(E_te).all()
    assert abs(spearmanr(E_tr[:, 0], t[train])[0]) > 0.99
    assert abs(spearmanr(E_te[:, 0], t[test])[0]) > 0.99
    E_self = le.transform_from_knn(idx, dist, train[:200])
    assert np.corrcoef(E_self[:, 0], E_tr[:200, 0])[0, 1] > 0.999


def test_eigenmap_is_deterministic():
    X, _, _ = _swiss_roll(1500, seed=3)
    idx, dist = knn_table(X, 30)
    a = LaplacianEigenmap(4, 15, random_state=7).fit_from_knn(idx, dist, np.arange(1500))
    b = LaplacianEigenmap(4, 15, random_state=7).fit_from_knn(idx, dist, np.arange(1500))
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-8)


def test_group_mask_removes_same_group_edges():
    X, _, _ = _swiss_roll(1000, seed=2)
    groups = np.repeat(np.arange(500), 2)
    Xd = X[groups] + 1e-3 * np.random.default_rng(0).standard_normal((1000, 3))
    idx, dist = knn_table(Xd, 40)
    assert (groups[idx[:, 0]] == groups).mean() > 0.9         # twins dominate the raw graph
    le = LaplacianEigenmap(4, 15, mask_same_group=True)
    le.fit_from_knn(idx, dist, np.arange(1000), groups=groups)
    assert le.diagnostics_["frac_short_neighbour_lists"] == 0.0


def test_harmonic_residuals_flag_new_coordinate():
    X, _, h = _swiss_roll(5000, seed=0)
    idx, dist = knn_table(X, 40)
    E = LaplacianEigenmap(8, 15).fit_from_knn(idx, dist, np.arange(5000))
    r = harmonic_residuals(E, n_sample=800)
    j_h = int(np.argmax([abs(spearmanr(E[:, j], h)[0]) for j in range(8)]))
    assert r[0] == 1.0
    assert r[j_h] > 0.5
    assert np.median(np.delete(r, [0, j_h])) < 0.3


def test_spectral_clusters_label_count():
    X, _, _ = _swiss_roll(1200)
    idx, dist = knn_table(X, 30)
    E = LaplacianEigenmap(6, 15).fit_from_knn(idx, dist, np.arange(1200))
    lab = spectral_clusters(E, 4, seed=0)
    assert lab.shape == (1200,) and set(np.unique(lab)) == {0, 1, 2, 3}


def test_group_splits_never_leak_groups():
    groups = np.tile(np.arange(100), 4)
    for tr, te in group_splits(groups, 5, seed=0):
        assert np.intersect1d(groups[tr], groups[te]).size == 0
        assert tr.size + te.size == groups.size


def test_balanced_scores_perfect_and_pairs():
    y = np.repeat(np.arange(1, 5), 10)
    proba = np.eye(4)[y - 1]
    s = balanced_scores(y, proba, np.arange(1, 5))
    assert s["ba4"] == 1.0
    assert all(s[f"ba_{a}{b}"] == 1.0 for a, b in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
