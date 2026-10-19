"""Tests for src/core/voigt.py — VoigtFit-driven decomposition and line statistics."""

import warnings

import numpy as np
import pytest

from src.core.voigt import (LYA_REST_A, VoigtSightlineFitter, _hybrid_voigt, line_feature_names,
                            line_statistics_features, tau_voigt)

N_PIX = 2048
DV = 2.6365028403447024
DLNW = 8.79537706e-06


def _grid():
    wave = 1580.37113 * np.exp(DLNW * np.arange(N_PIX))     # shape: (2048,)
    vel = DV * np.arange(N_PIX)
    return wave, vel


def _z_at(wave: np.ndarray, v_kms: float) -> float:
    return wave[0] * np.exp(v_kms / DV * DLNW) / LYA_REST_A - 1.0


def test_hybrid_profile_matches_exact_and_is_finite():
    wave, _ = _grid()
    z = wave[700] / LYA_REST_A - 1.0                         # line centre exactly on a pixel
    h = _hybrid_voigt(wave, LYA_REST_A, 0.416, 1e14, 2.5e6, 6.265e8, z=z)
    e = tau_voigt(wave, z, 25.0, 14.0)
    assert np.isfinite(h).all()
    assert np.max(np.abs(h - e) / np.maximum(e, 1e-300)) < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("v,b,logN", [(2000.0, 25.0, 13.5), (5395.0, 30.0, 14.0), (2700.0, 15.0, 16.0)])
def test_single_line_recovery_noiseless(v, b, logN):
    warnings.filterwarnings("ignore")
    wave, vel = _grid()
    period = np.exp(N_PIX * DLNW)
    z = _z_at(wave, v)
    tau = sum(tau_voigt(wave, (1 + z) * period ** s - 1, b, logN) for s in (-1, 0, 1))
    comps, diag = VoigtSightlineFitter(wave, vel).fit(np.exp(-tau))
    assert diag["converged"] == 1.0
    k = int(np.argmax(comps[:, 2]))
    assert abs(comps[k, 0] - v) < 0.5 or abs(abs(comps[k, 0] - v) - N_PIX * DV) < 0.5
    assert abs(comps[k, 1] - b) / b < 0.01
    assert abs(comps[k, 2] - logN) < 0.01


def test_fit_rejects_invalid_flux():
    wave, vel = _grid()
    fitter = VoigtSightlineFitter(wave, vel)
    with pytest.raises(ValueError):
        fitter.fit(np.full(N_PIX, np.nan))
    with pytest.raises(ValueError):
        fitter.fit(np.full(N_PIX, 1.5))


def test_line_statistics_features_shape_and_imputation():
    rng = np.random.default_rng(0)
    n = 5
    sl = np.concatenate([np.zeros(6, int), np.ones(1, int), np.full(4, 3)])   # sightlines 2, 4 empty
    v = rng.uniform(0, 5000, sl.size)
    b = rng.uniform(10, 60, sl.size)
    logN = rng.uniform(12.2, 14.5, sl.size)
    X = line_statistics_features(sl, v, b, logN, n_sightlines=n, span_kms=N_PIX * DV)
    names = line_feature_names()
    assert X.shape == (n, len(names))
    assert np.isfinite(X).all()
    col = {k: i for i, k in enumerate(names)}
    assert X[2, col["has_lines"]] == 0.0 and X[0, col["has_lines"]] == 1.0
    assert X[0, col["n_lines_12"]] == 6
    assert X[2, col["n_lines_12"]] == 0
