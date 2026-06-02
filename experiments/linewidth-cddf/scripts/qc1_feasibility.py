"""Q-C1 feasibility audit — τ-to-line-stats pipeline.

Three sub-tasks per experiments/linewidth-cddf/SCOPING.md §2 Q-C1:

  1. Empirical line-detection feasibility on real Sherwood τ (classes 1, 4).
  2. Voigt-fitting library landscape audit (this file: scipy.special.wofz +
     astropy.modeling.Voigt1D availability probe).
  3. b-parameter retrievability on synthetic Voigt profiles.

Run:
    PYTHONPATH=. uv run python experiments/linewidth-cddf/scripts/qc1_feasibility.py

Outputs are printed to stdout for capture by the auditing markdown.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

# Use absolute paths, no chdir.
DATA_BASE = r"d:\CosmoGasPeruser\data\preprocessed\Sherwood_z0.3_inf"
DELTA_V = 2.6365  # km/s per pixel (pk-feedback-classifier [D-04])
N_PIX = 2048
SEED = 42


# ---------------------------------------------------------------------------
# Voigt machinery
# ---------------------------------------------------------------------------

# Lyα atomic constants (CGS-ish, used as ratios)
LAMBDA0 = 1215.6701  # Angstrom
F_OSC = 0.4164
GAMMA = 6.265e8  # s^-1 (Einstein A → natural broadening)
C_KMS = 2.99792458e5  # km/s


def voigt_tau(v_kms: np.ndarray, v0: float, b: float, log_N: float) -> np.ndarray:
    """τ(v) for a single Lyα Voigt absorption line.

    v_kms: velocity grid (km/s)
    v0: line centre (km/s)
    b: Doppler parameter (km/s)
    log_N: log10(N_HI / cm^-2)

    Uses the Faddeeva function: τ(v) = (sqrt(π) e^2 / m_e c) * f * λ0 * N / b * Re[w(z)]
    Numerically we collapse all constants into a single prefactor calibrated so
    a (b=25, log_N=14) optically-thin line integrates to N · f · λ0 · (πe²/m_e c).

    To keep this self-contained, we build τ from the standard Hjerting H(a,u):
        u = (v - v0) / b
        a = (GAMMA λ0 / 4π c) / b   (dimensionless damping)
        H(a, u) = Re[w(u + i a)]
        τ0 (line-centre) = 1.4973e-15 (cm^2 km/s) · N_HI · f · λ0_Å / b_km/s
              ≈ 7.580e-13 · 10^log_N · f · λ0_Å / b
    """
    # Constant in (cm^2 km/s) units; standard Lyα formula.
    # tau0 = sqrt(pi) e^2 f λ0 / (m_e c b) · N_HI
    # numerical coefficient = 1.4973e-15 cm^2 km/s when λ in Å and b in km/s
    # (this is the standard "tau0" expression in Lyα-forest texts).
    tau0_coeff = 1.4973e-15  # cm^2 km/s
    N = 10.0 ** log_N
    tau0 = tau0_coeff * N * F_OSC * LAMBDA0 / b

    # Lyα wavelength in cm: λ0_cm = LAMBDA0 * 1e-8
    lambda0_cm = LAMBDA0 * 1e-8
    # damping parameter a = γ λ / (4π b) with b converted to cm/s
    b_cms = b * 1e5
    a = GAMMA * lambda0_cm / (4.0 * np.pi * b_cms)
    u = (v_kms - v0) / b
    z = u + 1j * a
    H = np.real(wofz(z))
    return tau0 * H


def fit_single_voigt(
    v: np.ndarray,
    tau: np.ndarray,
    v0_init: float,
    b_init: float = 25.0,
    log_N_init: float = 13.5,
):
    """Single-line Voigt fit; returns (b̂, log N̂, v̂0, success_flag).

    A narrow ±150 km/s window around v0_init is used to limit blending.
    """
    half_window = 150.0
    mask = (v >= v0_init - half_window) & (v <= v0_init + half_window)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan, False
    v_win = v[mask]
    tau_win = tau[mask]

    def model(v_, v0, b, log_N):
        return voigt_tau(v_, v0, b, log_N)

    try:
        popt, _pcov = curve_fit(
            model,
            v_win,
            tau_win,
            p0=[v0_init, b_init, log_N_init],
            bounds=([v0_init - 50, 5.0, 11.0], [v0_init + 50, 200.0, 17.0]),
            maxfev=2000,
        )
        return popt[1], popt[2], popt[0], True
    except Exception:
        return np.nan, np.nan, np.nan, False


# ---------------------------------------------------------------------------
# Sub-task 1: empirical line detection on Sherwood τ
# ---------------------------------------------------------------------------

def subtask1_real_tau_line_counts() -> dict:
    rng = np.random.default_rng(SEED)
    results = {}
    for cls in (1, 4):
        path = os.path.join(DATA_BASE, str(cls), "tau.npy")
        tau_all = np.load(path, mmap_mode="r")
        n_sl = tau_all.shape[0]
        idx = rng.choice(n_sl, size=10, replace=False)
        idx.sort()
        per_tau_counts: dict[float, list[int]] = {}
        for tau_min in (0.05, 0.1, 0.3):
            counts = []
            for i in idx:
                row = np.asarray(tau_all[i], dtype=np.float64)
                # find_peaks: prominence floor at tau_min, min separation 1 pixel.
                # Using height as additional gate so a low-prominence broad bump above tau_min still counts.
                peaks, _props = find_peaks(
                    row,
                    height=tau_min,
                    prominence=tau_min,
                    distance=1,
                )
                counts.append(int(len(peaks)))
            per_tau_counts[tau_min] = counts
        results[f"class_{cls}"] = {
            "indices": idx.tolist(),
            "counts": {f"{k}": v for k, v in per_tau_counts.items()},
            "stats": {
                f"{k}": {
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "min": int(np.min(v)),
                    "max": int(np.max(v)),
                }
                for k, v in per_tau_counts.items()
            },
        }
    return results


# ---------------------------------------------------------------------------
# Sub-task 2: library landscape (probe imports)
# ---------------------------------------------------------------------------

def subtask2_library_probe() -> dict:
    out = {}
    try:
        import scipy.special as _sp
        out["scipy.special.wofz"] = {"available": True, "version": __import__("scipy").__version__}
    except Exception as e:
        out["scipy.special.wofz"] = {"available": False, "error": str(e)}
    try:
        import astropy
        from astropy.modeling.models import Voigt1D
        out["astropy.modeling.Voigt1D"] = {"available": True, "version": astropy.__version__}
    except Exception as e:
        out["astropy.modeling.Voigt1D"] = {"available": False, "error": str(e)}
    try:
        import linetools  # noqa: F401
        out["linetools"] = {"available": True, "version": linetools.__version__}
    except Exception as e:
        out["linetools"] = {"available": False, "error": str(e)}
    return out


# ---------------------------------------------------------------------------
# Sub-task 3: synthetic b-recovery
# ---------------------------------------------------------------------------

def subtask3_synthetic_recovery() -> dict:
    """Generate 10 noise realizations on a 5×5 (b, log_N) grid; fit each."""
    v = np.arange(N_PIX) * DELTA_V  # 0 .. 5396 km/s
    b_grid = [15.0, 25.0, 40.0, 60.0, 100.0]
    logN_grid = [13.0, 13.5, 14.0, 14.5, 15.0]
    v0 = (N_PIX // 2) * DELTA_V  # centre of grid

    rng = np.random.default_rng(SEED)
    # Sherwood "mean-preserving LSF/noise stage" is not pinned numerically in the
    # LEDGER (pk-feedback-classifier §2 says "small mean-preserving LSF/noise,
    # not bit-exact"). We use σ_F = 0.01 (1% Gaussian on flux δ_F) as a
    # representative conservative case AND flag this assumption in the report.
    SIGMA_F = 0.01
    n_real = 10

    cells = []
    for b_true in b_grid:
        for logN_true in logN_grid:
            tau_clean = voigt_tau(v, v0, b_true, logN_true)
            flux_clean = np.exp(-tau_clean)
            b_hat_list = []
            logN_hat_list = []
            for _ in range(n_real):
                noise = rng.normal(0.0, SIGMA_F, size=v.shape)
                flux_noisy = np.clip(flux_clean + noise, 1e-6, None)
                tau_noisy = -np.log(flux_noisy)
                b_hat, logN_hat, _v0_hat, ok = fit_single_voigt(
                    v, tau_noisy, v0_init=v0, b_init=25.0, log_N_init=13.5
                )
                if ok and np.isfinite(b_hat):
                    b_hat_list.append(b_hat)
                    logN_hat_list.append(logN_hat)
            if not b_hat_list:
                cells.append(
                    {
                        "b_true": b_true,
                        "logN_true": logN_true,
                        "n_ok": 0,
                        "b_bias_pct": None,
                        "b_std": None,
                        "logN_bias": None,
                    }
                )
                continue
            b_hat_arr = np.array(b_hat_list)
            logN_hat_arr = np.array(logN_hat_list)
            cells.append(
                {
                    "b_true": b_true,
                    "logN_true": logN_true,
                    "n_ok": len(b_hat_list),
                    "b_mean": float(np.mean(b_hat_arr)),
                    "b_bias_pct": float(100.0 * (np.mean(b_hat_arr) - b_true) / b_true),
                    "b_std": float(np.std(b_hat_arr, ddof=0)),
                    "logN_mean": float(np.mean(logN_hat_arr)),
                    "logN_bias": float(np.mean(logN_hat_arr) - logN_true),
                }
            )
    return {"sigma_F_assumed": SIGMA_F, "n_realizations": n_real, "cells": cells}


def main():
    print("=" * 70)
    print("Q-C1 FEASIBILITY AUDIT")
    print("=" * 70)

    print("\n--- Sub-task 1: real τ line counts ---")
    s1 = subtask1_real_tau_line_counts()
    print(json.dumps(s1, indent=2))

    print("\n--- Sub-task 2: library probe ---")
    s2 = subtask2_library_probe()
    print(json.dumps(s2, indent=2))

    print("\n--- Sub-task 3: synthetic b-recovery ---")
    s3 = subtask3_synthetic_recovery()
    print(json.dumps(s3, indent=2))

    print("\n" + "=" * 70)
    print("VERDICT INPUTS")
    print("=" * 70)
    # quick summary numbers
    mean_lines_005 = {
        cls: np.mean(s1[cls]["counts"]["0.05"]) for cls in s1
    }
    print(f"mean lines/sightline @τ_min=0.05: {mean_lines_005}")
    rep_cell = [c for c in s3["cells"] if c["b_true"] == 25.0 and c["logN_true"] == 14.0]
    if rep_cell:
        print(f"representative cell (b=25, logN=14): {rep_cell[0]}")


if __name__ == "__main__":
    main()
