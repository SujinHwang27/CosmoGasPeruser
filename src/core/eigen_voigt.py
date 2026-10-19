"""
eigen-voigt-embedding track: data variants and per-sightline representations.

Scientific purpose
------------------
Supplies the inputs the track compares: the fiducial noiseless Sherwood flux, the
tau-rescaled control (each class's optical depth scaled by a constant A_c so that every
class has the SAME global mean flux — the standard way of absorbing a UV-background /
photoionization-rate difference, and the strongest test that a feedback signal is not
the known mean-flux offset re-derived), the sightline-position groups that every CV
split must respect (sightline index i is the same skewer through shared initial
conditions in all four classes), and the per-sightline representations the
eigenvector embeddings are built on (raw flux, Voigt line statistics, P_F(k)) plus the
scalar confound baselines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from src.core.data import SignalClusteringData
from src.core.transforms import FluxPowerSpectrumTransform

FLUX_BASE = SignalClusteringData.FLUX_BASE
N_CLASSES = 4
VARIANTS = ("fiducial", "taurescaled")


def load_grid(base: str = FLUX_BASE) -> Tuple[np.ndarray, np.ndarray]:
    """Shared wavelength (Å) and velocity (km/s) grids; identical across classes."""
    wave = np.load(Path(base) / "1" / "wave.npy")           # shape: (n_pix,)
    vel = np.load(Path(base) / "1" / "vel.npy")             # shape: (n_pix,)
    for c in range(2, N_CLASSES + 1):
        if not (np.array_equal(wave, np.load(Path(base) / str(c) / "wave.npy"))
                and np.array_equal(vel, np.load(Path(base) / str(c) / "vel.npy"))):
            raise ValueError(f"load_grid: class {c} grid differs from class 1")
    return wave, vel


def _mean_flux_scaled(tau: np.ndarray, a: float, block: int = 2048) -> float:
    tot = 0.0
    for s in range(0, tau.shape[0], block):
        tot += float(np.exp(-a * np.asarray(tau[s:s + block])).sum())
    return tot / tau.size


def tau_rescale_factors(base: str = FLUX_BASE) -> Dict[str, object]:
    """A_c such that <exp(-A_c tau_c)> equals the pooled mean flux of the 4 classes."""
    taus = [np.load(Path(base) / str(c) / "tau.npy", mmap_mode="r") for c in range(1, N_CLASSES + 1)]
    means = [_mean_flux_scaled(t, 1.0) for t in taus]
    target = float(np.mean(means))
    factors = [brentq(lambda a, t=t: _mean_flux_scaled(t, a) - target, 0.05, 20.0, xtol=1e-10)
               for t in taus]
    return {"target_mean_flux": target, "class_mean_flux": means, "A": factors}


def load_flux_variant(variant: str, class_id: int, rows: Optional[np.ndarray] = None,
                      factors: Optional[List[float]] = None, base: str = FLUX_BASE) -> np.ndarray:
    """Flux of one class for a data variant, optionally restricted to `rows`.

    fiducial    : the stored flux (== roll(exp(-tau), 1), verified 2026-10-18).
    taurescaled : roll(exp(-A_c tau), 1), same pixel convention as the stored flux.
    """
    if variant not in VARIANTS:
        raise ValueError(f"load_flux_variant: unknown variant {variant!r}")
    sel = slice(None) if rows is None else rows
    if variant == "fiducial":
        F = np.asarray(np.load(Path(base) / str(class_id) / "flux.npy", mmap_mode="r")[sel],
                       dtype=np.float64)
    else:
        if factors is None:
            raise ValueError("load_flux_variant: taurescaled needs the A_c factors")
        tau = np.asarray(np.load(Path(base) / str(class_id) / "tau.npy", mmap_mode="r")[sel],
                         dtype=np.float64)
        F = np.roll(np.exp(-factors[class_id - 1] * tau), 1, axis=1)
    if not np.isfinite(F).all():
        raise ValueError(f"load_flux_variant: non-finite flux in class {class_id}")
    return np.clip(F, 0.0, 1.0)                              # shape: (n_rows, n_pix)


def stacked_labels_groups(n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """Class labels 1..4 and skewer-position groups for class-major stacking."""
    y = np.repeat(np.arange(1, N_CLASSES + 1), n_per_class)               # shape: (4n,)
    groups = np.tile(np.arange(n_per_class), N_CLASSES)                   # shape: (4n,)
    return y, groups


def scalar_baselines(flux: np.ndarray) -> np.ndarray:
    """[<F>, <tau_eff>] per sightline. Total equivalent width is affine in <F> (EW =
    N dv (1 - <F>)), so <tau_eff> = <-ln F> is the second, non-redundant scalar."""
    meanF = flux.mean(axis=1)
    taueff = -np.log(np.clip(flux, 1e-300, None)).clip(max=700.0).mean(axis=1)
    return np.column_stack([meanF, taueff])                  # shape: (n, 2)


def pk_features(flux_per_class: List[np.ndarray], dv_kms: float, n_bins: int = 32) -> np.ndarray:
    """Per-sightline log P_F(k), normalised by ONE pooled global mean flux.

    FluxPowerSpectrumTransform divides by the mean of whatever it is given; passing the
    stacked array of all classes gives a single pooled <F>, so no per-class mean (a label
    leak) enters.
    """
    X = np.vstack(flux_per_class)
    return FluxPowerSpectrumTransform(n_bins=n_bins, dv_kms=dv_kms).fit_transform(X)
