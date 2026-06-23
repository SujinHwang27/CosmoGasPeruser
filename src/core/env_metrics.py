"""
Per-sightline absorption-content metrics for the cluster-environment-gradient probe.

Scientific purpose
-------------------
Provides the two NEW per-sightline scalar metrics the SCOPING contract requires
(M1 `mean_1mF`, M3 `saturated_frac`) that are not already exposed by
`scripts/eda_sherwood.py`. These quantify total absorption strength and the
fraction of (near-)saturated pixels per line-of-sight, and feed the gradient test
that asks whether content-clusters sort sightlines by absorption strength.

These are deliberately one-liners kept in `src/core/` (never in the script, per
CLAUDE.md). They operate on raw flux F (not absorption) and compute absorption
A = 1 - F internally so callers pass the loader's flux directly.
"""

import numpy as np


def mean_1mF(flux: np.ndarray) -> np.ndarray:
    """
    M1 — per-sightline mean absorption = mean(1 - flux) over all pixels.

    Args:
        flux: Raw flux, shape (n_sightlines, n_pixels).  # shape: (N, 2048)

    Returns:
        Per-sightline mean(1 - F), shape (n_sightlines,).  # shape: (N,)
    """
    return np.mean(1.0 - flux, axis=1)


def saturated_frac(flux: np.ndarray, tau: float = 0.8) -> np.ndarray:
    """
    M3 — per-sightline fraction of pixels with absorption (1 - F) > tau.

    Args:
        flux: Raw flux, shape (n_sightlines, n_pixels).  # shape: (N, 2048)
        tau: Absorption threshold (default 0.8, i.e. flux < 0.2). SCOPING
            pre-commits tau=0.8 for the headline and tau=0.9 for a sensitivity check.

    Returns:
        Per-sightline saturated-pixel fraction, shape (n_sightlines,).  # shape: (N,)
    """
    return np.mean((1.0 - flux) > tau, axis=1)
