"""
Stage 1: Energy Per Level
=========================
Loads the cached per-level wavelet features (features_D1.npy … features_A6.npy),
computes E_k = mean_over_sightlines( sum_j w_{k,j}^2 ) for each class,
and prints a Markdown table suitable for insertion into feature_extraction_plan.md.

Features layout per file: (n_probes, 4 * coeff_len)
  cols 0           .. coeff_len-1  → class 1 (NoFeedback)
  cols coeff_len   .. 2*coeff_len-1 → class 2 (StellarWind)
  cols 2*coeff_len .. 3*coeff_len-1 → class 3 (WindAGN)
  cols 3*coeff_len .. 4*coeff_len-1 → class 4 (WindStrongAGN)
"""

import numpy as np
import os

DATA_DIR = "data/feature_discovery/experiments/wavelet_per_level"
LEVELS   = ["D1", "D2", "D3", "D4", "D5", "D6", "A6"]
CLASSES  = ["NoFeedback", "StellarWind", "WindAGN", "WindStrongAGN"]

def mean_energy(coeffs):
    """Mean energy across sightlines: mean_i( sum_j w_j^2 ) = mean_i( ||w_i||^2 )"""
    return float(np.mean(np.sum(coeffs ** 2, axis=1)))

results = {}   # level → list of 4 energies

for lvl in LEVELS:
    path = os.path.join(DATA_DIR, f"features_{lvl}.npy")
    data = np.load(path)                      # (n_probes, 4 * coeff_len)
    n, total = data.shape
    coeff_len = total // 4
    energies = []
    for cls in range(4):
        c = data[:, cls * coeff_len : (cls + 1) * coeff_len]
        energies.append(mean_energy(c))
    results[lvl] = energies

# ── Print Markdown table ──────────────────────────────────────────────────────
header_cols = ["Level"] + CLASSES
print("| " + " | ".join(header_cols) + " |")
print("|" + "|".join([":---:"] * len(header_cols)) + "|")

for lvl in LEVELS:
    row = [lvl] + [f"{e:.4f}" for e in results[lvl]]
    print("| " + " | ".join(row) + " |")

# ── Also print a quick ratio table (relative to NoFeedback) ──────────────────
print()
print("**Ratio relative to NoFeedback** (E_class / E_NoFeedback)")
print()
print("| " + " | ".join(header_cols) + " |")
print("|" + "|".join([":---:"] * len(header_cols)) + "|")
for lvl in LEVELS:
    base = results[lvl][0]
    row = [lvl] + [f"{e/base:.3f}" if base > 0 else "—" for e in results[lvl]]
    print("| " + " | ".join(row) + " |")
