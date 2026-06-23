"""
cluster-environment-gradient probe — measurement logic.

Scientific purpose
------------------
Tests (by measurement, NOT by inheriting v1's interpretive "Void/Forest/Dense"
labels) whether content-clustered Sherwood sightlines order into a monotone,
distributionally-separated, representation-stable absorption-strength gradient.
See experiments/cluster-environment-gradient/SCOPING.md for the binding
pre-committed PASS / NULL / KILL bars. Logic lives here (not in the script) per
CLAUDE.md.

Two CONTENT clustering arms (K-Means, K=5, seed=42):
  - raw-content-summary: cluster the 5 z-scored Tier-1 absorption scalars.
  - wavelet: cluster the per-level z-scored db8/l6 wavelet features (2048-dim).
Per-cluster distributions of 6 absorption metrics (M1..M6) → Spearman monotonicity
(G-a), KS separation (G-b), cross-representation stability (X).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.core.cluster import contingency_matrix, fit_kmeans
from src.core.data import SignalClusteringData
from src.core.env_metrics import mean_1mF, saturated_frac

_K = 5
_SEED = 42
_METRICS = ["mean_1mF", "depth_mean", "saturated_frac", "total_ew", "line_count", "A6_energy"]
# M1 is the ordering axis; the gradient test checks M2..M6 against the M1 order.
_CROSS_METRICS = ["depth_mean", "saturated_frac", "total_ew", "line_count", "A6_energy"]


def _per_sightline_metrics(flux: np.ndarray, wave: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute M1..M6 per sightline on the canonical loader row order.

    Args:
        flux: shape (N, 2048) raw flux in canonical class-concatenated order.
        wave: shape (2048,) wavelength axis (Angstrom).

    Returns:
        dict metric_name -> (N,) float array.
    """
    import pywt
    from scripts.eda_sherwood import absorption_depths, detect_local_minima

    n = flux.shape[0]
    # Vectorized metrics.
    m1 = mean_1mF(flux)                                   # M1 mean absorption
    m3 = saturated_frac(flux, tau=0.8)                   # M3 saturated fraction
    m4 = np.trapezoid(1.0 - flux, wave, axis=1)          # M4 total EW (Angstrom)
    # M6 A6-band energy on absorption A=1-F (fresh DWT; avoids stored z-score).
    coeffs = pywt.wavedec(1.0 - flux, "db8", level=6, mode="periodization", axis=1)
    cA6 = coeffs[0]                                       # shape: (N, 32)
    m6 = np.sum(cA6.astype(np.float64) ** 2, axis=1)     # M6 A6 energy
    # Per-sightline loop for the minima-dependent metrics (M2 depth, M5 count).
    m2 = np.zeros(n, dtype=np.float64)
    m5 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mi = detect_local_minima(flux[i])
        m5[i] = float(len(mi))
        if len(mi):
            m2[i] = float(np.mean(absorption_depths(flux[i], mi)))
        # else leave 0.0 (a sightline with no detected line has zero mean depth)
    out = {
        "mean_1mF": m1, "depth_mean": m2, "saturated_frac": m3,
        "total_ew": m4, "line_count": m5, "A6_energy": m6,
    }
    for k, v in out.items():
        assert v.shape == (n,) and np.all(np.isfinite(v)), f"bad metric {k}"
    return out


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _wavelet_content_features(flux: np.ndarray) -> np.ndarray:
    """Faithful v1 content representation: standardized D3-D6+A6 of absorption.

    v1 dropped the noise-dominated D1/D2 bands (signal_clustering_analysis.md
    line 220: "D1 is noise-dominated, 99.5% sparse ... Retained feature space:
    512-dim D3-D6 + A6"). We reproduce that: db8/L6 DWT of A=1-F, keep cA6,cD6,
    cD5,cD4,cD3 (32+32+64+128+256 = 512-dim), per-dim z-score for K-Means
    (StandardScaler-space, the project clustering convention).
    """
    import pywt
    coeffs = pywt.wavedec(1.0 - flux, "db8", level=6, mode="periodization", axis=1)
    # coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]; keep [0:5] = A6,D6,D5,D4,D3.
    feat = np.hstack([np.asarray(c, dtype=np.float64) for c in coeffs[:5]])
    assert feat.shape[1] == 512, feat.shape
    return _zscore(feat)


def _cluster_summary(
    labels: np.ndarray, metrics: Dict[str, np.ndarray]
) -> Dict[int, Dict[str, object]]:
    """Per-cluster size + median/IQR of each metric."""
    summ: Dict[int, Dict[str, object]] = {}
    for c in range(_K):
        mask = labels == c
        row: Dict[str, object] = {"size": int(mask.sum())}
        for name, v in metrics.items():
            vc = v[mask]
            q25, q50, q75 = np.quantile(vc, [0.25, 0.5, 0.75]) if vc.size else (0, 0, 0)
            row[name] = {"median": float(q50), "iqr": float(q75 - q25)}
        summ[c] = row
    return summ


def _gradient_test(
    labels: np.ndarray, metrics: Dict[str, np.ndarray], summ: Dict[int, Dict]
) -> Dict[str, object]:
    """G-a Spearman monotonicity + G-b KS separation, ordered by median(M1)."""
    from scipy.stats import ks_2samp, spearmanr

    order = sorted(range(_K), key=lambda c: summ[c]["mean_1mF"]["median"])  # asc M1
    ranks = list(range(_K))  # 0..4 in M1-ascending order
    # G-a: Spearman between M1-rank and cluster-median(Mj) for j in cross-metrics.
    spearman: Dict[str, float] = {}
    for mj in _CROSS_METRICS:
        med_in_order = [summ[c][mj]["median"] for c in order]
        rho = float(spearmanr(ranks, med_in_order).statistic)
        spearman[mj] = round(rho, 4)
    n_pass_rho = sum(abs(r) >= 0.9 for r in spearman.values())
    # G-b: KS of M1 between adjacent clusters in the M1 order.
    m1 = metrics["mean_1mF"]
    ks_adj: List[float] = []
    for a, b in zip(order[:-1], order[1:]):
        ks = float(ks_2samp(m1[labels == a], m1[labels == b]).statistic)
        ks_adj.append(round(ks, 4))
    n_pass_ks = sum(k >= 0.2 for k in ks_adj)
    return {
        "m1_order": order,
        "spearman_vs_m1rank": spearman,
        "n_metrics_rho_ge_0.9": n_pass_rho,
        "ks_adjacent_m1": ks_adj,
        "n_adjacencies_ks_ge_0.2": n_pass_ks,
        "G_a_pass": n_pass_rho >= 4,          # >=4 of 5
        "G_b_pass": n_pass_ks >= (_K - 2),    # >=3 of 4
    }


def _cross_rep_stability(
    labels_a: np.ndarray, labels_b: np.ndarray,
    summ_a: Dict[int, Dict], summ_b: Dict[int, Dict],
) -> Dict[str, object]:
    """(X): match arms by contingency overlap, rank-correlate matched median(M1)."""
    from scipy.stats import spearmanr

    cont = contingency_matrix(labels_a, labels_b, _K, _K)  # (5,5) counts
    matched_b = cont.argmax(axis=1)                        # for each a-cluster, best b
    matched_mass = int(cont[np.arange(_K), matched_b].sum())
    total = int(cont.sum())
    med_a = [summ_a[c]["mean_1mF"]["median"] for c in range(_K)]
    med_b = [summ_b[int(matched_b[c])]["mean_1mF"]["median"] for c in range(_K)]
    rho = float(spearmanr(med_a, med_b).statistic)
    frac = matched_mass / total
    return {
        "matched_b_for_each_a": matched_b.tolist(),
        "matched_mass_fraction": round(frac, 4),
        "matched_median_m1_spearman": round(rho, 4),
        "X_pass": (rho >= 0.8) and (frac >= 0.60),
    }


def _verdict(raw_g: Dict, wav_g: Dict, cross: Dict) -> Tuple[str, str]:
    """Apply SCOPING §3 PASS / NULL bars to the wavelet (real-test) arm + X."""
    g_a = wav_g["G_a_pass"]
    g_b = wav_g["G_b_pass"]
    x = cross["X_pass"]
    if g_a and g_b and x:
        return "PASS", (
            "Content-clusters (wavelet arm) sort sightlines into a monotone, "
            "distributionally-separated, representation-stable absorption-strength "
            "gradient. CEILING: this supports 'clusters sort by absorption "
            "strength' — it does NOT establish 'dense gas cloud vs cosmic void' "
            "(no ground-truth halo/density catalog). The v1 environment LABELS "
            "remain interpretive."
        )
    fails = []
    if not g_a:
        fails.append("ordering not monotone (G-a)")
    if not g_b:
        fails.append("clusters not distributionally separated on M1 (G-b)")
    if not x:
        fails.append("ordering does not survive raw-vs-wavelet (X)")
    return "NULL", (
        "Content-clusters do NOT order into a monotone/separated/stable absorption "
        "gradient: " + "; ".join(fails) + ". v1's Void/Forest/Dense labels are not "
        "supported by measured per-cluster absorption at K=5."
    )


def run_env_gradient_probe(out_dir: Path) -> Dict[str, object]:
    """Run the full probe; write figure + CSV + provenance; return the result dict."""
    out_dir = Path(out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    loader = SignalClusteringData()
    flux, y = loader.load_flux()                       # (65536, 2048), canonical order
    wave = np.load(Path(loader.flux_path) / "1" / "wave.npy").astype(np.float64)
    n = flux.shape[0]

    metrics = _per_sightline_metrics(flux, wave)

    # --- arm 1: raw-content-summary (5 z-scored scalars) ---
    summ_feat = _zscore(np.column_stack([
        metrics["total_ew"], metrics["depth_mean"], metrics["mean_1mF"],
        metrics["saturated_frac"], metrics["line_count"],
    ]))                                                # (65536, 5)
    labels_raw, _ = fit_kmeans(summ_feat, _K, seed=_SEED)

    # --- arm 2: wavelet content (faithful 512-dim D3-D6+A6, standardized) ---
    wav = _wavelet_content_features(flux)             # (65536, 512), z-scored
    assert wav.shape[0] == n, f"wavelet rows {wav.shape[0]} != flux rows {n}"
    labels_wav, _ = fit_kmeans(wav, _K, seed=_SEED)

    summ_raw = _cluster_summary(labels_raw, metrics)
    summ_wav = _cluster_summary(labels_wav, metrics)
    g_raw = _gradient_test(labels_raw, metrics, summ_raw)
    g_wav = _gradient_test(labels_wav, metrics, summ_wav)
    cross = _cross_rep_stability(labels_raw, labels_wav, summ_raw, summ_wav)
    verdict, reading = _verdict(g_raw, g_wav, cross)

    _write_figure(out_dir / "figs" / "cluster_env_gradient.png",
                  metrics, labels_raw, labels_wav, summ_raw, summ_wav,
                  g_raw, g_wav, verdict)
    _write_summary_csv(out_dir / "per_cluster_summary.csv", summ_raw, summ_wav)

    result = {
        "verdict": verdict,
        "reading": reading,
        "n_sightlines": n,
        "raw_summary_arm": {"gradient": g_raw},
        "wavelet_arm": {"gradient": g_wav},
        "cross_representation": cross,
        "per_cluster": {"raw_summary": summ_raw, "wavelet": summ_wav},
        "ceiling_claim": (
            "A measured absorption gradient supports 'clusters sort by absorption "
            "strength'; it does NOT establish 'dense gas vs void' (no halo/density "
            "catalog). Labels remain interpretive even on PASS."
        ),
    }
    with open(out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def _write_summary_csv(path: Path, summ_raw: Dict, summ_wav: Dict) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "cluster", "size"] + [f"{m}_median" for m in _METRICS]
                   + [f"{m}_iqr" for m in _METRICS])
        for arm, summ in (("raw_summary", summ_raw), ("wavelet", summ_wav)):
            for c in range(_K):
                row = [arm, c, summ[c]["size"]]
                row += [repr(summ[c][m]["median"]) for m in _METRICS]
                row += [repr(summ[c][m]["iqr"]) for m in _METRICS]
                w.writerow(row)


def _write_figure(path, metrics, labels_raw, labels_wav, summ_raw, summ_wav,
                  g_raw, g_wav, verdict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(_METRICS), figsize=(4 * len(_METRICS), 9))
    for r, (arm, labels, summ, g) in enumerate([
        ("raw-summary", labels_raw, summ_raw, g_raw),
        ("wavelet", labels_wav, summ_wav, g_wav),
    ]):
        order = g["m1_order"]
        for ci, m in enumerate(_METRICS):
            ax = axes[r, ci]
            data = [metrics[m][labels == c] for c in order]
            ax.boxplot(data, showfliers=False, labels=[str(c) for c in order])
            ax.set_title(f"{arm}: {m}", fontsize=9)
            if ci == 0:
                ax.set_ylabel("clusters ordered by median(M1=mean 1-F)", fontsize=8)
    fig.suptitle(
        f"cluster-environment-gradient probe — VERDICT: {verdict}  |  "
        f"wavelet-arm: G-a rho>=0.9 on {g_wav['n_metrics_rho_ge_0.9']}/5, "
        f"G-b KS>=0.2 on {g_wav['n_adjacencies_ks_ge_0.2']}/4",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
