"""
eigen-voigt-embedding track: figures.

Scientific purpose
------------------
Renders the three EV-3 figures from saved outputs (no recomputation):
  1. eigenmap - PCA balanced-accuracy delta vs embedding dimension for the primary
     classifier (RF, masked graph), per input and per data variant, with block-bootstrap
     95% bands, for the three confound modes (none / spline-residualised / scalar-augmented);
  2. per-sightline 4-class BA of the main representations, fiducial vs tau-rescaled;
  3. Voigt Doppler-parameter distributions per feedback class (log N >= 12 lines),
     fiducial vs tau-rescaled — the line-width channel the catalogs single out.
Colours follow the validated categorical order (dataviz reference palette); identity is
always carried by a legend and direct labels as well as colour.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLASS_NAMES = {1: "NoFeedback", 2: "StellarWind", 3: "WindAGN", 4: "WindStrongAGN"}
RESID_NAMES = {0: "no control", 1: "spline-residualised on ⟨F⟩, ⟨τ⟩", 2: "+ scalars (increment)"}
INPUT_NAMES = {"raw": "raw flux", "voigt": "Voigt line statistics", "pk": "P_F(k)"}
VARIANT_NAMES = {"fiducial": "fiducial", "taurescaled": "τ-rescaled (equal class ⟨F⟩)"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _label_ends(ax: plt.Axes, ends: List[List[float]], min_gap: float = 0.011) -> None:
    """Direct end-of-line value labels, nudged apart vertically so they never overlap."""
    order = sorted(range(len(ends)), key=lambda i: ends[i][0])
    ys = [ends[i][0] for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for i, y in zip(order, ys):
        ax.annotate(f"{ends[i][0]:+.3f}", (ends[i][1], ends[i][0]), xytext=(ends[i][1] * 1.08, y),
                    textcoords="data", fontsize=7, color=INK2, va="center")


def fig_le_minus_pca(res_dirs: Dict[str, Path], out: Path) -> Optional[Path]:
    """Small multiples: rows = variant, columns = input; series = confound mode."""
    variants = [v for v in ("fiducial", "taurescaled") if (res_dirs.get(v) and (res_dirs[v] / "le_minus_pca.csv").exists())]
    if not variants:
        return None
    reps = ("raw", "voigt", "pk")
    fig, axes = plt.subplots(len(variants), 3, figsize=(11, 3.3 * len(variants)), squeeze=False,
                             sharey=True, facecolor=SURFACE)
    for i, v in enumerate(variants):
        df = pd.read_csv(res_dirs[v] / "le_minus_pca.csv")
        df = df[(df.le_mode == "le_masked") & (df.clf == "rf")]
        for j, rep in enumerate(reps):
            ax = axes[i, j]
            _style(ax)
            ax.axhline(0, color=INK2, linewidth=0.8)
            sub_all = df[df.rep == rep]
            ends = []
            for k, resid in enumerate((0, 1, 2)):
                s = sub_all[sub_all.resid == resid].sort_values("d")
                if s.empty:
                    continue
                ax.fill_between(s.d, s.ci95_lo, s.ci95_hi, color=SERIES[k], alpha=0.15, linewidth=0)
                ax.plot(s.d, s.delta_ba4, color=SERIES[k], linewidth=2, marker="o", markersize=4,
                        label=RESID_NAMES[resid])
                ends.append([float(s.delta_ba4.iloc[-1]), float(s.d.iloc[-1])])
            _label_ends(ax, ends)
            ax.axvline(8, color=GRID, linewidth=1, linestyle=":")
            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32])
            ax.set_xticklabels(["2", "4", "8", "16", "32"])
            ax.set_title(f"{INPUT_NAMES[rep]} — {VARIANT_NAMES[v]}", fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("ΔBA (eigenmap − PCA)", fontsize=8, color=INK2)
            if i == len(variants) - 1:
                ax.set_xlabel("embedding dimension d (primary cell d = 8, dotted)", fontsize=8, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle("Masked Laplacian eigenmap vs PCA at matched dimension (random forest, 95% block-bootstrap bands)",
                 fontsize=10, color=INK, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=8)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_representation_ba(res_dirs: Dict[str, Path], out: Path) -> Optional[Path]:
    """Horizontal grouped bars: seed-averaged BA (RF, no control) per representation."""
    rows: List[Dict[str, object]] = []
    picks = [("scalar", "meanF_meantau", 2, "⟨F⟩, ⟨τ⟩ scalars"), ("raw", "pca", 8, "raw flux PCA(8)"),
             ("raw", "le_masked", 8, "raw flux eigenmap(8)"), ("raw", "direct", 2048, "raw flux, all 2048 px"),
             ("pk", "pca", 8, "P_F(k) PCA(8)"), ("pk", "le_masked", 8, "P_F(k) eigenmap(8)"),
             ("pk", "direct", None, "P_F(k), all bins"), ("voigt", "pca", 8, "Voigt PCA(8)"),
             ("voigt", "le_masked", 8, "Voigt eigenmap(8)"), ("voigt", "direct", None, "Voigt, all features")]
    for v in ("fiducial", "taurescaled"):
        p = res_dirs.get(v)
        if not p or not (p / "summary_by_config.csv").exists():
            continue
        s = pd.read_csv(p / "summary_by_config.csv")
        s = s[(s.clf == "rf") & (s.resid == 0)]
        for rep, method, d, label in picks:
            m = (s.rep == rep) & (s.method == method)
            if d is not None:
                m &= s.d == d
            if m.any():
                r = s[m].iloc[0]
                rows.append({"variant": v, "label": label, "med": r.ba4_median, "lo": r.ba4_p16, "hi": r.ba4_p84})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    labels = [p[3] for p in picks if p[3] in set(df.label)]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(labels) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(labels))
    variants = list(dict.fromkeys(df.variant))
    h = 0.8 / len(variants)
    for k, v in enumerate(variants):
        sub = df[df.variant == v].set_index("label").reindex(labels)
        yy = y + (k - (len(variants) - 1) / 2) * h
        ax.barh(yy, sub.med, height=h * 0.9, color=SERIES[k], label=VARIANT_NAMES[v], edgecolor=SURFACE, linewidth=1)
        ax.errorbar(sub.med, yy, xerr=[sub.med - sub.lo, sub.hi - sub.med], fmt="none", ecolor=INK2,
                    elinewidth=0.8, capsize=0)
        for yi, val, hi in zip(yy, sub.med, sub.hi):
            if np.isfinite(val):
                ax.text(hi + 0.004, yi, f"{val:.3f}", va="center", fontsize=7, color=INK2)
    ax.axvline(0.25, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(0.252, len(labels) - 0.4, "chance", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.2, max(0.55, float(df.hi.max()) + 0.03))
    ax.set_xlabel("4-class balanced accuracy per sightline (RF; median, 16–84% over splits)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_b_distributions(art_dir: Path, mode: str, out: Path) -> Optional[Path]:
    """Cumulative distribution of b (log N >= 12) per class, fiducial vs tau-rescaled."""
    variants = [v for v in ("fiducial", "taurescaled") if (art_dir / f"catalog_{v}_{mode}" / "class1.npz").exists()]
    if not variants:
        return None
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 3.4), squeeze=False,
                             sharey=True, facecolor=SURFACE)
    grid_b = np.linspace(0, 150, 301)
    for i, v in enumerate(variants):
        ax = axes[0, i]
        _style(ax)
        for c in range(1, 5):
            cat = np.load(art_dir / f"catalog_{v}_{mode}" / f"class{c}.npz")
            b = np.sort(cat["b_kms"][cat["logN"] >= 12.0])
            cdf = np.searchsorted(b, grid_b, side="right") / b.size
            ax.plot(grid_b, cdf, color=SERIES[c - 1], linewidth=2, label=f"{c} {CLASS_NAMES[c]}")
            med = float(np.median(b))
            ax.annotate(f"class {c}: {med:.1f}", (85, 0.62 - 0.07 * (c - 1)), fontsize=7, color=INK2)
        ax.axhline(0.5, color=GRID, linewidth=1, linestyle=":")
        ax.text(85, 0.69, "median b (km/s)", fontsize=7, color=INK2)
        ax.set_title(VARIANT_NAMES[v], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Doppler parameter b (km/s), lines with log N_HI ≥ 12", fontsize=8, color=INK2)
        if i == 0:
            ax.set_ylabel("cumulative fraction of lines", fontsize=8, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.suptitle("Voigt line widths by feedback class", fontsize=10,
                 color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out
