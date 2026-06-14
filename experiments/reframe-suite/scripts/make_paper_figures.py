"""Render the three headline paper figures from the reframe-suite hardening CSVs.

Scientific purpose: produce publication-quality, reproducible figures for the
feedback-recipe distinguishability paper draft:
  F1   distinguishability lattice (4x4 heatmap, balanced-acc p16 at per_sightline, M=64)
  F2   distinguishability vs stacking depth M (one line per pair, p16-p84 bands)
  Finj injection-recovery sensitivity (balanced acc vs injection amplitude alpha)

All input is read from results/reframe_suite/hardening/*.csv; nothing is hardcoded
beyond the percentile thresholds documented in the paper. Run from repo root:
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/make_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
HARD = REPO / "results" / "reframe_suite" / "hardening"
OUTDIR = REPO / "papers" / "shared" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

PAIR_CSV = HARD / "pair_binary_summary.csv"
INJ_CSV = HARD / "injection_recovery_summary.csv"

REGIME = "per_sightline"
LATTICE_M = 64
QUAL_BAR = 0.75  # pair separability floor (visual reference only)
DET_BAR = 0.90   # detection bar (visual reference only)

# Colorblind-safe (Okabe-Ito) palette for the per-pair lines in F2.
PAIR_COLORS: Dict[str, str] = {
    "C1-C2": "#0072B2",  # blue
    "C1-C3": "#56B4E9",  # sky blue
    "C1-C4": "#009E73",  # bluish green
    "C2-C3": "#D55E00",  # vermillion (headline weak-signal pair)
    "C2-C4": "#E69F00",  # orange
    "C3-C4": "#CC79A7",  # reddish purple
    "C4-rest": "#000000",  # black (detection)
}
PAIR_LABELS: Dict[str, str] = {
    "C1-C2": "C1–C2 (NoFB / StellarWind)",
    "C1-C3": "C1–C3 (NoFB / WindAGN)",
    "C1-C4": "C1–C4 (NoFB / WindStrongAGN)",
    "C2-C3": "C2–C3 (StellarWind / WindAGN)",
    "C2-C4": "C2–C4 (StellarWind / WindStrongAGN)",
    "C3-C4": "C3–C4 (WindAGN / WindStrongAGN)",
    "C4-rest": "C4 vs rest (detection)",
}

CLASS_ORDER = ["C1", "C2", "C3", "C4"]
CLASS_NAMES = {
    "C1": "C1\nNoFeedback",
    "C2": "C2\nStellarWind",
    "C3": "C3\nWindAGN",
    "C4": "C4\nWindStrongAGN",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
        }
    )


def _save(fig: plt.Figure, stem: str) -> Tuple[Path, Path]:
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# F1 — distinguishability lattice
# ---------------------------------------------------------------------------
def figure_lattice(df: pd.DataFrame) -> Tuple[Path, Path]:
    sub = df[(df["regime"] == REGIME) & (df["M"] == LATTICE_M)]
    # p16 per off-diagonal pair, keyed by frozenset of the two class ids.
    p16: Dict[frozenset, float] = {}
    for _, row in sub.iterrows():
        pair = row["pair"]
        if pair == "C4-rest":
            continue
        a, b = pair.split("-")
        p16[frozenset((a, b))] = float(row["p16"])

    n = len(CLASS_ORDER)
    grid = np.full((n, n), np.nan)
    for i, ci in enumerate(CLASS_ORDER):
        for j, cj in enumerate(CLASS_ORDER):
            if i == j:
                continue
            grid[i, j] = p16[frozenset((ci, cj))]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cmap = plt.get_cmap("viridis").copy()
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")
    cmap.set_bad(color="#cccccc")

    # Grey-out the diagonal explicitly.
    for d in range(n):
        ax.add_patch(
            plt.Rectangle((d - 0.5, d - 0.5), 1, 1, facecolor="#cccccc", edgecolor="white")
        )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = grid[i, j]
            # white text on dark cells, black on light cells
            txt_color = "white" if val >= 0.72 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER], fontsize=9)
    ax.set_yticklabels([CLASS_NAMES[c] for c in CLASS_ORDER], fontsize=9)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(
        f"Pairwise distinguishability lattice\n(per-sightline, $M={LATTICE_M}$, balanced-acc p16)"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Balanced accuracy (p16)")
    cbar.ax.axhline(QUAL_BAR, color="white", linewidth=1.0, linestyle="--")
    cbar.ax.text(1.6, QUAL_BAR, f"{QUAL_BAR:.2f}", va="center", fontsize=8, color="black")

    fig.tight_layout()
    return _save(fig, "fig_lattice_M64")


# ---------------------------------------------------------------------------
# F2 — distinguishability vs stacking depth M
# ---------------------------------------------------------------------------
def figure_acc_vs_M(df: pd.DataFrame) -> Tuple[Path, Path]:
    sub = df[df["regime"] == REGIME]
    headline_band_pairs = {"C2-C3", "C4-rest"}

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for pair in PAIR_COLORS:
        d = sub[(sub["pair"] == pair) & (sub["exploratory_flag"] == 0)].sort_values("M")
        d = d[d["M"] <= 256]
        if d.empty:
            continue
        color = PAIR_COLORS[pair]
        lw = 2.2 if pair in headline_band_pairs else 1.4
        ax.plot(
            d["M"],
            d["median"],
            marker="o",
            markersize=4,
            color=color,
            linewidth=lw,
            label=PAIR_LABELS[pair],
            zorder=3 if pair in headline_band_pairs else 2,
        )
        if pair in headline_band_pairs:
            ax.fill_between(
                d["M"], d["p16"], d["p84"], color=color, alpha=0.18, linewidth=0, zorder=1
            )

    # Exploratory high-M C2-C3 points (hollow markers).
    expl = sub[(sub["pair"] == "C2-C3") & (sub["exploratory_flag"] == 1)].sort_values("M")
    if not expl.empty:
        ax.plot(
            expl["M"],
            expl["median"],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=PAIR_COLORS["C2-C3"],
            markeredgewidth=1.6,
            label="C2–C3 (exploratory, few groups)",
            zorder=4,
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.0)
    ax.text(1.05, 0.508, "chance (0.50)", fontsize=8, color="grey")
    ax.axvline(LATTICE_M, color="grey", linestyle=":", linewidth=1.0)
    ax.text(
        LATTICE_M * 1.04,
        0.33,
        f"$M={LATTICE_M}$\n(lattice read)",
        fontsize=8,
        color="grey",
        rotation=0,
        va="bottom",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels([1, 4, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xlabel("Stacking depth $M$ (groups stacked per sample)")
    ax.set_ylabel("Balanced accuracy (median; bands = p16–p84)")
    ax.set_ylim(0.28, 1.02)
    ax.set_title("Distinguishability vs stacking depth")
    ax.legend(loc="lower right", framealpha=0.95, ncol=1)
    ax.grid(True, which="major", alpha=0.25)

    fig.tight_layout()
    return _save(fig, "fig_acc_vs_M")


# ---------------------------------------------------------------------------
# F-inj — injection-recovery sensitivity
# ---------------------------------------------------------------------------
def figure_injection(df: pd.DataFrame) -> Tuple[Path, Path, float]:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    delta_min = float("nan")

    for M, color, alpha_band in ((64, "#0072B2", True), (256, "#999999", False)):
        d = df[df["M"] == M].sort_values("alpha")
        lw = 2.2 if M == 64 else 1.3
        label = f"$M={M}$" + ("" if M == 64 else " (variance-dominated)")
        ax.plot(
            d["alpha"],
            d["median"],
            marker="o",
            markersize=5,
            color=color,
            linewidth=lw,
            label=label,
            zorder=3 if M == 64 else 2,
        )
        if alpha_band:
            ax.fill_between(
                d["alpha"], d["p16"], d["p84"], color=color, alpha=0.18, linewidth=0, zorder=1
            )
            # delta_min: first alpha where M=64 p16 >= 0.60
            clears = d[d["p16"] >= 0.60].sort_values("alpha")
            if not clears.empty:
                delta_min = float(clears.iloc[0]["alpha"])

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.0)
    ax.text(0.012, 0.508, "chance (0.50)", fontsize=8, color="grey")
    ax.axhline(0.60, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.text(0.012, 0.608, r"$\delta_{\min}$ threshold (0.60)", fontsize=8, color="#D55E00")

    if not np.isnan(delta_min):
        ax.axvline(delta_min, color="#0072B2", linestyle=":", linewidth=1.2)
        ax.annotate(
            rf"$\delta_{{\min}}(M{{=}}64)={delta_min:g}$" + f"\n({delta_min*100:g}% of $C4\\!-\\!C2$ template)",
            xy=(delta_min, 0.60),
            xytext=(delta_min * 2.6, 0.78),
            fontsize=9,
            color="#0072B2",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.0),
        )

    # mark the alpha=0 calibration control
    ctrl = df[(df["M"] == 64) & (df["alpha"] == 0.0)]
    if not ctrl.empty:
        ax.annotate(
            f"$\\alpha=0$ control\n(median {float(ctrl.iloc[0]['median']):.2f}, at chance)",
            xy=(0.0095, float(ctrl.iloc[0]["median"])),
            xytext=(0.0105, 0.40),
            fontsize=8,
            color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel(r"Injection amplitude $\alpha$ (units of $\Delta P_{C4-C2}$)")
    ax.set_ylabel("Balanced accuracy (median; band = p16–p84)")
    ax.set_ylim(0.33, 1.03)
    ax.set_xticks([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
    ax.set_title("Injection-recovery sensitivity")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, which="major", alpha=0.25)

    fig.tight_layout()
    pdf, png = _save(fig, "fig_injection_sensitivity")
    return pdf, png, delta_min


def main() -> None:
    _set_style()
    pair_df = pd.read_csv(PAIR_CSV)
    inj_df = pd.read_csv(INJ_CSV)

    f1_pdf, f1_png = figure_lattice(pair_df)
    f2_pdf, f2_png = figure_acc_vs_M(pair_df)
    finj_pdf, finj_png, delta_min = figure_injection(inj_df)

    # --- sanity-check echoes ---
    ps = pair_df[pair_df["regime"] == REGIME]
    c2c3 = ps[ps["pair"] == "C2-C3"].set_index("M")["median"]
    c4rest_p16 = float(
        ps[(ps["pair"] == "C4-rest") & (ps["M"] == 64)]["p16"].iloc[0]
    )
    print("=== sanity checks (per_sightline) ===")
    print(f"C2-C3 median  M=1   : {c2c3.loc[1]:.4f}  (expect ~0.36)")
    print(f"C2-C3 median  M=256 : {c2c3.loc[256]:.4f}  (expect ~0.69)")
    print(f"C4-rest p16   M=64  : {c4rest_p16:.4f}  (expect ~0.954)")
    print(f"delta_min(M=64)     : {delta_min:g}  (expect 0.02)")
    print("=== files written ===")
    for p in (f1_pdf, f1_png, f2_pdf, f2_png, finj_pdf, finj_png):
        print(p)


if __name__ == "__main__":
    main()
