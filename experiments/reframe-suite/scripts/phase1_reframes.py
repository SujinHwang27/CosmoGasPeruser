"""
reframe-suite Phase 1 analysis script.

Runs Reframes 1, 5, 7 on already-on-disk artifacts:

  - Reframe 1: Binary {C4 vs C1 union C2 union C3} re-aggregation
               of the Stage 1 4-class confusion matrices.
  - Reframe 5: 6-pair distinguishability lattice from the same
               confusions.
  - Reframe 7: K=5 cluster physical-axis interpretation (one-way
               ANOVA, eta^2 effect size) on signal-clustering-v2
               labels against mean_flux / integrated_tau / line_count.

Invocation:

    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/phase1_reframes.py

Outputs (CSV) to results/reframe_suite/phase1/.

Inputs read-only:
  - results/pk_feedback_classifier/stage1/cv_partial.csv
  - data/feature_discovery/labels_wavelet_k5.npy
  - data/feature_discovery/labels_raw_k5.npy
  - data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,tau}.npy

Confusion-matrix convention (verified against run_stage1.py line 298-303):
  cm rows are row-normalized recall (rows sum to 1), index 0..3
  maps to true labels 1..4 in the project's class numbering
  (1=NoFB, 2=StellarWind, 3=WindAGN, 4=WindStrongAGN).

Sightline-to-class layout for Reframe 7:
  The K=5 cluster labels of shape (16384,) are per-POSITION
  (not per-(class, position)). Each of the 16,384 entries indexes
  a spatial position that has one realization in EACH of the four
  classes. We therefore compute the physical axes per class at each
  position and report both per-class eta^2 and a class-averaged
  proxy. See src/core/probe.py / scripts/run_probe.py for the
  per-position probe definition that produced these labels.

Deterministic; no RNG required.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import f_oneway


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_CSV = REPO_ROOT / "results" / "pk_feedback_classifier" / "stage1" / "cv_partial.csv"
FEATURE_DISCOVERY = REPO_ROOT / "data" / "feature_discovery"
PREPROCESSED = REPO_ROOT / "data" / "preprocessed" / "Sherwood_z0.3_inf"
OUT_DIR = REPO_ROOT / "results" / "reframe_suite" / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reframe 1 + 5 — shared confusion-matrix parsing
# ---------------------------------------------------------------------------

def parse_confusion(row_str: str) -> np.ndarray:
    """Parse the comma-separated 16-float confusion_flat back to (4, 4)."""
    vals = np.array([float(v) for v in row_str.split(",")], dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"confusion_flat has {vals.size} values, expected 16")
    return vals.reshape(4, 4)


def binary_c4_vs_rest_balacc(cm_norm: np.ndarray) -> float:
    """
    Compute binary {C4 vs C1 union C2 union C3} balanced accuracy from a
    row-normalized 4x4 confusion (rows = true class recall vector).

    Sensitivity (C4 recall) = cm[3, 3]
    Specificity (rest recall) = mean over rows 0..2 of (1 - cm[i, 3])

    Each row already sums to 1, so for the merged-negative aggregation
    of {C1, C2, C3} under equal class weights, the binary balanced
    accuracy is the unweighted average of the per-class "correct
    negative" rates plus the C4 recall.
    """
    sens = cm_norm[3, 3]
    spec_per_class = 1.0 - cm_norm[0:3, 3]
    spec = float(np.mean(spec_per_class))
    return 0.5 * (float(sens) + spec)


def pair_balacc_from_cm(cm_norm: np.ndarray, a: int, b: int) -> float:
    """
    Pairwise binary balanced accuracy for true-class pair (a, b),
    a != b, indices 0..3.

    Each row i is the row-normalized recall vector across all 4
    predicted classes. We restrict to the 2x2 sub-block on rows {a, b}
    and columns {a, b}, then renormalize each row to sum to 1
    (i.e. condition on the predicted label being in {a, b}). Balanced
    accuracy = 0.5 * (TP_a / (TP_a + FN_a) + TN_a / (TN_a + FP_a)) on
    that renormalized 2x2.
    """
    sub = np.array(
        [
            [cm_norm[a, a], cm_norm[a, b]],
            [cm_norm[b, a], cm_norm[b, b]],
        ],
        dtype=np.float64,
    )
    row_sums = sub.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    sub_n = sub / row_sums
    sens_a = sub_n[0, 0]  # recall_a within {a, b}
    sens_b = sub_n[1, 1]  # recall_b within {a, b} == specificity for a
    return 0.5 * (float(sens_a) + float(sens_b))


# Six ordered pairs (0-indexed in confusion, 1-indexed in label) ------
PAIRS: List[Tuple[int, int, str]] = [
    (0, 1, "C1-C2"),
    (0, 2, "C1-C3"),
    (0, 3, "C1-C4"),
    (1, 2, "C2-C3"),
    (1, 3, "C2-C4"),
    (2, 3, "C3-C4"),
]


def reframe1_and_5() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(STAGE1_CSV)
    print(f"[Reframes 1+5] Loaded cv_partial.csv: {len(df)} rows")

    cms = np.stack([parse_confusion(s) for s in df["confusion_flat"].values])

    # Reframe 1: binary {C4 vs rest} per row
    bin_acc = np.array([binary_c4_vs_rest_balacc(cm) for cm in cms])
    df = df.copy()
    df["binary_c4_vs_rest"] = bin_acc

    # Aggregate per (regime, M)
    r1_rows = []
    for (regime, M), g in df.groupby(["regime", "M"]):
        med = float(np.median(g["binary_c4_vs_rest"]))
        p16 = float(np.percentile(g["binary_c4_vs_rest"], 16))
        p84 = float(np.percentile(g["binary_c4_vs_rest"], 84))
        four_med = float(np.median(g["balanced_acc"]))
        r1_rows.append(
            dict(
                regime=regime,
                M=int(M),
                n=int(len(g)),
                binary_balanced_acc_median=med,
                binary_balanced_acc_p16=p16,
                binary_balanced_acc_p84=p84,
                four_class_balanced_acc_median=four_med,
                delta_binary_minus_4class=med - four_med,
            )
        )
    r1 = pd.DataFrame(r1_rows).sort_values(["regime", "M"]).reset_index(drop=True)

    # Reframe 5: 6 pairwise per row, then aggregate per (regime, M, pair)
    pair_arrs = {label: np.array([pair_balacc_from_cm(cm, a, b) for cm in cms])
                 for (a, b, label) in PAIRS}
    r5_rows = []
    for (regime, M), idx in df.groupby(["regime", "M"]).groups.items():
        idx = np.array(list(idx))
        for (_a, _b, label) in PAIRS:
            vals = pair_arrs[label][idx]
            r5_rows.append(
                dict(
                    regime=regime,
                    M=int(M),
                    pair=label,
                    n=int(len(vals)),
                    pair_balanced_acc_median=float(np.median(vals)),
                    pair_balanced_acc_p16=float(np.percentile(vals, 16)),
                    pair_balanced_acc_p84=float(np.percentile(vals, 84)),
                )
            )
    r5 = pd.DataFrame(r5_rows).sort_values(["regime", "M", "pair"]).reset_index(drop=True)

    # Verdicts ------------------------------------------------------------
    verdicts: dict = {}

    # Reframe 1: per_sightline @ M=64
    sel = r1[(r1["regime"] == "per_sightline") & (r1["M"] == 64)].iloc[0]
    sel_global = r1[(r1["regime"] == "global") & (r1["M"] == 64)].iloc[0]
    p16_pass = sel["binary_balanced_acc_p16"] >= 0.90
    delta_pass = sel["delta_binary_minus_4class"] >= 0.10
    parity = abs(sel["binary_balanced_acc_median"]
                 - sel_global["binary_balanced_acc_median"]) <= 0.005
    verdicts["reframe1"] = dict(
        pass_overall=bool(p16_pass and delta_pass),
        p16_at_M64_persightline=float(sel["binary_balanced_acc_p16"]),
        median_at_M64_persightline=float(sel["binary_balanced_acc_median"]),
        delta_vs_4class=float(sel["delta_binary_minus_4class"]),
        median_at_M64_global=float(sel_global["binary_balanced_acc_median"]),
        regime_parity_delta=float(sel["binary_balanced_acc_median"]
                                  - sel_global["binary_balanced_acc_median"]),
        regime_parity_pass=bool(parity),
    )

    # Reframe 5: per_sightline @ M=64
    r5_sel = r5[(r5["regime"] == "per_sightline") & (r5["M"] == 64)].copy()
    non_c2c3 = r5_sel[r5_sel["pair"] != "C2-C3"]
    cleared = non_c2c3["pair_balanced_acc_p16"] >= 0.75
    n_cleared = int(cleared.sum())
    count_pass = n_cleared >= 3
    # Structural: at least one of {C1-C2, C1-C3} OR one of {C2-C4, C3-C4} above 0.75
    nofb_pairs = r5_sel[r5_sel["pair"].isin(["C1-C2", "C1-C3"])]
    fbfb_pairs = r5_sel[r5_sel["pair"].isin(["C2-C4", "C3-C4"])]
    nofb_above = (nofb_pairs["pair_balanced_acc_p16"] >= 0.75).any()
    fbfb_above = (fbfb_pairs["pair_balanced_acc_p16"] >= 0.75).any()
    struct_pass = bool(nofb_above or fbfb_above)
    verdicts["reframe5"] = dict(
        pass_count=bool(count_pass),
        pass_structural=struct_pass,
        pass_overall=bool(count_pass and struct_pass),
        n_cleared_of_5=n_cleared,
        cleared_pairs=[p for p, ok in zip(non_c2c3["pair"], cleared) if ok],
        per_pair_p16={r["pair"]: float(r["pair_balanced_acc_p16"])
                      for _, r in r5_sel.iterrows()},
    )

    return r1, r5, verdicts


# ---------------------------------------------------------------------------
# Reframe 7 — K=5 cluster physical-axis interpretation
# ---------------------------------------------------------------------------

def compute_physical_axes(tau_min: float = 0.05) -> dict:
    """
    Compute three physical axes per (class, position):

      mean_flux       : mean of F along the 2048-pixel sightline.
      integrated_tau  : mean of tau along the 2048-pixel sightline
                        (thermal-state proxy).
      line_count      : # peaks via scipy.signal.find_peaks on tau
                        with height=tau_min, prominence=tau_min,
                        distance=1. (Matches Q-C1 audit protocol.)

    Returns a dict with shape (4, 16384) arrays per axis (4 classes
    by 16384 positions).
    """
    n_classes = 4
    n_positions = 16384
    mean_flux = np.zeros((n_classes, n_positions), dtype=np.float64)
    integrated_tau = np.zeros((n_classes, n_positions), dtype=np.float64)
    line_count = np.zeros((n_classes, n_positions), dtype=np.int32)

    for ci, cls in enumerate(range(1, 5)):
        flux_path = PREPROCESSED / str(cls) / "flux.npy"
        tau_path = PREPROCESSED / str(cls) / "tau.npy"
        print(f"[Reframe 7] Loading class {cls}: {flux_path.name}, {tau_path.name}")
        flux = np.load(flux_path, mmap_mode="r")
        tau = np.load(tau_path, mmap_mode="r")
        assert flux.shape == (n_positions, 2048), f"unexpected flux shape {flux.shape}"
        assert tau.shape == (n_positions, 2048), f"unexpected tau shape {tau.shape}"

        # mean flux per sightline
        mean_flux[ci] = np.asarray(flux).mean(axis=1)
        # integrated tau per sightline (mean across pixels; per SCOPING brief)
        integrated_tau[ci] = np.asarray(tau).mean(axis=1)
        # line count via find_peaks with Q-C1 protocol
        print(f"[Reframe 7]   computing line_count via find_peaks, n={n_positions} ...")
        tau_arr = np.asarray(tau)
        counts = np.empty(n_positions, dtype=np.int32)
        for i in range(n_positions):
            peaks, _ = find_peaks(
                tau_arr[i], height=tau_min, prominence=tau_min, distance=1
            )
            counts[i] = peaks.size
        line_count[ci] = counts

    return dict(
        mean_flux=mean_flux,
        integrated_tau=integrated_tau,
        line_count=line_count.astype(np.float64),
    )


def eta_squared_oneway(values: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute one-way ANOVA F-statistic, p-value, and eta-squared.

    eta^2 = SS_between / SS_total

    Returns (eta2, F, p).
    """
    unique = np.unique(labels)
    groups = [values[labels == g] for g in unique]
    F, p = f_oneway(*groups)

    grand_mean = values.mean()
    ss_between = sum(
        len(grp) * (grp.mean() - grand_mean) ** 2 for grp in groups
    )
    ss_total = float(((values - grand_mean) ** 2).sum())
    eta2 = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    return eta2, float(F), float(p)


def reframe7() -> Tuple[pd.DataFrame, dict]:
    labels_wav = np.load(FEATURE_DISCOVERY / "labels_wavelet_k5.npy")
    labels_raw = np.load(FEATURE_DISCOVERY / "labels_raw_k5.npy")
    assert labels_wav.shape == (16384,) and labels_raw.shape == (16384,)
    print(f"[Reframe 7] labels_wavelet unique={np.unique(labels_wav)}")
    print(f"[Reframe 7] labels_raw     unique={np.unique(labels_raw)}")

    axes = compute_physical_axes()

    # Class-averaged axis: mean across the 4 class realizations at each
    # of the 16384 positions, giving one scalar per position aligned
    # with the (16384,) cluster label vector.
    axis_avg = {
        name: arr.mean(axis=0)  # shape (16384,)
        for name, arr in axes.items()
    }

    rows = []
    for cluster_set, labels in [("wavelet_k5", labels_wav), ("raw_k5", labels_raw)]:
        for axis_name in ("mean_flux", "integrated_tau", "line_count"):
            eta2, F, p = eta_squared_oneway(axis_avg[axis_name], labels)
            rows.append(
                dict(
                    cluster_set=cluster_set,
                    physical_axis=axis_name,
                    eta_squared=eta2,
                    f_statistic=F,
                    p_value=p,
                    aggregation="class_averaged",
                )
            )
            # Per-class breakouts (honest backup, not headline)
            for ci, cls in enumerate(range(1, 5)):
                eta2_c, F_c, p_c = eta_squared_oneway(axes[axis_name][ci], labels)
                rows.append(
                    dict(
                        cluster_set=cluster_set,
                        physical_axis=axis_name,
                        eta_squared=eta2_c,
                        f_statistic=F_c,
                        p_value=p_c,
                        aggregation=f"class_{cls}",
                    )
                )

    r7 = pd.DataFrame(rows)

    headline = r7[r7["aggregation"] == "class_averaged"].copy()
    best = headline.loc[headline["eta_squared"].idxmax()]
    pass_overall = bool(headline["eta_squared"].max() >= 0.20)

    verdicts = dict(
        pass_overall=pass_overall,
        top_eta2=float(best["eta_squared"]),
        top_cluster_set=str(best["cluster_set"]),
        top_axis=str(best["physical_axis"]),
        all_class_averaged={
            f"{r['cluster_set']}|{r['physical_axis']}": float(r["eta_squared"])
            for _, r in headline.iterrows()
        },
    )
    return r7, verdicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("reframe-suite Phase 1 — Reframes 1, 5, 7")
    print("=" * 72)

    r1, r5, verdicts_15 = reframe1_and_5()
    r1_path = OUT_DIR / "reframe1_binary_c4_vs_rest.csv"
    r5_path = OUT_DIR / "reframe5_lattice.csv"
    r1.to_csv(r1_path, index=False)
    r5.to_csv(r5_path, index=False)
    print(f"\n[Reframe 1] wrote {r1_path}")
    print(f"[Reframe 5] wrote {r5_path}")

    r7, verdicts_7 = reframe7()
    r7_path = OUT_DIR / "reframe7_cluster_axes.csv"
    r7.to_csv(r7_path, index=False)
    print(f"[Reframe 7] wrote {r7_path}")

    print("\n" + "=" * 72)
    print("VERDICTS")
    print("=" * 72)
    v1 = verdicts_15["reframe1"]
    print(f"Reframe 1: {'PASS' if v1['pass_overall'] else 'FAIL'}")
    print(f"  per_sightline @ M=64: median={v1['median_at_M64_persightline']:.4f}, "
          f"p16={v1['p16_at_M64_persightline']:.4f}")
    print(f"  delta vs 4-class median: {v1['delta_vs_4class']:.4f}")
    print(f"  regime parity (|per_sightline - global|): "
          f"{abs(v1['regime_parity_delta']):.4f} (pass={v1['regime_parity_pass']})")
    print(f"  global @ M=64 median: {v1['median_at_M64_global']:.4f}")

    v5 = verdicts_15["reframe5"]
    print(f"\nReframe 5: {'PASS' if v5['pass_overall'] else 'FAIL'} "
          f"(count={v5['pass_count']}, structural={v5['pass_structural']})")
    print(f"  cleared {v5['n_cleared_of_5']}/5 non-(C2-C3) pairs at p16 >= 0.75")
    print(f"  cleared pairs: {v5['cleared_pairs']}")
    print(f"  per-pair p16 @ per_sightline,M=64: {v5['per_pair_p16']}")

    v7 = verdicts_7
    print(f"\nReframe 7: {'PASS' if v7['pass_overall'] else 'FAIL'}")
    print(f"  top eta^2 = {v7['top_eta2']:.4f}  "
          f"({v7['top_cluster_set']} x {v7['top_axis']})")
    print(f"  all class-averaged eta^2: {v7['all_class_averaged']}")

    # Phase-2 authorization
    phase2 = v1["pass_overall"]
    print("\n" + "=" * 72)
    print(f"PHASE-2 GATE: {'AUTHORIZED' if phase2 else 'NOT-AUTHORIZED'} "
          f"(per SCOPING §3 — gated on Reframe 1 PASS)")
    print("=" * 72)


if __name__ == "__main__":
    main()
