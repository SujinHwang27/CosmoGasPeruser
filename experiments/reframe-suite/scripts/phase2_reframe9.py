"""
reframe-suite Phase 2 — Reframe 9 (mean-flux-removed P_F(k) variant).

Scientific purpose
------------------
Re-runs a focused subset of Stage 1 of the pk-feedback-classifier track
with one preprocessing change: replace [D-06]'s k=0 + lowest-log-bin drop
with explicit per-sightline <F> subtraction in flux space prior to FT
(``norm='per_sightline_mean_removed'`` in
``src/core/transforms.py:FluxPowerSpectrum``).

Reframe 9 is the load-bearing honesty-hurdle control on Reframe 1 — the
shape-vs-<F> disentanglement. Per SCOPING §2 R9 (a), PASS iff binary
{C4 vs rest} balanced_acc at M=64 drops by less than W = 0.05 (5 pp)
relative to the Reframe-1 baseline at (per_sightline, M=64): median
0.971, p16 0.960.

Protocol (per SCOPING §3 Phase 2 budget)
----------------------------------------
- regime: ``per_sightline_mean_removed`` only.
- M-sweep: {64, 256}.
- seeds: {42, 43, 44, 45, 46} (5 seeds; half of Stage 1's 10 — Phase 2
  "250 fits" budget).
- folds: 5 (StratifiedKFold over post-stacking groups; [D-20] S3
  fold-leakage assert preserved).
- Output schema mirrors Stage 1's ``cv_partial.csv`` (so the Phase 1
  binary aggregator can be re-applied to the new confusions).

Invocation
----------
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/phase2_reframe9.py

Outputs to ``results/reframe_suite/phase2/``:
    cv_partial_meanremoved.csv  — per (M, seed, fold) row, includes
                                  confusion_flat + 4-class metrics.
    summary_meanremoved.csv     — per M, median + p16 + p84.
    reframe9_verdict.csv        — binary {C4 vs rest} at M=64
                                  vs Reframe-1 baseline; verdict + drop.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.core.data import (
    SignalClusteringData,
    assert_velocity_axes_uniform,
    group_indices_within_class,
)
from src.core.transforms import FluxPowerSpectrum


# ---------------------------------------------------------------------------
# Configuration (Phase 2 Reframe 9 budget per SCOPING §3)
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
RESULTS_DIR: Path = REPO_ROOT / "results" / "reframe_suite" / "phase2"

N_KBINS: int = 20
M_VALUES: List[int] = [64, 256]
REGIME: str = "per_sightline_mean_removed"
SEEDS: List[int] = [42, 43, 44, 45, 46]  # 5 seeds (Phase 2 half-budget)
N_FOLDS: int = 5

RF_N_ESTIMATORS: int = 300

# Reframe-1 baseline (from PHASE1_REPORT §2): per_sightline @ M=64.
REFRAME1_BASELINE_MEDIAN: float = 0.9711
REFRAME1_BASELINE_P16: float = 0.9603

# SCOPING §2 R9 (a) PASS bar.
W_PASS: float = 0.05

# CSV columns mirror Stage 1 (subset; Phase 2 doesn't run G2/G5/G6).
CV_PARTIAL_COLUMNS: List[str] = [
    "regime", "M", "seed", "fold",
    "n_train_groups", "n_test_groups",
    "balanced_acc",
    "binary_c4_vs_rest",
    "mid_k_frac",
    "confusion_flat",
    "runtime_sec",
]

# G3 mid-k band (kept for cross-check parity with Stage 1).
MID_K_LOW: float = 0.04
MID_K_HIGH: float = 0.15


# ---------------------------------------------------------------------------
# Data + transform helpers (mirror run_stage1.py, see headers there)
# ---------------------------------------------------------------------------
def _load_flux_per_class() -> List[np.ndarray]:
    """Load 4-class flux as a list of (N, 2048) float64 arrays."""
    loader = SignalClusteringData()
    flux_list, _ = loader.load_flux_per_class()
    return [np.asarray(f, dtype=np.float64) for f in flux_list]


def _compute_pk_per_sightline(
    flux_per_class: List[np.ndarray], regime: str, delta_v: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Compute P_F(k) per sightline for each class under the given norm."""
    pk_per_class: List[np.ndarray] = []
    k_centers: Optional[np.ndarray] = None
    for flux in flux_per_class:
        t = FluxPowerSpectrum(norm=regime, n_kbins=N_KBINS, delta_v=delta_v)
        pk = t.fit_transform(flux)  # shape: (N, n_kbins_eff)
        assert np.all(np.isfinite(pk)), f"non-finite P(k) in regime={regime}"
        pk_per_class.append(pk)
        if k_centers is None:
            k_centers = np.asarray(t.k_bin_centers_, dtype=np.float64)
    assert k_centers is not None
    return pk_per_class, k_centers


def _stack_pk_within_class(
    pk_per_class: List[np.ndarray], M: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]:
    """Average per-sightline P(k) into M-sized disjoint random groups per class.

    Returns
    -------
    Z : (n_total_groups, n_kbins_eff) float64
    y : (n_total_groups,) int64 — class label in {1,2,3,4}.
    membership : list of length n_total_groups; each entry is
        (class_idx_in_{0..3}, np.ndarray of source-sightline indices of len M).
    """
    z_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    membership: List[Tuple[int, np.ndarray]] = []
    for c_idx, pk in enumerate(pk_per_class):
        n = pk.shape[0]
        groups = group_indices_within_class(n, M=M, seed=seed)
        chunk = np.stack([pk[g].mean(axis=0) for g in groups], axis=0)
        z_chunks.append(chunk)
        y_chunks.append(np.full(chunk.shape[0], c_idx + 1, dtype=np.int64))
        for g in groups:
            membership.append((c_idx, np.asarray(g, dtype=np.int64)))
    Z = np.vstack(z_chunks)
    y = np.concatenate(y_chunks)
    assert Z.dtype == np.float64
    assert np.all(np.isfinite(Z))
    return Z, y, membership


def _mid_k_mask(k_centers: np.ndarray) -> np.ndarray:
    return (k_centers >= MID_K_LOW) & (k_centers <= MID_K_HIGH)


def _binary_c4_vs_rest_balacc(cm_norm: np.ndarray) -> float:
    """Same aggregator as Phase 1 ``phase1_reframes.binary_c4_vs_rest_balacc``.

    cm_norm rows are row-normalized recall over labels 1..4 (indices 0..3).
    Sensitivity = cm[3, 3], specificity = mean over rows 0..2 of (1 - cm[i, 3]).
    """
    sens = cm_norm[3, 3]
    spec_per_class = 1.0 - cm_norm[0:3, 3]
    spec = float(np.mean(spec_per_class))
    return 0.5 * (float(sens) + spec)


# ---------------------------------------------------------------------------
# Per-fold compute (subset of Stage 1's _compute_one_fold)
# ---------------------------------------------------------------------------
def _compute_one_fold(
    Z: np.ndarray, y: np.ndarray,
    train_grp: np.ndarray, test_grp: np.ndarray,
    k_centers: np.ndarray, seed: int,
) -> Dict[str, Any]:
    """Train RF, return 4-class bal_acc + confusion + mid_k_frac + binary."""
    X_train = Z[train_grp]
    X_test = Z[test_grp]
    y_train = y[train_grp]
    y_test = y[test_grp]

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    labels_all = np.array([1, 2, 3, 4])
    cm = confusion_matrix(y_test, y_pred, labels=labels_all).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cm_norm = cm / row_sums

    binary_c4 = _binary_c4_vs_rest_balacc(cm_norm)

    importances = np.asarray(rf.feature_importances_, dtype=np.float64)
    mk_mask = _mid_k_mask(k_centers)
    imp_sum = float(importances.sum())
    mid_k_frac = float(importances[mk_mask].sum() / imp_sum) if imp_sum > 0 else 0.0

    return {
        "balanced_acc": bal_acc,
        "binary_c4_vs_rest": binary_c4,
        "mid_k_frac": mid_k_frac,
        "confusion_flat": ",".join(f"{v:.6f}" for v in cm_norm.flatten()),
        "n_train_groups": int(X_train.shape[0]),
        "n_test_groups": int(X_test.shape[0]),
    }


# ---------------------------------------------------------------------------
# Fold-leakage runtime assert ([D-20] S3, copied from run_stage1.py)
# ---------------------------------------------------------------------------
def _assert_no_fold_leakage(
    train_grp: np.ndarray, test_grp: np.ndarray,
    membership: List[Tuple[int, np.ndarray]],
    regime: str, M: int, seed: int, fold_id: int,
) -> None:
    train_pairs: Set[Tuple[int, int]] = set()
    test_pairs: Set[Tuple[int, int]] = set()
    for gi in train_grp:
        c_idx, src = membership[int(gi)]
        for s in src.tolist():
            train_pairs.add((c_idx, int(s)))
    for gi in test_grp:
        c_idx, src = membership[int(gi)]
        for s in src.tolist():
            test_pairs.add((c_idx, int(s)))
    overlap = train_pairs & test_pairs
    if overlap:
        print(
            f"[FATAL] fold-leakage: regime={regime} M={M} seed={seed} "
            f"fold={fold_id} — {len(overlap)} pairs leaked. "
            f"Sample: {list(overlap)[:5]}",
            file=sys.stderr,
        )
        sys.exit(7)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _load_completed_tuples(cv_path: Path) -> Set[Tuple[str, int, int, int]]:
    completed: Set[Tuple[str, int, int, int]] = set()
    if not cv_path.exists():
        return completed
    df = pd.read_csv(cv_path)
    for _, row in df.iterrows():
        completed.add(
            (str(row["regime"]), int(row["M"]), int(row["seed"]), int(row["fold"]))
        )
    return completed


def _append_cv_partial(cv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not cv_path.exists()
    with cv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CV_PARTIAL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CV_PARTIAL_COLUMNS})


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def _run_sweep(
    flux_per_class: List[np.ndarray], delta_v: float, cv_path: Path,
) -> None:
    completed = _load_completed_tuples(cv_path)
    if completed:
        print(
            f"[restart] resuming; {len(completed)} (regime, M, seed, fold) "
            f"tuples already complete"
        )

    print(
        f"\n[phase2] computing per-sightline P(k) per class "
        f"under norm={REGIME!r}..."
    )
    t0 = time.time()
    pk_per_class, k_centers = _compute_pk_per_sightline(
        flux_per_class, REGIME, delta_v
    )
    print(
        f"  ... done in {time.time()-t0:.1f}s; "
        f"k_centers shape={k_centers.shape}; "
        f"P(k) shapes={[p.shape for p in pk_per_class]}"
    )

    for M in M_VALUES:
        for seed in SEEDS:
            Z, y, membership = _stack_pk_within_class(
                pk_per_class, M=M, seed=seed
            )
            skf = StratifiedKFold(
                n_splits=N_FOLDS, shuffle=True, random_state=seed
            )
            splits = list(skf.split(np.arange(len(y)), y))

            for fold_id, (train_grp, test_grp) in enumerate(splits, start=1):
                key = (REGIME, M, seed, fold_id)
                if key in completed:
                    continue

                _assert_no_fold_leakage(
                    train_grp, test_grp, membership,
                    regime=REGIME, M=M, seed=seed, fold_id=fold_id,
                )

                t_start = time.time()
                result = _compute_one_fold(
                    Z=Z, y=y, train_grp=train_grp, test_grp=test_grp,
                    k_centers=k_centers, seed=seed,
                )
                runtime_sec = time.time() - t_start

                row = {
                    "regime": REGIME, "M": M, "seed": seed, "fold": fold_id,
                    "runtime_sec": runtime_sec,
                    **result,
                }
                _append_cv_partial(cv_path, row)
                print(
                    f"  [{REGIME} M={M} seed={seed} fold={fold_id}] "
                    f"4cls={result['balanced_acc']:.4f} "
                    f"binC4={result['binary_c4_vs_rest']:.4f} "
                    f"midk_frac={result['mid_k_frac']:.3f} "
                    f"(t={runtime_sec:.1f}s)"
                )


# ---------------------------------------------------------------------------
# Aggregation + verdict
# ---------------------------------------------------------------------------
def _aggregate_summary(cv_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cv_path)
    rows: List[Dict[str, Any]] = []
    for (regime, M), g in df.groupby(["regime", "M"]):
        rows.append({
            "regime": regime, "M": int(M),
            "n_rows": int(len(g)),
            "four_class_balanced_acc_median": float(np.median(g["balanced_acc"])),
            "four_class_balanced_acc_p16": float(np.percentile(g["balanced_acc"], 16)),
            "four_class_balanced_acc_p84": float(np.percentile(g["balanced_acc"], 84)),
            "binary_c4_vs_rest_median": float(np.median(g["binary_c4_vs_rest"])),
            "binary_c4_vs_rest_p16": float(np.percentile(g["binary_c4_vs_rest"], 16)),
            "binary_c4_vs_rest_p84": float(np.percentile(g["binary_c4_vs_rest"], 84)),
            "mid_k_frac_median": float(np.median(g["mid_k_frac"])),
        })
    summary = pd.DataFrame(rows).sort_values(["regime", "M"]).reset_index(drop=True)
    summary.to_csv(out_path, index=False)
    print(f"[summary] wrote {out_path}")
    return summary


def _compute_verdict(summary: pd.DataFrame, out_path: Path) -> Dict[str, Any]:
    sel = summary[(summary["regime"] == REGIME) & (summary["M"] == 64)]
    if sel.empty:
        raise RuntimeError(
            f"No summary row for (regime={REGIME}, M=64). cv_partial incomplete?"
        )
    sel = sel.iloc[0]
    bin_median = float(sel["binary_c4_vs_rest_median"])
    bin_p16 = float(sel["binary_c4_vs_rest_p16"])

    drop_median = REFRAME1_BASELINE_MEDIAN - bin_median
    drop_p16 = REFRAME1_BASELINE_P16 - bin_p16

    pass_median = drop_median < W_PASS  # SCOPING §2 R9 (a): drop < 5 pp on median
    verdict = "PASS" if pass_median else "FAIL"

    out_rows = [{
        "metric": "binary_c4_vs_rest_median_at_M64",
        "phase2_value": bin_median,
        "phase1_baseline": REFRAME1_BASELINE_MEDIAN,
        "drop": drop_median,
        "W_pass_threshold": W_PASS,
        "pass": pass_median,
    }, {
        "metric": "binary_c4_vs_rest_p16_at_M64",
        "phase2_value": bin_p16,
        "phase1_baseline": REFRAME1_BASELINE_P16,
        "drop": drop_p16,
        "W_pass_threshold": W_PASS,
        "pass": drop_p16 < W_PASS,
    }, {
        "metric": "verdict_overall",
        "phase2_value": verdict,
        "phase1_baseline": "",
        "drop": "",
        "W_pass_threshold": W_PASS,
        "pass": pass_median,
    }]
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"[verdict] wrote {out_path}")
    return {
        "binary_c4_median_M64": bin_median,
        "binary_c4_p16_M64": bin_p16,
        "drop_median": drop_median,
        "drop_p16": drop_p16,
        "verdict": verdict,
        "pass": pass_median,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_path = RESULTS_DIR / "cv_partial_meanremoved.csv"
    summary_path = RESULTS_DIR / "summary_meanremoved.csv"
    verdict_path = RESULTS_DIR / "reframe9_verdict.csv"

    print(f"[phase2] results dir = {RESULTS_DIR}")
    print(f"[phase2] regime={REGIME} M_values={M_VALUES} seeds={SEEDS} folds={N_FOLDS}")
    print(f"[phase2] expected fits = {len(M_VALUES) * len(SEEDS) * N_FOLDS} (5 main RF per (M,seed)x5 folds)")

    print("[phase2] loading flux per class...")
    flux_per_class = _load_flux_per_class()
    for i, f in enumerate(flux_per_class, start=1):
        print(
            f"   class {i}: shape={f.shape} dtype={f.dtype} "
            f"min={f.min():.3e} max={f.max():.6f} mean={f.mean():.6f}"
        )

    print("[phase2] asserting uniform velocity axis across classes...")
    delta_v = assert_velocity_axes_uniform()
    print(f"   delta_v = {delta_v} km/s/pixel")

    t_overall = time.time()
    _run_sweep(flux_per_class, delta_v, cv_path)

    print("\n[phase2] aggregating...")
    summary = _aggregate_summary(cv_path, summary_path)
    verdict = _compute_verdict(summary, verdict_path)

    print()
    print("=" * 72)
    print("PHASE 2 -- REFRAME 9 VERDICT")
    print("=" * 72)
    print(f"  regime: {REGIME}")
    print(f"  binary {{C4 vs rest}} @ M=64:")
    print(f"    median = {verdict['binary_c4_median_M64']:.4f}  "
          f"(baseline {REFRAME1_BASELINE_MEDIAN}, drop = {verdict['drop_median']:+.4f})")
    print(f"    p16    = {verdict['binary_c4_p16_M64']:.4f}  "
          f"(baseline {REFRAME1_BASELINE_P16}, drop = {verdict['drop_p16']:+.4f})")
    print(f"  W_pass threshold = {W_PASS} (SCOPING section 2 R9 (a))")
    print(f"  VERDICT: {verdict['verdict']}")
    print("=" * 72)
    print(f"\n[phase2] wall time: {(time.time()-t_overall)/60:.1f} min")


if __name__ == "__main__":
    main()
