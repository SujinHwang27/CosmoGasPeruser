"""Stage 1 six-gate G1-G6 run for the pk-feedback-classifier track.

Binding: experiments/pk-feedback-classifier/LEDGER.md
  [D-09] Stage 1 spec, [D-15]-[D-22] amendment block, [D-23] C1/C3/C6/C7 binds.

Implements the unified-sprint G1-G6 protocol over a denser M-sweep
({1, 4, 16, 32, 64, 128, 256}) on both normalization regimes
(per_sightline PRIMARY, global SECONDARY) with 5-fold StratifiedKFold over
post-stacking GROUPS ([D-20] S3) and 10-seed bootstrap. Permutation null for
G2 (M=64, per_sightline only per [D-18]); G6 ablation drop of top-3 mid-k bins
(idx 10/11/12) with threshold max(5pp, 2 * empirical SD) per [D-23] C1; G5
per-class |Pearson r(<F>_los, predict_proba_max)| reported and routed per
[D-23] C7 (C4-only violator vs physics-class violator).

NO DVC stage. NO MLflow per [D-01]. Single self-contained CPU script.

The 64-cell G1-G6 outcome-band routing table lives in STAGE1_SPEC.md (per
[D-23] C5) and is NOT encoded here — this script emits raw gate verdicts; the
spec doc maps verdict-tuples to outcome bands and must be PI-APPROVE'd before
sbatch fires.

Invoke from repo root:
    PYTHONPATH=. uv run python experiments/pk-feedback-classifier/run_stage1.py
    PYTHONPATH=. uv run python experiments/pk-feedback-classifier/run_stage1.py --smoke

Outputs to results/pk_feedback_classifier/stage1/.

Honest-reporting note ([D-37]): the runtime estimate below is order-of-magnitude.
Stage 0 measured ~1m 4s on Juno `normal` for ~ a dozen RF fits + transforms;
Stage 1 has ~ 1400 main RF fits (2 regimes x 7 M x 10 seeds x 5 folds x 2
classifiers) + ~5000 permutation fits + ~140 ablation fits ~= 6500-8000 fits.
Most fits are faster than Stage 0's M=1 due to smaller N_train at high M, but
per-sightline P(k) transform overhead at M=1 (65536 sightlines) is the
dominant slice. Realistic estimate: 1.5-4h wall on Juno `normal`. If above 6h
this triggers [D-23] C4 (smoke-timing + ceiling-raise decision).
"""
# RUNTIME_ESTIMATE_HOURS = 6  # honest revision after smoke timing
# Smoke (1 regime, M=4 only, 1 seed, 5 folds + ablation) took 16.6 min wall
# locally; ~13 min was the 5 main fits + 5 ablation fits, ~3 min was the
# per-sightline P(k) compute. Full sweep multiplies fits by ~140x (2 regimes
# x 7 M-values x 10 seeds; ablation included inline), but at high M the
# per-fold N_train shrinks (16384/M groups -> M=256 has only 64 groups/class
# so RF fit is sub-second). M=1 (per-sightline, 65536 rows) and M=4
# (16384 rows) dominate. Local laptop wall estimate: 4-8h; on Juno `normal`
# 16-core, likely 3-5h. Above 6h triggers [D-23] C4 -- infrastructure-manager
# does the smoke-timing + ceiling-raise decision pre-sbatch.
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
# Configuration (binding per LEDGER [D-09] + [D-15]-[D-23])
# ---------------------------------------------------------------------------
RESULTS_DIR: Path = Path("results/pk_feedback_classifier/stage1")

N_KBINS: int = 20
M_VALUES: List[int] = [1, 4, 16, 32, 64, 128, 256]
REGIMES: List[str] = ["per_sightline", "global"]
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10-seed bootstrap
N_FOLDS: int = 5

# G3 mid-k band per [D-09] / [D-11].
MID_K_LOW: float = 0.04  # s/km
MID_K_HIGH: float = 0.15  # s/km
G3_MEDIAN_THRESHOLD: float = 0.50

# G1 lower-CI threshold per [D-09].
G1_LOWER_CI_THRESHOLD: float = 0.65

# G4 median threshold per [D-09].
G4_MEDIAN_THRESHOLD: float = 0.60

# G5 per-class |Pearson r| threshold per [D-21].
G5_R_THRESHOLD: float = 0.30

# G6 ablation per [D-16] / [D-23] C1.
G6_BIN_FLOOR_PP: float = 0.05  # 5 percentage points
G6_ABLATION_BIN_IDX: Tuple[int, int, int] = (10, 11, 12)
G6_REPRESENTATIVE_M: int = 64
G6_REPRESENTATIVE_REGIME: str = "per_sightline"

# G2 permutation null per [D-18].
N_PERMUTATIONS: int = 1000
G2_PERCENTILE: float = 99.0
G2_M: int = 64
G2_REGIME: str = "per_sightline"
G2_CLASSES: Tuple[int, int, int] = (1, 2, 3)

# RF protocol per [D-09].
RF_N_ESTIMATORS: int = 300

# CSV columns for cv_partial.csv — bound to the parser used on restart.
CV_PARTIAL_COLUMNS: List[str] = [
    "regime", "M", "seed", "fold",
    "n_train_groups", "n_test_groups",
    "balanced_acc",
    "balanced_acc_3class",
    "bal_acc_c2_vs_c3",
    "mid_k_frac",
    "abs_r_c1", "abs_r_c2", "abs_r_c3", "abs_r_c4",
    "ablation_acc_drop",
    "confusion_flat",
    "kbin_importance_flat",
    "runtime_sec",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 G1-G6 run (per LEDGER [D-09]/[D-23]).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke run: 1 regime (per_sightline), M=4 only, 1 seed (42), "
             "5 folds. Skips G2 permutations + G6 ablation. ~1 min wall.",
    )
    p.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help="Results dir (default: results/pk_feedback_classifier/stage1).",
    )
    p.add_argument(
        "--skip-permutation-null", action="store_true",
        help="Skip the G2 permutation null (debugging convenience).",
    )
    p.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip the G6 ablation pass (debugging convenience).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data + transform helpers
# ---------------------------------------------------------------------------
def _load_flux_per_class() -> List[np.ndarray]:
    """Load 4-class flux as a list of (N, 2048) float64 arrays."""
    loader = SignalClusteringData()
    flux_list, _ = loader.load_flux_per_class()
    return [np.asarray(f, dtype=np.float64) for f in flux_list]


def _compute_pk_per_sightline(
    flux_per_class: List[np.ndarray], regime: str, delta_v: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Compute P_F(k) per sightline for each class.

    Returns
    -------
    pk_per_class : list of (N, n_kbins_eff) float64
    k_centers : (n_kbins_eff,) float64 — k-bin centers (s/km).
    """
    pk_per_class: List[np.ndarray] = []
    k_centers: Optional[np.ndarray] = None
    for flux in flux_per_class:
        t = FluxPowerSpectrum(norm=regime, n_kbins=N_KBINS, delta_v=delta_v)
        pk = t.fit_transform(flux)  # shape: (N, n_kbins or n_kbins-1)
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
    Z : (n_total_groups, n_kbins_eff) float64 — stacked P(k) feature matrix.
    y : (n_total_groups,) int64 — class label in {1,2,3,4} per group.
    membership : list of length n_total_groups; each entry is
        (class_idx_in_{0..3}, np.ndarray of source-sightline indices of len M).
        Used to materialize source-sightline sets for the fold-leakage assert.
    """
    z_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    membership: List[Tuple[int, np.ndarray]] = []
    for c_idx, pk in enumerate(pk_per_class):
        n = pk.shape[0]
        groups = group_indices_within_class(n, M=M, seed=seed)
        # shape: (n_groups, n_kbins_eff)
        chunk = np.stack([pk[g].mean(axis=0) for g in groups], axis=0)
        z_chunks.append(chunk)
        y_chunks.append(np.full(chunk.shape[0], c_idx + 1, dtype=np.int64))
        for g in groups:
            membership.append((c_idx, np.asarray(g, dtype=np.int64)))
    Z = np.vstack(z_chunks)  # shape: (n_total_groups, n_kbins_eff)
    y = np.concatenate(y_chunks)  # shape: (n_total_groups,)
    assert Z.dtype == np.float64
    assert np.all(np.isfinite(Z))
    return Z, y, membership


def _mean_flux_per_group(
    flux_per_class: List[np.ndarray],
    membership: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    """Compute the mean flux <F>_los per post-stacking group.

    For each group of M source sightlines, returns the mean across all pixels
    of those M sightlines' raw flux. This is the "per-group <F>" used in G5 —
    correlates with predict_proba_max to surface mean-flux leakage.

    Returns
    -------
    mean_flux : (n_total_groups,) float64
    """
    out = np.empty(len(membership), dtype=np.float64)
    for gi, (c_idx, src_idx) in enumerate(membership):
        out[gi] = float(flux_per_class[c_idx][src_idx].mean())
    return out


# ---------------------------------------------------------------------------
# Mid-k mask + ablation
# ---------------------------------------------------------------------------
def _mid_k_mask(k_centers: np.ndarray) -> np.ndarray:
    """Boolean mask over k-bin centers in the G3 mid-k band [0.04, 0.15] s/km."""
    return (k_centers >= MID_K_LOW) & (k_centers <= MID_K_HIGH)


def _ablation_indices(k_centers: np.ndarray) -> List[int]:
    """Resolve the G6 top-3 mid-k bin indices.

    Per [D-16], these are 'k-axis indices 10, 11, 12' relative to the 20-bin
    log-spaced axis. In the global regime the lowest log-bin is dropped, so
    indices shift by -1; we resolve against the supplied k_centers length.
    """
    n = len(k_centers)
    # Per-sightline regime has 20 bins; global has 19.
    if n == N_KBINS:
        return list(G6_ABLATION_BIN_IDX)
    elif n == N_KBINS - 1:
        # Shift indices by -1 to track the same physical k bins.
        return [i - 1 for i in G6_ABLATION_BIN_IDX]
    else:
        raise ValueError(
            f"unexpected k_centers length {n}; expected {N_KBINS} or {N_KBINS-1}"
        )


# ---------------------------------------------------------------------------
# Per-fold compute
# ---------------------------------------------------------------------------
def _compute_one_fold(
    Z: np.ndarray,
    y: np.ndarray,
    mean_flux_per_group: np.ndarray,
    train_grp: np.ndarray,
    test_grp: np.ndarray,
    k_centers: np.ndarray,
    seed: int,
    run_ablation: bool,
) -> Dict[str, Any]:
    """Train RF on this fold, return all G1-G6 per-fold quantities.

    Includes the [D-20] S3 runtime fold-leakage assert (defense-in-depth).
    """
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

    # Per-class confusion (row-normalized) over labels 1-4.
    labels_all = np.array([1, 2, 3, 4])
    cm = confusion_matrix(y_test, y_pred, labels=labels_all).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cm_norm = cm / row_sums

    # 3-class restricted balanced_acc on {1,2,3} test stacks only.
    mask3 = np.isin(y_test, np.array(G2_CLASSES))
    if mask3.sum() > 0:
        bal_acc_3class = float(
            balanced_accuracy_score(y_test[mask3], y_pred[mask3])
        )
    else:
        bal_acc_3class = float("nan")

    # G4: C2-vs-C3 binary sub-confusion balanced_acc — restricted to test
    # stacks whose TRUE label is in {2,3}, asking whether the RF distinguishes
    # them (treat any pred not in {2,3} as a confusion landing on the OTHER
    # of the two — concretely, use a fallback majority of the {2,3} pair).
    mask23 = np.isin(y_test, np.array([2, 3]))
    if mask23.sum() > 0:
        y_true_23 = y_test[mask23]
        y_pred_23 = y_pred[mask23].copy()
        # For preds outside {2,3}, treat as wrong-class within the pair (pick
        # the OPPOSITE of the true label to penalize cross-pair leakage).
        bad = ~np.isin(y_pred_23, np.array([2, 3]))
        y_pred_23[bad] = np.where(y_true_23[bad] == 2, 3, 2)
        bal_acc_c2_vs_c3 = float(balanced_accuracy_score(y_true_23, y_pred_23))
    else:
        bal_acc_c2_vs_c3 = float("nan")

    # G3: mid-k importance fraction.
    importances = np.asarray(rf.feature_importances_, dtype=np.float64)
    mk_mask = _mid_k_mask(k_centers)
    imp_sum = float(importances.sum())
    mid_k_frac = float(importances[mk_mask].sum() / imp_sum) if imp_sum > 0 else 0.0

    # G5: per-class |Pearson r(<F>_los, predict_proba_max)| on the TEST stacks.
    proba = rf.predict_proba(X_test)  # shape: (n_test, n_classes)
    proba_max = proba.max(axis=1)
    test_mean_flux = mean_flux_per_group[test_grp]
    abs_r_per_class: Dict[int, float] = {}
    for c in (1, 2, 3, 4):
        m = y_test == c
        if m.sum() >= 3 and np.std(test_mean_flux[m]) > 0 and np.std(proba_max[m]) > 0:
            r, _ = pearsonr(test_mean_flux[m], proba_max[m])
            abs_r_per_class[c] = float(abs(r))
        else:
            abs_r_per_class[c] = float("nan")

    # G6: ablation — zero out the top-3 mid-k bins, retrain, recompute bal_acc.
    if run_ablation:
        ablation_idx = _ablation_indices(k_centers)
        X_train_abl = X_train.copy()
        X_test_abl = X_test.copy()
        X_train_abl[:, ablation_idx] = 0.0
        X_test_abl[:, ablation_idx] = 0.0
        rf_abl = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1
        )
        rf_abl.fit(X_train_abl, y_train)
        bal_acc_abl = float(
            balanced_accuracy_score(y_test, rf_abl.predict(X_test_abl))
        )
        ablation_drop = bal_acc - bal_acc_abl
    else:
        ablation_drop = float("nan")

    return {
        "balanced_acc": bal_acc,
        "balanced_acc_3class": bal_acc_3class,
        "bal_acc_c2_vs_c3": bal_acc_c2_vs_c3,
        "mid_k_frac": mid_k_frac,
        "abs_r_c1": abs_r_per_class[1],
        "abs_r_c2": abs_r_per_class[2],
        "abs_r_c3": abs_r_per_class[3],
        "abs_r_c4": abs_r_per_class[4],
        "ablation_acc_drop": ablation_drop,
        "confusion_flat": ",".join(f"{v:.6f}" for v in cm_norm.flatten()),
        "kbin_importance_flat": ",".join(f"{v:.6f}" for v in importances),
        "n_train_groups": int(X_train.shape[0]),
        "n_test_groups": int(X_test.shape[0]),
    }


# ---------------------------------------------------------------------------
# Restart / checkpointing
# ---------------------------------------------------------------------------
def _load_completed_tuples(cv_partial_path: Path) -> Set[Tuple[str, int, int, int]]:
    """Read existing cv_partial.csv (if any) and return completed (regime, M, seed, fold) keys."""
    completed: Set[Tuple[str, int, int, int]] = set()
    if not cv_partial_path.exists():
        return completed
    df = pd.read_csv(cv_partial_path)
    for _, row in df.iterrows():
        completed.add(
            (str(row["regime"]), int(row["M"]), int(row["seed"]), int(row["fold"]))
        )
    return completed


def _append_cv_partial(cv_partial_path: Path, row: Dict[str, Any]) -> None:
    """Append a single (regime, M, seed, fold) row to cv_partial.csv."""
    write_header = not cv_partial_path.exists()
    with cv_partial_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CV_PARTIAL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CV_PARTIAL_COLUMNS})


# ---------------------------------------------------------------------------
# Fold-leakage runtime assert ([D-20] S3 + [D-23] C6 defense-in-depth)
# ---------------------------------------------------------------------------
def _assert_no_fold_leakage(
    train_grp: np.ndarray,
    test_grp: np.ndarray,
    membership: List[Tuple[int, np.ndarray]],
    regime: str, M: int, seed: int, fold_id: int,
) -> None:
    """Materialize source-sightline sets for train/test stacks; assert disjoint.

    Failure exits the process with code 7 (distinct from PCV codes 2-6 used
    by the sbatch wrapper).
    """
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
            f"[FATAL] fold-leakage detected: regime={regime} M={M} seed={seed} "
            f"fold={fold_id} — {len(overlap)} (class_idx, source_sightline_idx) "
            f"pairs leaked. Sample: {list(overlap)[:5]}",
            file=sys.stderr,
        )
        sys.exit(7)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def _run_main_sweep(
    flux_per_class: List[np.ndarray],
    delta_v: float,
    regimes: List[str],
    m_values: List[int],
    seeds: List[int],
    cv_partial_path: Path,
    run_ablation_in_main: bool = False,
) -> None:
    """Outer loop: (regime, M, seed, fold) — append a row to cv_partial.csv per tuple."""
    completed = _load_completed_tuples(cv_partial_path)
    if completed:
        print(f"[restart] resuming; {len(completed)} (regime, M, seed, fold) tuples already complete")

    for regime in regimes:
        # Precompute per-sightline P(k) per class for this regime (expensive at M=1).
        print(f"\n[regime={regime}] computing per-sightline P(k) per class...")
        t0 = time.time()
        pk_per_class, k_centers = _compute_pk_per_sightline(flux_per_class, regime, delta_v)
        print(f"  ... done in {time.time()-t0:.1f}s; k_centers shape = {k_centers.shape}")

        for M in m_values:
            for seed in seeds:
                # Build stack memberships once per (M, seed); reuse across folds.
                Z, y, membership = _stack_pk_within_class(pk_per_class, M=M, seed=seed)
                mean_flux_per_group = _mean_flux_per_group(flux_per_class, membership)

                skf = StratifiedKFold(
                    n_splits=N_FOLDS, shuffle=True, random_state=seed
                )
                splits = list(skf.split(np.arange(len(y)), y))

                for fold_id, (train_grp, test_grp) in enumerate(splits, start=1):
                    key = (regime, M, seed, fold_id)
                    if key in completed:
                        continue

                    _assert_no_fold_leakage(
                        train_grp, test_grp, membership,
                        regime=regime, M=M, seed=seed, fold_id=fold_id,
                    )

                    t_start = time.time()
                    result = _compute_one_fold(
                        Z=Z, y=y,
                        mean_flux_per_group=mean_flux_per_group,
                        train_grp=train_grp, test_grp=test_grp,
                        k_centers=k_centers, seed=seed,
                        run_ablation=run_ablation_in_main,
                    )
                    runtime_sec = time.time() - t_start

                    row = {
                        "regime": regime, "M": M, "seed": seed, "fold": fold_id,
                        "runtime_sec": runtime_sec,
                        **result,
                    }
                    _append_cv_partial(cv_partial_path, row)
                    print(
                        f"  [{regime} M={M} seed={seed} fold={fold_id}] "
                        f"bal_acc={result['balanced_acc']:.4f} "
                        f"midk_frac={result['mid_k_frac']:.3f} "
                        f"(t={runtime_sec:.1f}s)"
                    )


# ---------------------------------------------------------------------------
# G2 permutation null
# ---------------------------------------------------------------------------
def _run_g2_permutation_null(
    flux_per_class: List[np.ndarray],
    delta_v: float,
    output_path: Path,
    n_permutations: int = N_PERMUTATIONS,
) -> None:
    """Permutation null for G2 per [D-18].

    Restricted to G2_REGIME (per_sightline) + G2_M (=64) + G2_CLASSES ({1,2,3}).
    Stack memberships are computed once with seed=42 and held fixed; the
    permutation acts on the labels across post-stacking groups. 5-fold CV per
    permutation; per-permutation balanced_acc is the 5-fold MEAN.
    """
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if len(existing) >= n_permutations:
            print(f"[G2 null] {output_path} already has {len(existing)} rows; skipping")
            return
        already_done = len(existing)
    else:
        already_done = 0

    print(f"\n[G2 null] permuting labels {n_permutations} times "
          f"(starting at permutation {already_done+1}); regime={G2_REGIME} "
          f"M={G2_M} classes={G2_CLASSES}")

    # Compute P(k), stack, restrict to {1,2,3}.
    pk_per_class, k_centers = _compute_pk_per_sightline(
        flux_per_class, G2_REGIME, delta_v
    )
    Z_full, y_full, _ = _stack_pk_within_class(pk_per_class, M=G2_M, seed=42)
    mask3 = np.isin(y_full, np.array(G2_CLASSES))
    Z = Z_full[mask3]
    y = y_full[mask3]

    # Append-on-restart contract: open with 'a' and write header iff fresh.
    write_header = not output_path.exists()
    f = output_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=["perm_id", "balanced_acc"])
    if write_header:
        writer.writeheader()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    base_splits = list(skf.split(np.arange(len(y)), y))

    rng = np.random.default_rng(seed=42)
    # Burn through the already-completed permutations to keep RNG bit-aligned.
    for _ in range(already_done):
        _ = rng.permutation(len(y))

    for p in range(already_done, n_permutations):
        y_perm = y[rng.permutation(len(y))]
        fold_accs: List[float] = []
        for train_grp, test_grp in base_splits:
            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
            )
            rf.fit(Z[train_grp], y_perm[train_grp])
            y_pred = rf.predict(Z[test_grp])
            fold_accs.append(float(balanced_accuracy_score(y_perm[test_grp], y_pred)))
        bal_acc_mean = float(np.mean(fold_accs))
        writer.writerow({"perm_id": p, "balanced_acc": bal_acc_mean})
        f.flush()
        if (p + 1) % 50 == 0:
            print(f"  [G2 null] {p+1}/{n_permutations} done; latest mean bal_acc = {bal_acc_mean:.4f}")
    f.close()


# ---------------------------------------------------------------------------
# Aggregation: summary + G5 routing + G6 ablation summary
# ---------------------------------------------------------------------------
def _aggregate_summary(cv_partial_path: Path, output_path: Path) -> None:
    """One row per (regime, M): median + 16/84-percentile of balanced_acc.

    Per [D-09]'s bar-coarsening: lower-CI edge is the 16th percentile.
    """
    df = pd.read_csv(cv_partial_path)
    rows: List[Dict[str, Any]] = []
    for (regime, M), g in df.groupby(["regime", "M"]):
        accs = g["balanced_acc"].to_numpy()
        rows.append({
            "regime": regime, "M": int(M),
            "n_rows": len(accs),
            "balanced_acc_median": float(np.median(accs)),
            "balanced_acc_p16": float(np.percentile(accs, 16)),
            "balanced_acc_p84": float(np.percentile(accs, 84)),
            "bal_acc_3class_median": float(np.median(g["balanced_acc_3class"])),
            "bal_acc_c2_vs_c3_median": float(np.median(g["bal_acc_c2_vs_c3"])),
            "mid_k_frac_median": float(np.median(g["mid_k_frac"])),
        })
    pd.DataFrame(rows).sort_values(["regime", "M"]).to_csv(output_path, index=False)
    print(f"[summary] wrote {output_path}")


def _compute_g5_routing(cv_partial_path: Path, output_path: Path) -> Dict[str, Any]:
    """Per-class median |r| across (seeds x folds x M-sweep); C7 routing.

    Returns a verdict dict for downstream gate summary.
    """
    df = pd.read_csv(cv_partial_path)
    medians: Dict[int, float] = {}
    for c in (1, 2, 3, 4):
        col = f"abs_r_c{c}"
        medians[c] = float(np.nanmedian(df[col]))

    rows: List[Dict[str, Any]] = []
    physics_violation = any(medians[c] >= G5_R_THRESHOLD for c in (1, 2, 3))
    c4_violation = medians[4] >= G5_R_THRESHOLD
    c4_only_violator_global = (c4_violation and not physics_violation)

    for c in (1, 2, 3, 4):
        gate_pass = medians[c] < G5_R_THRESHOLD
        c4_only_violator = (c == 4 and c4_only_violator_global)
        rows.append({
            "class": c, "median_abs_r": medians[c],
            "gate_pass": gate_pass,
            "c4_only_violator": c4_only_violator,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[G5] wrote {output_path}; median |r| per class = {medians}")
    return {
        "medians": medians,
        "physics_violation": physics_violation,
        "c4_only_violator": c4_only_violator_global,
        "g5_pass": (not physics_violation) and (not c4_violation),
    }


def _compute_g6_ablation_summary(cv_partial_path: Path, output_path: Path) -> Dict[str, Any]:
    """G6 ablation Δ + max(5pp, 2 * empirical_SD) threshold per [D-23] C1.

    Empirical SD computed from the 4-class balanced_acc ensemble at M=64,
    per_sightline regime (10 seeds x 5 folds = 50 entries).
    """
    df = pd.read_csv(cv_partial_path)
    repr_mask = (
        (df["regime"] == G6_REPRESENTATIVE_REGIME)
        & (df["M"] == G6_REPRESENTATIVE_M)
    )
    repr_df = df[repr_mask]
    if len(repr_df) == 0:
        print(f"[G6] no rows at regime={G6_REPRESENTATIVE_REGIME} M={G6_REPRESENTATIVE_M}; skipping")
        return {"g6_pass": False, "reason": "no data"}

    empirical_sd = float(np.std(repr_df["balanced_acc"], ddof=1))
    threshold = max(G6_BIN_FLOOR_PP, 2.0 * empirical_sd)
    observed_drop = float(np.nanmedian(repr_df["ablation_acc_drop"]))
    g6_pass = observed_drop >= threshold

    out_rows = [{
        "representative_regime": G6_REPRESENTATIVE_REGIME,
        "representative_M": G6_REPRESENTATIVE_M,
        "empirical_SD": empirical_sd,
        "threshold_pp": threshold,
        "threshold_pp_floor": G6_BIN_FLOOR_PP,
        "threshold_pp_2sigma": 2.0 * empirical_sd,
        "observed_ablation_drop_median": observed_drop,
        "g6_pass": g6_pass,
    }]
    pd.DataFrame(out_rows).to_csv(output_path, index=False)
    print(f"[G6] wrote {output_path}; observed Δ={observed_drop:.4f} threshold={threshold:.4f} PASS={g6_pass}")
    return {"g6_pass": g6_pass, "observed_drop": observed_drop, "threshold": threshold,
            "empirical_sd": empirical_sd}


def _compute_g1_g4_summary(
    cv_partial_path: Path, g2_null_path: Path, output_path: Path
) -> Dict[str, Any]:
    """Aggregate G1, G2, G3, G4 gate verdicts (G5 + G6 reported separately)."""
    df = pd.read_csv(cv_partial_path)

    # G1: 4-class CV balanced_acc lower CI (16th pct) >= 0.65 at the best M.
    g1_pass_per_M: List[Dict[str, Any]] = []
    for (regime, M), g in df.groupby(["regime", "M"]):
        p16 = float(np.percentile(g["balanced_acc"], 16))
        g1_pass_per_M.append({
            "regime": regime, "M": int(M),
            "balanced_acc_p16": p16,
            "g1_pass": p16 >= G1_LOWER_CI_THRESHOLD,
        })
    g1_pass_any = any(r["g1_pass"] for r in g1_pass_per_M)

    # G2: observed bal_acc_3class at G2_REGIME/G2_M above 99th percentile of null.
    g2_pass = False
    g2_observed_median: float = float("nan")
    g2_null_p99: float = float("nan")
    if g2_null_path.exists():
        null_df = pd.read_csv(g2_null_path)
        if len(null_df) > 0:
            g2_null_p99 = float(np.percentile(null_df["balanced_acc"], G2_PERCENTILE))
            obs_mask = (df["regime"] == G2_REGIME) & (df["M"] == G2_M)
            if obs_mask.any():
                g2_observed_median = float(np.median(df.loc[obs_mask, "balanced_acc_3class"]))
                g2_pass = g2_observed_median > g2_null_p99

    # G3: mid_k_frac median >= 0.50 (per [D-09]); aggregated across (seeds, folds, M, regime).
    mk_median = float(np.nanmedian(df["mid_k_frac"]))
    g3_pass = mk_median >= G3_MEDIAN_THRESHOLD

    # G4: C2-vs-C3 binary balanced_acc median >= 0.60.
    g4_median = float(np.nanmedian(df["bal_acc_c2_vs_c3"]))
    g4_pass = g4_median >= G4_MEDIAN_THRESHOLD

    out = {
        "g1_pass_any_regime_M": g1_pass_any,
        "g1_per_regime_M": g1_pass_per_M,
        "g2_observed_3class_median": g2_observed_median,
        "g2_null_p99": g2_null_p99,
        "g2_pass": g2_pass,
        "g3_mid_k_frac_median": mk_median,
        "g3_pass": g3_pass,
        "g4_bal_acc_c2_vs_c3_median": g4_median,
        "g4_pass": g4_pass,
    }
    # Flatten g1 detail rows + summary into one CSV.
    rows: List[Dict[str, Any]] = []
    for r in g1_pass_per_M:
        rows.append({
            "gate": "G1", "regime": r["regime"], "M": r["M"],
            "metric": "balanced_acc_p16", "value": r["balanced_acc_p16"],
            "pass": r["g1_pass"],
        })
    rows.append({"gate": "G2", "regime": G2_REGIME, "M": G2_M,
                 "metric": "obs_median_3class_vs_null_p99",
                 "value": g2_observed_median, "pass": g2_pass})
    rows.append({"gate": "G2_null_p99", "regime": "", "M": "",
                 "metric": "null_p99", "value": g2_null_p99, "pass": ""})
    rows.append({"gate": "G3", "regime": "all", "M": "all",
                 "metric": "mid_k_frac_median", "value": mk_median, "pass": g3_pass})
    rows.append({"gate": "G4", "regime": "all", "M": "all",
                 "metric": "bal_acc_c2_vs_c3_median", "value": g4_median, "pass": g4_pass})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[G1-G4] wrote {output_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cv_partial = results_dir / "cv_partial.csv"
    g2_null = results_dir / "g2_permutation_null.csv"
    summary_path = results_dir / "summary.csv"
    g5_path = results_dir / "g5_routing.csv"
    g6_path = results_dir / "g6_ablation.csv"
    g1_g4_path = results_dir / "g1_g4_summary.csv"

    print(f"[stage1] results dir = {results_dir.resolve()}")
    print(f"[stage1] smoke = {args.smoke}")

    print("[stage1] loading flux per class...")
    flux_per_class = _load_flux_per_class()
    for i, f in enumerate(flux_per_class, start=1):
        print(f"   class {i}: shape={f.shape} dtype={f.dtype} "
              f"min={f.min():.3e} max={f.max():.6f} mean={f.mean():.6f}")

    print("[stage1] asserting uniform velocity axis across classes...")
    delta_v = assert_velocity_axes_uniform()
    print(f"   delta_v = {delta_v} km/s/pixel")

    if args.smoke:
        regimes = ["per_sightline"]
        m_values = [4]
        seeds = [42]
        run_ablation_in_main = True
        skip_perm = True
    else:
        regimes = REGIMES
        m_values = M_VALUES
        seeds = SEEDS
        # Run ablation inline at the representative slice (per_sightline, M=64);
        # the aggregator filters down. We compute ablation at EVERY (regime, M,
        # seed, fold) to keep restart logic simple — it's a single extra RF
        # fit per tuple, comparable to the main fit cost.
        run_ablation_in_main = not args.skip_ablation
        skip_perm = args.skip_permutation_null

    t_overall = time.time()

    print(f"\n[stage1] MAIN SWEEP: regimes={regimes} Ms={m_values} seeds={seeds}")
    _run_main_sweep(
        flux_per_class=flux_per_class, delta_v=delta_v,
        regimes=regimes, m_values=m_values, seeds=seeds,
        cv_partial_path=cv_partial,
        run_ablation_in_main=run_ablation_in_main,
    )

    if not skip_perm:
        _run_g2_permutation_null(
            flux_per_class=flux_per_class, delta_v=delta_v,
            output_path=g2_null,
            n_permutations=N_PERMUTATIONS,
        )

    # Aggregations.
    print("\n[stage1] aggregating gate verdicts...")
    _aggregate_summary(cv_partial, summary_path)
    g5 = _compute_g5_routing(cv_partial, g5_path)
    g6 = _compute_g6_ablation_summary(cv_partial, g6_path)
    g14 = _compute_g1_g4_summary(cv_partial, g2_null, g1_g4_path)

    # Headline raw-verdict print (NOT band-routed — that's STAGE1_SPEC.md's job).
    print()
    print("=" * 70)
    print("STAGE 1 RAW GATE VERDICTS (band routing per STAGE1_SPEC.md PI-APPROVE)")
    print("=" * 70)
    print(f"  G1 (4-class CV lower-CI >= {G1_LOWER_CI_THRESHOLD}): "
          f"PASS at any (regime, M)? {g14['g1_pass_any_regime_M']}")
    print(f"  G2 (3-class obs > null p99): {g14['g2_pass']} "
          f"(obs_median={g14['g2_observed_3class_median']:.4f} null_p99={g14['g2_null_p99']:.4f})")
    print(f"  G3 (mid_k_frac median >= {G3_MEDIAN_THRESHOLD}): "
          f"{g14['g3_pass']} (median={g14['g3_mid_k_frac_median']:.4f})")
    print(f"  G4 (C2-vs-C3 bal_acc median >= {G4_MEDIAN_THRESHOLD}): "
          f"{g14['g4_pass']} (median={g14['g4_bal_acc_c2_vs_c3_median']:.4f})")
    print(f"  G5 (all-class |r| < {G5_R_THRESHOLD}): {g5['g5_pass']} "
          f"(C4_only_violator={g5['c4_only_violator']} "
          f"physics_violation={g5['physics_violation']})")
    print(f"  G5 medians: {g5['medians']}")
    print(f"  G6 (ablation Δ >= max(5pp, 2σ)): {g6['g6_pass']}")
    print("=" * 70)
    print(f"\n[stage1] total wall time: {(time.time()-t_overall)/60:.1f} min")
    print(f"[stage1] artifacts under: {results_dir.resolve()}")
    print("\n[stage1] BAND ROUTING DEFERRED to STAGE1_SPEC.md (PI APPROVE per [D-23] C5).")


if __name__ == "__main__":
    main()
