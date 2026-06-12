"""Item (i): dedicated binary pair classifiers (lattice v2).

Binding spec: experiments/reframe-suite/HARDENING_SPEC.md §2 (+ §5 H1/H2).

Defense-panel attack 1: every reframe-suite lattice number — including the
load-bearing C2-C3 null and the R1 binary {C4 vs rest} headline — was a
row-normalized sub-block of a single 4-class RF confusion matrix
(`phase1_reframes.py::pair_balacc_from_cm`), NOT a calibrated binary
classifier. This script trains a *dedicated* binary RandomForest per pair,
under the identical Stage-1 protocol (stacking, CV, seeds), so the resulting
balanced accuracies become the numbers of record. The C2-C3 M-curve here is
the H2 exclusion-bound measurement.

ONE estimator, ONE number per cell (spec §2): RandomForestClassifier(
n_estimators=300, random_state=seed, n_jobs=-1), balanced_accuracy_score on
held-out folds. No SVM arm, no hyperparameter search — matched to Stage-1.

Tasks: 6 one-vs-one pairs {C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4} + the
binary C4-vs-rest detection task (the R1 re-measurement).

Protocol reuse (NO re-implementation): the per-sightline P(k) transform and the
within-class stacking come verbatim from
experiments/pk-feedback-classifier/run_stage1.py via import, so stack
memberships are bit-identical to Stage-1 for the same (M, seed).

Invoke from repo root:
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/hardening_pair_binary.py --smoke
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/hardening_pair_binary.py

Outputs to results/reframe_suite/hardening/.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# --- Reuse Stage-1 protocol helpers verbatim (import is side-effect-free; the
#     run_stage1 main() is guarded by `if __name__ == "__main__"`). ----------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_DIR = _REPO_ROOT / "experiments" / "pk-feedback-classifier"
if str(_STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE1_DIR))

from run_stage1 import (  # noqa: E402  (path injection required first)
    _compute_pk_per_sightline,
    _load_flux_per_class,
    _stack_pk_within_class,
)
from src.core.data import assert_velocity_axes_uniform  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration (binding per HARDENING_SPEC §2)
# ---------------------------------------------------------------------------
RESULTS_DIR: Path = Path("results/reframe_suite/hardening")

M_VALUES: List[int] = [1, 4, 16, 32, 64, 128, 256]
REGIMES: List[str] = ["per_sightline", "global"]
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10-seed bootstrap
N_FOLDS: int = 5
RF_N_ESTIMATORS: int = 300

# Saturation extension: C2-C3 only, per_sightline only, exploratory (spec §2).
SATURATION_M_VALUES: List[int] = [512, 1024]

# Pair tasks: (label_a, label_b). C4-vs-rest handled as a special task.
PAIRS: List[Tuple[int, int]] = [
    (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
]
PAIR_NAMES: Dict[Tuple[int, int], str] = {
    (1, 2): "C1-C2", (1, 3): "C1-C3", (1, 4): "C1-C4",
    (2, 3): "C2-C3", (2, 4): "C2-C4", (3, 4): "C3-C4",
}
C4_VS_REST: str = "C4-rest"

CV_COLUMNS: List[str] = [
    "regime", "M", "pair", "seed", "fold",
    "n_train_groups", "n_test_groups",
    "balanced_acc", "exploratory_flag", "runtime_sec",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Item (i) dedicated binary pair classifiers.")
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke: M=64 only, seed=42 only, all 7 tasks, both regimes. ~2 min.",
    )
    p.add_argument(
        "--regime", type=str, default=None, choices=REGIMES,
        help="Restrict to a single regime (default: both).",
    )
    p.add_argument(
        "--out-dir", type=str, default=str(RESULTS_DIR),
        help="Results dir (default: results/reframe_suite/hardening).",
    )
    p.add_argument(
        "--no-saturation", action="store_true",
        help="Skip the M in {512,1024} C2-C3 saturation extension.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Binary subset + fold-leakage assert
# ---------------------------------------------------------------------------
def _binary_subset(
    Z: np.ndarray,
    y: np.ndarray,
    membership: List[Tuple[int, np.ndarray]],
    task: str,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]:
    """Reduce the 4-class stacked matrix to a binary task.

    For a pair "Ca-Cb": keep groups with label in {a, b}; binary label 0/1.
    For C4-vs-rest: label 1 = C4, label 0 = {C1,C2,C3}.

    Returns (Z_bin, y_bin in {0,1}, membership_bin aligned to Z_bin rows).
    """
    if task == C4_VS_REST:
        mask = np.ones(len(y), dtype=bool)
        y_bin = (y == 4).astype(np.int64)  # 1 = C4, 0 = {C1,C2,C3}; full length
    else:
        a, b = (int(task.split("-")[0][1:]), int(task.split("-")[1][1:]))
        mask = np.isin(y, np.array([a, b]))
        y_bin = np.where(y[mask] == b, 1, 0).astype(np.int64)  # 1 = higher label
    Z_bin = Z[mask]
    membership_bin = [membership[i] for i in np.nonzero(mask)[0]]
    # shape: (n_groups_bin, n_kbins_eff)
    assert Z_bin.shape[0] == len(y_bin) == len(membership_bin)
    return Z_bin, y_bin, membership_bin


def _assert_no_fold_leakage(
    train_grp: np.ndarray,
    test_grp: np.ndarray,
    membership: List[Tuple[int, np.ndarray]],
    tag: str,
) -> None:
    """Materialize (class_idx, source_sightline) sets; assert train/test disjoint.

    Exits code 7 on leakage (matches Stage-1 convention, distinct from PCV
    codes 2-6 used by the sbatch wrapper).
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
        print(f"[FATAL] fold-leakage: {tag} — {len(overlap)} pairs leaked. "
              f"Sample: {list(overlap)[:5]}", file=sys.stderr)
        sys.exit(7)


# ---------------------------------------------------------------------------
# Restart / checkpointing
# ---------------------------------------------------------------------------
def _load_completed(cv_path: Path) -> Set[Tuple[str, int, str, int, int]]:
    """Completed (regime, M, pair, seed, fold) keys for idempotent resume."""
    done: Set[Tuple[str, int, str, int, int]] = set()
    if not cv_path.exists():
        return done
    df = pd.read_csv(cv_path)
    for _, r in df.iterrows():
        done.add((str(r["regime"]), int(r["M"]), str(r["pair"]),
                  int(r["seed"]), int(r["fold"])))
    return done


def _append_row(cv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not cv_path.exists()
    with cv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CV_COLUMNS})


# ---------------------------------------------------------------------------
# One binary task at one (regime, M, seed)
# ---------------------------------------------------------------------------
def _run_task_cv(
    Z: np.ndarray,
    y: np.ndarray,
    membership: List[Tuple[int, np.ndarray]],
    task: str,
    regime: str,
    M: int,
    seed: int,
    exploratory: int,
    cv_path: Path,
    completed: Set[Tuple[str, int, str, int, int]],
) -> None:
    Z_bin, y_bin, mem_bin = _binary_subset(Z, y, membership, task)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(skf.split(np.arange(len(y_bin)), y_bin))
    for fold_id, (train_grp, test_grp) in enumerate(splits, start=1):
        key = (regime, M, task, seed, fold_id)
        if key in completed:
            continue
        _assert_no_fold_leakage(
            train_grp, test_grp, mem_bin,
            tag=f"{regime} M={M} {task} seed={seed} fold={fold_id}",
        )
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1
        )
        rf.fit(Z_bin[train_grp], y_bin[train_grp])
        bal = float(balanced_accuracy_score(
            y_bin[test_grp], rf.predict(Z_bin[test_grp])
        ))
        _append_row(cv_path, {
            "regime": regime, "M": M, "pair": task, "seed": seed, "fold": fold_id,
            "n_train_groups": int(len(train_grp)),
            "n_test_groups": int(len(test_grp)),
            "balanced_acc": bal, "exploratory_flag": exploratory,
            "runtime_sec": time.time() - t0,
        })
        print(f"  [{regime} M={M} {task} seed={seed} fold={fold_id}] "
              f"bal_acc={bal:.4f}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(cv_path: Path, summary_path: Path) -> None:
    df = pd.read_csv(cv_path)
    rows: List[Dict[str, Any]] = []
    for (regime, M, pair), g in df.groupby(["regime", "M", "pair"]):
        accs = g["balanced_acc"].to_numpy()
        rows.append({
            "regime": regime, "M": int(M), "pair": pair, "n": len(accs),
            "median": float(np.median(accs)),
            "p16": float(np.percentile(accs, 16)),
            "p84": float(np.percentile(accs, 84)),
            "exploratory_flag": int(g["exploratory_flag"].max()),
        })
    pd.DataFrame(rows).sort_values(["regime", "pair", "M"]).to_csv(
        summary_path, index=False
    )
    print(f"[summary] wrote {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_path = out_dir / "pair_binary_cv.csv"
    summary_path = out_dir / "pair_binary_summary.csv"

    print(f"[item-i] results dir = {out_dir.resolve()}  smoke={args.smoke}")
    flux_per_class = _load_flux_per_class()
    for i, f in enumerate(flux_per_class, start=1):
        print(f"   class {i}: shape={f.shape} dtype={f.dtype}")
    delta_v = assert_velocity_axes_uniform()
    print(f"   delta_v = {delta_v} km/s/pixel")

    if args.smoke:
        regimes = [args.regime] if args.regime else ["per_sightline", "global"]
        m_values = [64]
        seeds = [42]
        do_saturation = False
    else:
        regimes = [args.regime] if args.regime else REGIMES
        m_values = M_VALUES
        seeds = SEEDS
        do_saturation = not args.no_saturation

    tasks = [PAIR_NAMES[p] for p in PAIRS] + [C4_VS_REST]
    completed = _load_completed(cv_path)
    if completed:
        print(f"[restart] {len(completed)} (regime,M,pair,seed,fold) tuples done")

    t_all = time.time()
    for regime in regimes:
        print(f"\n[regime={regime}] computing per-sightline P(k)...")
        t0 = time.time()
        pk_per_class, _k = _compute_pk_per_sightline(flux_per_class, regime, delta_v)
        print(f"  ... done in {time.time()-t0:.1f}s")
        for M in m_values:
            for seed in seeds:
                Z, y, membership = _stack_pk_within_class(pk_per_class, M=M, seed=seed)
                for task in tasks:
                    _run_task_cv(
                        Z, y, membership, task, regime, M, seed,
                        exploratory=0, cv_path=cv_path, completed=completed,
                    )

    # Saturation extension: C2-C3 only, per_sightline only, exploratory.
    if do_saturation:
        print("\n[saturation] C2-C3 only, per_sightline, M in {512,1024} (exploratory)...")
        pk_ps, _k = _compute_pk_per_sightline(flux_per_class, "per_sightline", delta_v)
        for M in SATURATION_M_VALUES:
            for seed in seeds:
                Z, y, membership = _stack_pk_within_class(pk_ps, M=M, seed=seed)
                _run_task_cv(
                    Z, y, membership, "C2-C3", "per_sightline", M, seed,
                    exploratory=1, cv_path=cv_path, completed=completed,
                )

    print("\n[item-i] aggregating...")
    _aggregate(cv_path, summary_path)
    print(f"[item-i] total wall: {(time.time()-t_all)/60:.1f} min")
    print(f"[item-i] artifacts under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
