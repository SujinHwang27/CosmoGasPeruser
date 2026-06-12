"""Item (ii): mean-flux scalar baseline (cheapest-decisive control).

Binding spec: experiments/reframe-suite/HARDENING_SPEC.md §3 (+ §5 H3).

Defense-panel attack 4: the reframe-suite lattice p16 ranking matches the
per-class |Δ⟨F⟩| ranking ([D-02]) with Spearman ≈ 0.94 — so the whole lattice
may be mean-flux ordering in disguise. The decisive control (proposed at the
[D-13] panel, item P4, never run): can a classifier given ONLY the stack-mean
flux ⟨F⟩ reproduce the lattice? If P_F(k) does not materially beat this scalar
baseline (H3: Δ = dedicated_PFk_p16 − fbar_p16 < 0.02 per cell → KILL), the
P_F(k)-methodology contribution collapses.

ONE estimator, same protocol as item (i) and Stage-1: RandomForestClassifier(
n_estimators=300, random_state=seed, n_jobs=-1), balanced_accuracy_score,
5-fold StratifiedKFold over post-stacking groups, same (M, seed) stack
memberships (bit-identical, via the shared run_stage1 helpers).

Feature sets:
  PRIMARY  "fbar"      — stack-level ⟨F⟩ only (1-dim), from RAW flux.
  SECONDARY "fbar_var" — [⟨F⟩, var(F)] (2-dim), disclosed, non-gate-bearing.
⟨F⟩ is computed from raw flux F (pre-normalization); the per_sightline δ_F
regime destroys ⟨F⟩ by construction, so the baseline is regime-independent and
recorded with regime label "raw_flux".

Tasks: 6 pairs + C4-vs-rest + the 4-class problem (for completeness).

Invoke from repo root:
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/hardening_fbar_baseline.py --smoke
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/hardening_fbar_baseline.py

Outputs to results/reframe_suite/hardening/.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_DIR = _REPO_ROOT / "experiments" / "pk-feedback-classifier"
if str(_STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE1_DIR))

from run_stage1 import _load_flux_per_class  # noqa: E402
from src.core.data import (  # noqa: E402
    assert_velocity_axes_uniform,
    group_indices_within_class,
)


RESULTS_DIR: Path = Path("results/reframe_suite/hardening")
M_VALUES: List[int] = [1, 4, 16, 32, 64, 128, 256]
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_FOLDS: int = 5
RF_N_ESTIMATORS: int = 300

PAIR_NAMES: Dict[Tuple[int, int], str] = {
    (1, 2): "C1-C2", (1, 3): "C1-C3", (1, 4): "C1-C4",
    (2, 3): "C2-C3", (2, 4): "C2-C4", (3, 4): "C3-C4",
}
C4_VS_REST: str = "C4-rest"
FOUR_CLASS: str = "4class"
FEATURE_SETS: List[str] = ["fbar", "fbar_var"]

CV_COLUMNS: List[str] = [
    "M", "task", "feature_set", "seed", "fold",
    "n_train_groups", "n_test_groups", "balanced_acc", "runtime_sec",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Item (ii) mean-flux scalar baseline.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke: M=64, seed=42, all tasks, both feature sets.")
    p.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    return p.parse_args()


def _stack_scalar_features(
    flux_per_class: List[np.ndarray], M: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]:
    """Build per-group scalar features [⟨F⟩, var(F)] using the SAME groups as
    `_stack_pk_within_class` (same `group_indices_within_class(n, M, seed)`),
    so stack memberships are bit-identical to item (i) and Stage-1.

    Returns
    -------
    X2 : (n_total_groups, 2) float64 — columns [mean_flux, var_flux].
    y  : (n_total_groups,) int64 — class label in {1,2,3,4}.
    membership : list aligned to rows; (class_idx_0based, source_sightline_idx).
    """
    feat_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    membership: List[Tuple[int, np.ndarray]] = []
    for c_idx, flux in enumerate(flux_per_class):
        n = flux.shape[0]
        groups = group_indices_within_class(n, M=M, seed=seed)
        means = np.empty(len(groups), dtype=np.float64)
        varis = np.empty(len(groups), dtype=np.float64)
        for gi, g in enumerate(groups):
            block = flux[np.asarray(g, dtype=np.int64)]  # shape: (M, 2048)
            means[gi] = float(block.mean())
            varis[gi] = float(block.var())
            membership.append((c_idx, np.asarray(g, dtype=np.int64)))
        feat_chunks.append(np.column_stack([means, varis]))  # (n_groups, 2)
        y_chunks.append(np.full(len(groups), c_idx + 1, dtype=np.int64))
    X2 = np.vstack(feat_chunks)  # shape: (n_total_groups, 2)
    y = np.concatenate(y_chunks)
    assert X2.dtype == np.float64 and np.all(np.isfinite(X2))
    return X2, y, membership


def _binary_or_multiclass(
    X2: np.ndarray, y: np.ndarray, task: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce to the requested task's (X, y_task). 4class keeps all labels."""
    if task == FOUR_CLASS:
        return X2, y
    if task == C4_VS_REST:
        return X2, (y == 4).astype(np.int64)
    a, b = (int(task.split("-")[0][1:]), int(task.split("-")[1][1:]))
    mask = np.isin(y, np.array([a, b]))
    return X2[mask], np.where(y[mask] == b, 1, 0).astype(np.int64)


def _select_features(X2: np.ndarray, feature_set: str) -> np.ndarray:
    """fbar -> column 0 only (1-dim); fbar_var -> both columns (2-dim)."""
    if feature_set == "fbar":
        return X2[:, [0]]
    return X2  # fbar_var


def _load_completed(cv_path: Path) -> Set[Tuple[int, str, str, int, int]]:
    done: Set[Tuple[int, str, str, int, int]] = set()
    if not cv_path.exists():
        return done
    df = pd.read_csv(cv_path)
    for _, r in df.iterrows():
        done.add((int(r["M"]), str(r["task"]), str(r["feature_set"]),
                  int(r["seed"]), int(r["fold"])))
    return done


def _append_row(cv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not cv_path.exists()
    with cv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CV_COLUMNS})


def _aggregate(cv_path: Path, summary_path: Path) -> None:
    df = pd.read_csv(cv_path)
    rows: List[Dict[str, Any]] = []
    for (M, task, fs), g in df.groupby(["M", "task", "feature_set"]):
        accs = g["balanced_acc"].to_numpy()
        rows.append({
            "M": int(M), "task": task, "feature_set": fs, "n": len(accs),
            "median": float(np.median(accs)),
            "p16": float(np.percentile(accs, 16)),
            "p84": float(np.percentile(accs, 84)),
        })
    pd.DataFrame(rows).sort_values(["feature_set", "task", "M"]).to_csv(
        summary_path, index=False
    )
    print(f"[summary] wrote {summary_path}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_path = out_dir / "fbar_baseline_cv.csv"
    summary_path = out_dir / "fbar_baseline_summary.csv"

    print(f"[item-ii] results dir = {out_dir.resolve()}  smoke={args.smoke}")
    flux_per_class = _load_flux_per_class()
    for i, f in enumerate(flux_per_class, start=1):
        print(f"   class {i}: shape={f.shape} dtype={f.dtype} mean={f.mean():.6f}")
    # Velocity uniformity not needed for scalar features, but assert for parity.
    _ = assert_velocity_axes_uniform()

    if args.smoke:
        m_values = [64]
        seeds = [42]
    else:
        m_values = M_VALUES
        seeds = SEEDS

    tasks = [PAIR_NAMES[p] for p in PAIR_NAMES] + [C4_VS_REST, FOUR_CLASS]
    completed = _load_completed(cv_path)
    if completed:
        print(f"[restart] {len(completed)} (M,task,feature_set,seed,fold) tuples done")

    t_all = time.time()
    for M in m_values:
        for seed in seeds:
            X2, y, membership = _stack_scalar_features(flux_per_class, M=M, seed=seed)
            for feature_set in FEATURE_SETS:
                for task in tasks:
                    Xt, yt = _binary_or_multiclass(X2, y, task)
                    Xt = _select_features(Xt, feature_set)
                    skf = StratifiedKFold(
                        n_splits=N_FOLDS, shuffle=True, random_state=seed
                    )
                    for fold_id, (tr, te) in enumerate(
                        skf.split(np.arange(len(yt)), yt), start=1
                    ):
                        key = (M, task, feature_set, seed, fold_id)
                        if key in completed:
                            continue
                        t0 = time.time()
                        rf = RandomForestClassifier(
                            n_estimators=RF_N_ESTIMATORS,
                            random_state=seed, n_jobs=-1,
                        )
                        rf.fit(Xt[tr], yt[tr])
                        bal = float(balanced_accuracy_score(
                            yt[te], rf.predict(Xt[te])
                        ))
                        _append_row(cv_path, {
                            "M": M, "task": task, "feature_set": feature_set,
                            "seed": seed, "fold": fold_id,
                            "n_train_groups": int(len(tr)),
                            "n_test_groups": int(len(te)),
                            "balanced_acc": bal, "runtime_sec": time.time() - t0,
                        })
            print(f"  [M={M} seed={seed}] done")

    print("\n[item-ii] aggregating...")
    _aggregate(cv_path, summary_path)
    print(f"[item-ii] total wall: {(time.time()-t_all)/60:.1f} min")
    print(f"[item-ii] artifacts under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
