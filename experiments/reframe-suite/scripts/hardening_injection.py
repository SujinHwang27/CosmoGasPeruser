"""Item (iii): injection-recovery sensitivity curve.

Binding spec: experiments/reframe-suite/HARDENING_SPEC.md §4 (+ §5 H4).

Defense-panel attack 5: "methodology eliminated" (the claim that the pipeline
*could* have detected a C2/C3 effect if one existed) is a sensitivity fallacy —
detecting the large C4 effect does not bound sensitivity to a small one, and no
injection-recovery test exists at the C2/C3 effect scale. This script calibrates
the pipeline's minimum detectable P_F(k)-shape effect δ_min(M), converting the
vacated verb into a quantitative sensitivity statement.

Design (spec §4):
  Template  ΔP(k) = mean_C4(P(k)) − mean_C2(P(k)), per k-bin, per_sightline
            regime, full sample. (Template uses C4 data; the classifier never
            sees C4 — no leakage.)
  Surrogate C2' = per-sightline P(k) + α·ΔP(k), injected BEFORE stacking
            (preserves the variance-vs-M structure), then standard stacking.
  Classifier: C2 vs C2', identical estimator/CV as item (i).
  Disjoint arms: the 16384 C2 sightlines are split in HALF per seed; arm A
            (control, label 0) and arm B (injected, label 1) draw from disjoint
            source sightlines, so the classifier cannot exploit identical
            underlying spectra — only the injected α·ΔP(k) shape difference.

  α = {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}; M = {64, 256}; 10 seeds × 5 folds.
  Calibration floor: α=0 must return balanced_acc within 0.50 ± 0.05 (H4 valid).
  δ_min(M) = smallest α with balanced_acc p16 ≥ 0.60.
  β = ⟨ΔP_{C3−C2}, ΔP_{C4−C2}⟩ / ‖ΔP_{C4−C2}‖² (projection of the real C3−C2
      difference onto the injection template).

SCOPE CAVEAT (pre-committed, spec §4, [D-37]): injection is in P(k) space, not
flux space — it cannot capture flux-space nonlinearity or phase structure. The
claim is strictly "pipeline sensitivity to P_F(k)-shape perturbations of
amplitude α·ΔP_{C4−C2}", never "sensitivity to feedback physics of strength α".

Invoke from repo root:
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/hardening_injection.py --smoke
    PYTHONPATH=. uv run python experiments/reframe-suite/scripts/hardening_injection.py

Outputs to results/reframe_suite/hardening/ (+ figs/).
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

from run_stage1 import _compute_pk_per_sightline, _load_flux_per_class  # noqa: E402
from src.core.data import (  # noqa: E402
    assert_velocity_axes_uniform,
    group_indices_within_class,
)


RESULTS_DIR: Path = Path("results/reframe_suite/hardening")
REGIME: str = "per_sightline"
ALPHAS: List[float] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
M_VALUES: List[int] = [64, 256]
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_FOLDS: int = 5
RF_N_ESTIMATORS: int = 300
DELTA_MIN_THRESHOLD: float = 0.60  # δ_min = smallest α with p16 ≥ 0.60

CV_COLUMNS: List[str] = [
    "M", "alpha", "seed", "fold",
    "n_train_groups", "n_test_groups", "balanced_acc", "runtime_sec",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Item (iii) injection-recovery sensitivity.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke: alpha in {0,0.5}, M=64, seed=42.")
    p.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    return p.parse_args()


def _stack_half(
    pk_half: np.ndarray, M: int, seed: int
) -> np.ndarray:
    """Average per-sightline P(k) of one disjoint arm into M-sized groups.

    Groups partition the half's local index space via the SAME
    `group_indices_within_class` used by Stage-1 stacking.

    Returns Z : (n_half // M, n_kbins_eff) float64.
    """
    n = pk_half.shape[0]
    groups = group_indices_within_class(n, M=M, seed=seed)
    Z = np.stack([pk_half[np.asarray(g, dtype=np.int64)].mean(axis=0)
                  for g in groups], axis=0)  # shape: (n_groups, n_kbins_eff)
    return Z


def _compute_templates(
    pk_per_class: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """ΔP_{C4−C2}, ΔP_{C3−C2}, and β.

    pk_per_class is 0-indexed: [C1, C2, C3, C4].
    """
    mean_c2 = pk_per_class[1].mean(axis=0)  # shape: (n_kbins_eff,)
    mean_c3 = pk_per_class[2].mean(axis=0)
    mean_c4 = pk_per_class[3].mean(axis=0)
    dP_c4_c2 = mean_c4 - mean_c2
    dP_c3_c2 = mean_c3 - mean_c2
    denom = float(np.dot(dP_c4_c2, dP_c4_c2))
    beta = float(np.dot(dP_c3_c2, dP_c4_c2) / denom) if denom > 0 else float("nan")
    return dP_c4_c2, dP_c3_c2, beta


def _load_completed(cv_path: Path) -> Set[Tuple[int, float, int, int]]:
    done: Set[Tuple[int, float, int, int]] = set()
    if not cv_path.exists():
        return done
    df = pd.read_csv(cv_path)
    for _, r in df.iterrows():
        done.add((int(r["M"]), float(r["alpha"]), int(r["seed"]), int(r["fold"])))
    return done


def _append_row(cv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not cv_path.exists()
    with cv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CV_COLUMNS})


def _aggregate(cv_path: Path, summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cv_path)
    rows: List[Dict[str, Any]] = []
    for (M, alpha), g in df.groupby(["M", "alpha"]):
        accs = g["balanced_acc"].to_numpy()
        rows.append({
            "M": int(M), "alpha": float(alpha), "n": len(accs),
            "median": float(np.median(accs)),
            "p16": float(np.percentile(accs, 16)),
            "p84": float(np.percentile(accs, 84)),
        })
    out = pd.DataFrame(rows).sort_values(["M", "alpha"])
    out.to_csv(summary_path, index=False)
    print(f"[summary] wrote {summary_path}")
    return out


def _delta_min(summary: pd.DataFrame, M: int) -> float:
    """Smallest α with p16 ≥ DELTA_MIN_THRESHOLD at this M (nan if none)."""
    g = summary[summary["M"] == M].sort_values("alpha")
    hit = g[g["p16"] >= DELTA_MIN_THRESHOLD]
    return float(hit["alpha"].iloc[0]) if len(hit) else float("nan")


def _make_figure(summary: pd.DataFrame, beta: float, fig_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[fig] matplotlib unavailable ({e}); skipping figure")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for M in sorted(summary["M"].unique()):
        g = summary[summary["M"] == M].sort_values("alpha")
        ax.plot(g["alpha"], g["median"], marker="o", label=f"M={int(M)} median")
        ax.fill_between(g["alpha"], g["p16"], g["p84"], alpha=0.15)
    ax.axhline(0.60, ls="--", c="grey", lw=1, label="δ_min threshold (p16≥0.60)")
    ax.axhline(0.50, ls=":", c="grey", lw=1, label="chance")
    for M in sorted(summary["M"].unique()):
        dm = _delta_min(summary, int(M))
        if np.isfinite(dm):
            ax.axvline(dm, ls="-", lw=0.8, alpha=0.4)
    ax.set_xlabel(r"injection amplitude $\alpha$ (units of $\Delta P_{C4-C2}$)")
    ax.set_ylabel("binary balanced accuracy (C2 vs C2′)")
    ax.set_title(
        "Item (iii) injection-recovery sensitivity\n"
        rf"$\beta$(C3−C2 onto C4−C2 template) = {beta:.3f}  |  "
        "P(k)-shape perturbations only (not flux-space)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    print(f"[fig] wrote {fig_path}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    cv_path = out_dir / "injection_recovery_cv.csv"
    summary_path = out_dir / "injection_recovery_summary.csv"
    templates_path = out_dir / "injection_templates.csv"
    fig_path = out_dir / "figs" / "injection_sensitivity_curve.png"

    print(f"[item-iii] results dir = {out_dir.resolve()}  smoke={args.smoke}")
    flux_per_class = _load_flux_per_class()
    delta_v = assert_velocity_axes_uniform()

    print(f"[item-iii] computing per-sightline P(k) (regime={REGIME})...")
    pk_per_class, k_centers = _compute_pk_per_sightline(flux_per_class, REGIME, delta_v)
    dP_c4_c2, dP_c3_c2, beta = _compute_templates(pk_per_class)
    print(f"[item-iii] β(C3−C2 onto C4−C2) = {beta:.4f}")

    # Emit templates immediately (cheap, useful even if the sweep is interrupted).
    pd.DataFrame({
        "k_s_per_km": k_centers,
        "deltaP_C4_C2": dP_c4_c2,
        "deltaP_C3_C2": dP_c3_c2,
        "beta": [beta] * len(k_centers),
    }).to_csv(templates_path, index=False)
    print(f"[templates] wrote {templates_path}")

    pk_c2 = pk_per_class[1]  # shape: (16384, n_kbins_eff)
    n_c2 = pk_c2.shape[0]

    if args.smoke:
        alphas = [0.0, 0.5]
        m_values = [64]
        seeds = [42]
    else:
        alphas = ALPHAS
        m_values = M_VALUES
        seeds = SEEDS

    completed = _load_completed(cv_path)
    if completed:
        print(f"[restart] {len(completed)} (M,alpha,seed,fold) tuples done")

    t_all = time.time()
    for seed in seeds:
        # Disjoint half-split of C2 source sightlines (per seed).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_c2)
        half = n_c2 // 2
        idx_a = perm[:half]   # control arm (label 0)
        idx_b = perm[half:2 * half]  # injected arm (label 1)
        pk_a = pk_c2[idx_a]   # shape: (8192, n_kbins_eff)
        pk_b_base = pk_c2[idx_b]
        for M in m_values:
            for alpha in alphas:
                # Inject BEFORE stacking on arm B only.
                pk_b = pk_b_base + alpha * dP_c4_c2[None, :]
                Z_a = _stack_half(pk_a, M=M, seed=seed)
                Z_b = _stack_half(pk_b, M=M, seed=seed)
                Z = np.vstack([Z_a, Z_b])
                y = np.concatenate([
                    np.zeros(Z_a.shape[0], dtype=np.int64),
                    np.ones(Z_b.shape[0], dtype=np.int64),
                ])
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                for fold_id, (tr, te) in enumerate(
                    skf.split(np.arange(len(y)), y), start=1
                ):
                    key = (M, float(alpha), seed, fold_id)
                    if key in completed:
                        continue
                    t0 = time.time()
                    rf = RandomForestClassifier(
                        n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1
                    )
                    rf.fit(Z[tr], y[tr])
                    bal = float(balanced_accuracy_score(y[te], rf.predict(Z[te])))
                    _append_row(cv_path, {
                        "M": M, "alpha": alpha, "seed": seed, "fold": fold_id,
                        "n_train_groups": int(len(tr)),
                        "n_test_groups": int(len(te)),
                        "balanced_acc": bal, "runtime_sec": time.time() - t0,
                    })
            print(f"  [seed={seed} M={M}] all alphas done")

    print("\n[item-iii] aggregating...")
    summary = _aggregate(cv_path, summary_path)
    _make_figure(summary, beta, fig_path)

    # Report H4 calibration + δ_min.
    alpha0 = summary[summary["alpha"] == 0.0]
    print("\n[H4] calibration floor (alpha=0):")
    for _, r in alpha0.iterrows():
        ok = abs(r["median"] - 0.5) <= 0.05 and r["p16"] <= 0.5 <= r["p84"]
        print(f"   M={int(r['M'])}: median={r['median']:.4f} "
              f"p16={r['p16']:.4f} p84={r['p84']:.4f}  calibrated={ok}")
    for M in m_values:
        print(f"[δ_min] M={M}: {_delta_min(summary, M)}  "
              f"(smallest α with p16 ≥ {DELTA_MIN_THRESHOLD})")
    print(f"[item-iii] β = {beta:.4f}")
    print(f"[item-iii] total wall: {(time.time()-t_all)/60:.1f} min")
    print(f"[item-iii] artifacts under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
