"""
De-risking probe for the pk-feedback-classifier track.

Runs flux -> P_F(k) -> global 4-class RandomForest end-to-end for both
normalization regimes (per_sightline PRIMARY, global SECONDARY confounder
probe) at variants M in {1 (per-sightline), 4, 16, 64} of within-class
stacking, and emits the gating artifacts described in
experiments/pk-feedback-classifier/LEDGER.md sections 1, 5, 6.

NO DVC stage. NO MLflow. Single self-contained CPU script.

Invoke from repo root:
    PYTHONPATH=. uv run python experiments/pk-feedback-classifier/run_probe.py

Outputs to results/pk_feedback_classifier/.

PASS / Ambiguous / NULL verdict printed to stdout per LEDGER section 5
outcome bands. The k-bin importance anti-degeneracy gate ([D-03]) is
encoded as: top-7 of 20 k-bins (upper third by k value) must hold > 50%
of total importance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.data import SignalClusteringData
from src.core.transforms import FluxPowerSpectrum
from src.core.models.rf_classifier import train_rf_global


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RESULTS_DIR = Path("results/pk_feedback_classifier")
FIGS_DIR = RESULTS_DIR / "figs"
N_KBINS = 20
M_VALUES: List[int] = [1, 4, 16, 64]
REGIMES: List[str] = ["per_sightline", "global"]
SEED = 42
BAR_PER_SIGHTLINE = 0.55  # LEDGER section 1 PASS condition
BAR_STACKED_M64 = 0.70    # LEDGER section 1 PASS condition
V02_BASELINE = 0.451
CHANCE = 0.25
HIGH_K_TOP_N = 7  # upper third of 20 bins
HIGH_K_IMPORTANCE_FRACTION = 0.50  # [D-03] anti-degeneracy gate

VEL_PATH_TEMPLATE = "data/preprocessed/Sherwood_z0.3_inf/{cls}/vel.npy"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _resolve_uniform_delta_v(n_classes: int = 4) -> float:
    """Read vel.npy from all 4 classes, assert uniform across classes, return delta_v."""
    deltas: List[float] = []
    for cls in range(1, n_classes + 1):
        vel = np.load(VEL_PATH_TEMPLATE.format(cls=cls))
        diffs = np.diff(vel)
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError(f"vel.npy class {cls} is not uniformly spaced.")
        deltas.append(float(diffs[0]))
    if not np.allclose(deltas, deltas[0], rtol=1e-6, atol=1e-9):
        raise ValueError(f"vel.npy delta_v differs across classes: {deltas}")
    return deltas[0]


def stack_flux_within_class(
    flux_per_class: List[np.ndarray], M: int, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Within-class disjoint random grouping into chunks of size M, returning the
    AVERAGED flux per group (used here as an alternative stacking pathway).

    Note on stacking semantics
    --------------------------
    For a power spectrum the SNR-correct operation is to average P(k) across M
    independent sightlines (P_F(k) variance falls ~1/M). Averaging flux first
    and then taking one P(k) is a different operation — it suppresses high-k
    power through coherent cancellation of small-scale fluctuations across
    independent sightlines and is NOT a faithful population P(k). The probe
    therefore defaults to averaging-P(k) (see stack_pk_within_class below);
    this flux-stacking helper is provided as a sibling for completeness.

    Returns
    -------
    stacked_flux : np.ndarray, shape (n_groups_total, n_pixels), float64
    y_stacked : np.ndarray, shape (n_groups_total,), int (class label 1..4)
    """
    rng = np.random.default_rng(seed)
    stacked_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    for c_idx, flux in enumerate(flux_per_class):
        n = flux.shape[0]
        idx = rng.permutation(n)
        n_groups = n // M
        groups = idx[: n_groups * M].reshape(n_groups, M)
        # Mean across rows in each group.
        group_means = np.stack(
            [flux[g].astype(np.float64).mean(axis=0) for g in groups], axis=0
        )  # shape: (n_groups, n_pixels)
        stacked_chunks.append(group_means)
        label_chunks.append(np.full(n_groups, c_idx + 1, dtype=np.int64))
    return np.vstack(stacked_chunks), np.concatenate(label_chunks)


def stack_pk_within_class(
    flux_per_class: List[np.ndarray],
    M: int,
    regime: str,
    delta_v: float,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, FluxPowerSpectrum]:
    """
    SNR-correct within-class stacking: compute P_F(k) per sightline then average
    across M-sized disjoint random groups within each class.

    Returns the stacked feature matrix Z, the labels y, and the fitted
    transformer (we want its k_bin_centers_ for the importance plot).
    """
    rng = np.random.default_rng(seed)
    # Compute P(k) per sightline for the entire (16384 x 4) corpus once.
    pk_per_class: List[np.ndarray] = []
    transformer = FluxPowerSpectrum(
        norm=regime, n_kbins=N_KBINS, delta_v=delta_v
    )
    first = True
    fitted_transformer: FluxPowerSpectrum = transformer
    for flux in flux_per_class:
        # Each fit_transform call sets k_bin_centers_ etc.; they are identical
        # across classes because delta_v and n_pixels are identical.
        if first:
            pk = transformer.fit_transform(np.asarray(flux, dtype=np.float64))
            fitted_transformer = transformer
            first = False
        else:
            t = FluxPowerSpectrum(norm=regime, n_kbins=N_KBINS, delta_v=delta_v)
            pk = t.fit_transform(np.asarray(flux, dtype=np.float64))
        pk_per_class.append(pk)

    # Disjoint M-grouped averages within each class.
    stacked_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    for c_idx, pk in enumerate(pk_per_class):
        n = pk.shape[0]
        idx = rng.permutation(n)
        n_groups = n // M
        groups = idx[: n_groups * M].reshape(n_groups, M)
        group_means = np.stack([pk[g].mean(axis=0) for g in groups], axis=0)
        stacked_chunks.append(group_means)
        label_chunks.append(np.full(n_groups, c_idx + 1, dtype=np.int64))

    return np.vstack(stacked_chunks), np.concatenate(label_chunks), fitted_transformer


def _high_k_importance_fraction(
    importances: np.ndarray, k_centers: np.ndarray, top_n: int
) -> Tuple[float, np.ndarray]:
    """
    Fraction of total importance held by the top_n highest-k bins, and the
    indices of those bins (by k value, descending).
    """
    if importances.sum() <= 0.0:
        return 0.0, np.array([], dtype=np.int64)
    # Indices of top_n bins by k value (highest k first).
    order = np.argsort(k_centers)[::-1]
    high_k_idx = order[:top_n]
    frac = float(importances[high_k_idx].sum() / importances.sum())
    return frac, high_k_idx


def _save_confusion_csv(
    cm: np.ndarray, class_labels: List[Any], path: Path
) -> None:
    df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_labels],
        columns=[f"pred_{c}" for c in class_labels],
    )
    df.to_csv(path)


def _save_kbin_importance_csv(
    importances: np.ndarray, k_centers: np.ndarray, path: Path
) -> None:
    df = pd.DataFrame(
        {
            "k_bin_idx": np.arange(len(importances)),
            "k_center_s_per_km": k_centers,
            "importance": importances,
        }
    )
    df.to_csv(path, index=False)


def _plot_confusion(cm: np.ndarray, class_labels: List[Any], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_kbin_importance(
    importances: np.ndarray, k_centers: np.ndarray, top_n_high: int,
    title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    # Sort by k for plotting.
    order = np.argsort(k_centers)
    k_sorted = k_centers[order]
    imp_sorted = importances[order]
    ax.bar(np.arange(len(k_sorted)), imp_sorted, color="#3776ab")
    ax.set_xticks(np.arange(len(k_sorted)))
    ax.set_xticklabels([f"{k:.2e}" for k in k_sorted], rotation=90, fontsize=7)
    ax.set_xlabel("k center [s/km] (log-binned)")
    ax.set_ylabel("RF feature importance")
    ax.set_title(title)
    # Mark the high-k region (rightmost top_n bins after sort-by-k).
    cutoff_idx = len(k_sorted) - top_n_high - 0.5
    ax.axvline(cutoff_idx, color="crimson", linestyle="--",
               label=f"high-k cutoff (top {top_n_high})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_acc_vs_M(
    rows: List[Dict[str, Any]], regime: str, path: Path
) -> None:
    sub = [r for r in rows if r["regime"] == regime]
    sub.sort(key=lambda r: r["M"])
    M_vals = [r["M"] for r in sub]
    accs = [r["balanced_acc_test"] for r in sub]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(M_vals, accs, marker="o", color="#3776ab", label="balanced acc (test)")
    ax.axhline(CHANCE, color="gray", linestyle=":", label=f"chance ({CHANCE})")
    ax.axhline(V02_BASELINE, color="goldenrod", linestyle="--",
               label=f"v0.2 baseline ({V02_BASELINE})")
    ax.axhline(BAR_PER_SIGHTLINE, color="forestgreen", linestyle="--",
               label=f"PASS bar per-sightline ({BAR_PER_SIGHTLINE})")
    ax.axhline(BAR_STACKED_M64, color="crimson", linestyle="--",
               label=f"PASS bar stacked M=64 ({BAR_STACKED_M64})")
    ax.set_xscale("log")
    ax.set_xlabel("Stack depth M")
    ax.set_ylabel("Balanced accuracy (test)")
    ax.set_title(f"Acc vs M — regime: {regime}")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading flux per class via SignalClusteringData...")
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()
    # Materialize from mmap and force float64.
    flux_per_class = [np.asarray(f, dtype=np.float64) for f in flux_per_class]
    for i, f in enumerate(flux_per_class, start=1):
        print(f"   class {i}: shape={f.shape} dtype={f.dtype} "
              f"min={f.min():.3e} max={f.max():.6f} mean={f.mean():.6f}")

    print("[2/4] Resolving delta_v from vel.npy (assert uniform across classes)...")
    delta_v = _resolve_uniform_delta_v()
    print(f"   delta_v = {delta_v} km/s/pixel")

    # Concatenated (65536, 2048) view used only for M=1.
    X_flux = np.vstack(flux_per_class)
    y_full = np.concatenate(
        [np.full(f.shape[0], i + 1, dtype=np.int64) for i, f in enumerate(flux_per_class)]
    )
    print(f"   X_flux: {X_flux.shape} {X_flux.dtype}; y: {y_full.shape}")

    summary_rows: List[Dict[str, Any]] = []
    # Cache fitted transformers' k_bin_centers_ per regime for the importance plot.
    k_centers_per_regime: Dict[str, np.ndarray] = {}

    print("[3/4] Running RF for each (regime, variant, M)...")
    for regime in REGIMES:
        for M in M_VALUES:
            if M == 1:
                variant = "per_sightline"
                transformer = FluxPowerSpectrum(
                    norm=regime, n_kbins=N_KBINS, delta_v=delta_v
                )
                Z = transformer.fit_transform(X_flux)
                y = y_full
                k_centers = transformer.k_bin_centers_
            else:
                variant = "stacked_pk"
                Z, y, transformer = stack_pk_within_class(
                    flux_per_class, M=M, regime=regime, delta_v=delta_v, seed=SEED
                )
                k_centers = transformer.k_bin_centers_

            k_centers_per_regime[regime] = k_centers

            assert Z.dtype == np.float64
            assert np.all(np.isfinite(Z)), "Z contains NaN/Inf"
            print(f"   regime={regime} variant={variant} M={M} "
                  f"Z={Z.shape} y={y.shape}")

            res = train_rf_global(Z, y, seed=SEED, n_estimators=300,
                                  return_importances=True)
            bal = res["balanced_acc_test"]
            cm = res["confusion_test"]
            imp = res["feature_importances"]
            class_labels = res["class_labels"]

            # Anti-degeneracy gate.
            high_k_frac, _ = _high_k_importance_fraction(
                imp, k_centers, HIGH_K_TOP_N
            )

            print(f"     balanced_acc_test = {bal:.4f}  "
                  f"high_k_frac(top{HIGH_K_TOP_N}) = {high_k_frac:.3f}")

            summary_rows.append({
                "regime": regime,
                "variant": variant,
                "M": M,
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "balanced_acc_test": bal,
                "high_k_importance_fraction": high_k_frac,
            })

            tag = f"{regime}_{variant}_M{M}"
            _save_confusion_csv(cm, class_labels, RESULTS_DIR / f"confusion_{tag}.csv")
            _save_kbin_importance_csv(imp, k_centers,
                                      RESULTS_DIR / f"kbin_importance_{tag}.csv")
            _plot_confusion(cm, class_labels,
                            title=f"Confusion — {tag}",
                            path=FIGS_DIR / f"fig_confusion_{tag}.png")
            _plot_kbin_importance(
                imp, k_centers, HIGH_K_TOP_N,
                title=f"k-bin importance — {tag} (high-k fraction {high_k_frac:.3f})",
                path=FIGS_DIR / f"fig_kbin_importance_{tag}.png",
            )

    print("[4/4] Writing summary + acc-vs-M plots...")
    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "balanced_acc_summary.csv", index=False
    )
    for regime in REGIMES:
        _plot_acc_vs_M(summary_rows, regime, FIGS_DIR / f"fig_acc_vs_M_{regime}.png")

    # ------------------------------------------------------------------------
    # Verdict per LEDGER section 5 outcome bands.
    # PASS requires: (per-sightline >= 0.55 OR stacked M=64 >= 0.70)
    #                AND high-k importance fraction > 0.50.
    # ------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("VERDICT  (per LEDGER section 5 outcome bands)")
    print("=" * 70)

    def _lookup(regime: str, M: int) -> Dict[str, Any]:
        for r in summary_rows:
            if r["regime"] == regime and r["M"] == M:
                return r
        raise KeyError(f"missing summary row regime={regime} M={M}")

    primary = "per_sightline"
    per_sl_primary = _lookup(primary, 1)
    stacked_m64_primary = _lookup(primary, 64)

    acc_per_sl = per_sl_primary["balanced_acc_test"]
    acc_m64 = stacked_m64_primary["balanced_acc_test"]
    high_k_per_sl = per_sl_primary["high_k_importance_fraction"]
    high_k_m64 = stacked_m64_primary["high_k_importance_fraction"]

    accuracy_pass = (acc_per_sl >= BAR_PER_SIGHTLINE) or (acc_m64 >= BAR_STACKED_M64)
    # Anti-degeneracy gate: enforce on the winning arm (whichever crossed the bar).
    if acc_per_sl >= BAR_PER_SIGHTLINE:
        anti_deg_pass = high_k_per_sl > HIGH_K_IMPORTANCE_FRACTION
    elif acc_m64 >= BAR_STACKED_M64:
        anti_deg_pass = high_k_m64 > HIGH_K_IMPORTANCE_FRACTION
    else:
        anti_deg_pass = False

    # NULL signature: per-sightline ~ baseline AND stacked acc is flat over M.
    stacked_accs = [_lookup(primary, M)["balanced_acc_test"] for M in M_VALUES]
    monotone_rise = all(
        stacked_accs[i + 1] >= stacked_accs[i] - 0.005 for i in range(len(stacked_accs) - 1)
    ) and (stacked_accs[-1] - stacked_accs[0] > 0.03)
    near_baseline = abs(acc_per_sl - V02_BASELINE) < 0.03
    flat_stack = (max(stacked_accs) - min(stacked_accs)) < 0.03

    if accuracy_pass and anti_deg_pass:
        verdict = "PASS"
        decision = "GO to full-track scoping"
    elif accuracy_pass and not anti_deg_pass:
        verdict = "Ambiguous (accuracy bar met but k-bin importance not concentrated at high-k)"
        decision = "Hold for confounder triangulation (likely mean-flux leak; check global regime)"
    elif (not accuracy_pass) and monotone_rise:
        verdict = "Ambiguous / SNR-limited"
        decision = "Conditional GO; spec a STACKED-pipeline track, not per-sightline"
    elif near_baseline and flat_stack:
        verdict = "NULL"
        decision = "VALID END STATE; NO-GO on the full track; record finding"
    else:
        verdict = "Ambiguous (no PASS, no clean NULL signature)"
        decision = "Manual review by PI required"

    print(f"  primary regime: {primary}")
    print(f"  per-sightline (M=1):    bal_acc = {acc_per_sl:.4f}  "
          f"high-k frac = {high_k_per_sl:.3f}  bar = {BAR_PER_SIGHTLINE}")
    print(f"  stacked     (M=64):     bal_acc = {acc_m64:.4f}  "
          f"high-k frac = {high_k_m64:.3f}  bar = {BAR_STACKED_M64}")
    print(f"  v0.2 baseline = {V02_BASELINE}  chance = {CHANCE}")
    print(f"  accuracy_pass = {accuracy_pass}  anti_degeneracy_pass = {anti_deg_pass}")
    print(f"  monotone_rise = {monotone_rise}  near_baseline = {near_baseline}  "
          f"flat_stack = {flat_stack}")
    print(f"  stacked accs by M {M_VALUES}: "
          f"{[f'{a:.4f}' for a in stacked_accs]}")
    print()
    print(f"  >>> VERDICT: {verdict}")
    print(f"  >>> DECISION: {decision}")
    print("=" * 70)

    print()
    print(f"Artifacts written under: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
