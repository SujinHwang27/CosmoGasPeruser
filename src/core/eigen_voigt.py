"""
eigen-voigt-embedding track: data variants and per-sightline representations.

Scientific purpose
------------------
Supplies the inputs the track compares: the fiducial noiseless Sherwood flux, the
tau-rescaled control (each class's optical depth scaled by a constant A_c so that every
class has the SAME global mean flux — the standard way of absorbing a UV-background /
photoionization-rate difference, and the strongest test that a feedback signal is not
the known mean-flux offset re-derived), the sightline-position groups that every CV
split must respect (sightline index i is the same skewer through shared initial
conditions in all four classes), and the per-sightline representations the
eigenvector embeddings are built on (raw flux, Voigt line statistics, P_F(k)) plus the
scalar confound baselines.
"""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from src.core.data import SignalClusteringData
from src.core.transforms import FluxPowerSpectrumTransform

FLUX_BASE = SignalClusteringData.FLUX_BASE
N_CLASSES = 4
VARIANTS = ("fiducial", "taurescaled")


def load_grid(base: str = FLUX_BASE) -> Tuple[np.ndarray, np.ndarray]:
    """Shared wavelength (Å) and velocity (km/s) grids; identical across classes."""
    wave = np.load(Path(base) / "1" / "wave.npy")           # shape: (n_pix,)
    vel = np.load(Path(base) / "1" / "vel.npy")             # shape: (n_pix,)
    for c in range(2, N_CLASSES + 1):
        if not (np.array_equal(wave, np.load(Path(base) / str(c) / "wave.npy"))
                and np.array_equal(vel, np.load(Path(base) / str(c) / "vel.npy"))):
            raise ValueError(f"load_grid: class {c} grid differs from class 1")
    return wave, vel


def _mean_flux_scaled(tau: np.ndarray, a: float, block: int = 2048) -> float:
    tot = 0.0
    for s in range(0, tau.shape[0], block):
        tot += float(np.exp(-a * np.asarray(tau[s:s + block])).sum())
    return tot / tau.size


def tau_rescale_factors(base: str = FLUX_BASE) -> Dict[str, object]:
    """A_c such that <exp(-A_c tau_c)> equals the pooled mean flux of the 4 classes."""
    taus = [np.load(Path(base) / str(c) / "tau.npy", mmap_mode="r") for c in range(1, N_CLASSES + 1)]
    means = [_mean_flux_scaled(t, 1.0) for t in taus]
    target = float(np.mean(means))
    factors = [brentq(lambda a, t=t: _mean_flux_scaled(t, a) - target, 0.05, 20.0, xtol=1e-10)
               for t in taus]
    return {"target_mean_flux": target, "class_mean_flux": means, "A": factors}


def load_flux_variant(variant: str, class_id: int, rows: Optional[np.ndarray] = None,
                      factors: Optional[List[float]] = None, base: str = FLUX_BASE) -> np.ndarray:
    """Flux of one class for a data variant, optionally restricted to `rows`.

    fiducial    : the stored flux (== roll(exp(-tau), 1), verified 2026-10-18).
    taurescaled : roll(exp(-A_c tau), 1), same pixel convention as the stored flux.
    """
    if variant not in VARIANTS:
        raise ValueError(f"load_flux_variant: unknown variant {variant!r}")
    sel = slice(None) if rows is None else rows
    if variant == "fiducial":
        F = np.asarray(np.load(Path(base) / str(class_id) / "flux.npy", mmap_mode="r")[sel],
                       dtype=np.float64)
    else:
        if factors is None:
            raise ValueError("load_flux_variant: taurescaled needs the A_c factors")
        tau = np.asarray(np.load(Path(base) / str(class_id) / "tau.npy", mmap_mode="r")[sel],
                         dtype=np.float64)
        F = np.roll(np.exp(-factors[class_id - 1] * tau), 1, axis=1)
    if not np.isfinite(F).all():
        raise ValueError(f"load_flux_variant: non-finite flux in class {class_id}")
    return np.clip(F, 0.0, 1.0)                              # shape: (n_rows, n_pix)


def stacked_labels_groups(n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """Class labels 1..4 and skewer-position groups for class-major stacking."""
    y = np.repeat(np.arange(1, N_CLASSES + 1), n_per_class)               # shape: (4n,)
    groups = np.tile(np.arange(n_per_class), N_CLASSES)                   # shape: (4n,)
    return y, groups


def scalar_baselines(flux: np.ndarray) -> np.ndarray:
    """[<F>, <tau_eff>] per sightline. Total equivalent width is affine in <F> (EW =
    N dv (1 - <F>)), so <tau_eff> = <-ln F> is the second, non-redundant scalar."""
    meanF = flux.mean(axis=1)
    taueff = -np.log(np.clip(flux, 1e-300, None)).clip(max=700.0).mean(axis=1)
    return np.column_stack([meanF, taueff])                  # shape: (n, 2)


def pk_features(flux_per_class: List[np.ndarray], dv_kms: float, n_bins: int = 32) -> np.ndarray:
    """Per-sightline log P_F(k), normalised by ONE pooled global mean flux.

    FluxPowerSpectrumTransform divides by the mean of whatever it is given; passing the
    stacked array of all classes gives a single pooled <F>, so no per-class mean (a label
    leak) enters.
    """
    X = np.vstack(flux_per_class)
    return FluxPowerSpectrumTransform(n_bins=n_bins, dv_kms=dv_kms).fit_transform(X)


# ---------------------------------------------------------------------------
# Stage EV-2 orchestration
# ---------------------------------------------------------------------------
def build_representations(variant: str, catalog_dir: Path, rows: Optional[np.ndarray],
                          factors: Optional[List[float]], knn_k: int = 64,
                          log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Raw flux, Voigt line statistics and P_F(k) for all 4 classes (class-major), plus
    scalars, labels, skewer groups, Voigt convergence flags and kNN tables."""
    from src.core.eigen_embedding import knn_table
    from src.core.voigt import line_statistics_features
    from sklearn.preprocessing import StandardScaler

    wave, vel = load_grid()
    dv = float(vel[1] - vel[0])
    span = wave.size * dv
    fluxes, lines, conv = [], [], []
    for c in range(1, N_CLASSES + 1):
        F = load_flux_variant(variant, c, rows=rows, factors=factors)          # shape: (n, 2048)
        cat = np.load(Path(catalog_dir) / f"class{c}.npz")
        n = F.shape[0]
        if cat["diag"].shape[0] != n:
            raise ValueError(f"build_representations: class {c} catalog has {cat['diag'].shape[0]} "
                             f"sightlines, flux has {n}")
        lines.append(line_statistics_features(cat["sightline"], cat["v_kms"], cat["b_kms"],
                                              cat["logN"], n_sightlines=n, span_kms=span))
        keys = list(cat["diag_keys"])
        conv.append(cat["diag"][:, keys.index("converged")])
        fluxes.append(F)
    n = fluxes[0].shape[0]
    y, groups = stacked_labels_groups(n)
    raw = np.vstack(fluxes).astype(np.float32)                                # shape: (4n, 2048)
    scal = scalar_baselines(np.vstack(fluxes))
    pk = pk_features(fluxes, dv)
    voigt = np.vstack(lines)
    reps: Dict[str, Dict[str, np.ndarray]] = {"raw": {"X": raw}, "voigt": {"X": voigt}, "pk": {"X": pk}}
    for name, R in reps.items():
        Xk = R["X"] if name == "raw" else StandardScaler().fit_transform(R["X"])
        R["knn_idx"], R["knn_dist"] = knn_table(Xk, knn_k)
        if log:
            log(f"  kNN table {name}: X {R['X'].shape}")
    return {"reps": reps, "scalars": scal, "y": y, "groups": groups,
            "voigt_converged": np.concatenate(conv), "n_per_class": n}


def run_evaluation(data: Dict[str, object], out_dir: Path, seeds: Sequence[int], n_folds: int = 5,
                   n_jobs: int = -1, log: Callable[[str], None] = print) -> None:
    """Grouped-CV evaluation; resumable per (seed, fold). Writes split_metrics.csv,
    split_diags.jsonl and oof_seed<S>.npz (out-of-fold predicted labels per config)."""
    import csv
    import json
    from src.core.eigen_eval import evaluate_split, group_splits

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "split_metrics.csv"
    done = set()
    if metrics_path.exists():
        with open(metrics_path) as fh:
            done = {(int(r["seed"]), int(r["fold"])) for r in csv.DictReader(fh)}
    y, groups = data["y"], data["groups"]
    for seed in seeds:
        oof_path = out_dir / f"oof_seed{seed}.npz"
        oof: Dict[str, np.ndarray] = dict(np.load(oof_path)) if oof_path.exists() else {}
        for fold, (train, test) in enumerate(group_splits(groups, n_folds, seed)):
            if (seed, fold) in done:
                continue
            t0 = time.time()
            rows, preds, diags = evaluate_split(data["reps"], data["scalars"], y, groups, train, test,
                                                seed=seed, n_jobs=n_jobs, raw_direct_rf=(seed == seeds[0]))
            for key, p in preds.items():
                arr = oof.setdefault(key, np.zeros(y.size, dtype=np.int8))
                arr[test] = p
            np.savez_compressed(oof_path, **oof)
            write_header = not metrics_path.exists()
            with open(metrics_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["seed", "fold"] + list(rows[0].keys()))
                if write_header:
                    w.writeheader()
                for r in rows:
                    w.writerow({"seed": seed, "fold": fold, **r})
            with open(out_dir / "split_diags.jsonl", "a") as fh:
                fh.write(json.dumps({"seed": seed, "fold": fold, **diags}) + "\n")
            log(f"[ev-2] seed {seed} fold {fold}: {len(rows)} configs in {time.time() - t0:.0f}s")


def summarise(data: Dict[str, object], out_dir: Path, seeds: Sequence[int], n_boot: int = 2000,
              log: Callable[[str], None] = print) -> None:
    """Per-config split statistics, LE-vs-PCA group-bootstrap deltas, scalar-baseline
    headroom, and the Voigt-converged-only sensitivity."""
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score
    from src.core.eigen_eval import DIMS, LE_MODES, group_bootstrap_delta

    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "split_metrics.csv")
    score_cols = [c for c in df.columns if c.startswith("ba")]
    keys = ["rep", "method", "d", "resid", "clf"]
    g = df.groupby(keys)[score_cols]
    summ = g.median().add_suffix("_median").join(g.quantile(0.16).add_suffix("_p16")) \
        .join(g.quantile(0.84).add_suffix("_p84")).join(g.size().rename("n_splits")).reset_index()
    summ.to_csv(out_dir / "summary_by_config.csv", index=False)

    y, groups = data["y"], data["groups"]
    conv_group = np.ones(groups.max() + 1, dtype=bool)
    np.logical_and.at(conv_group, groups, data["voigt_converged"] > 0.5)
    keep = conv_group[groups]
    oofs = {s: dict(np.load(out_dir / f"oof_seed{s}.npz")) for s in seeds}
    oof0 = oofs[seeds[0]]
    deltas = []
    for rep in ("raw", "voigt", "pk"):
        for mode in LE_MODES:
            for d in DIMS:
                for resid in (0, 1):
                    for clf in ("logreg", "rf"):
                        ka, kb = f"{rep}|{mode}|{d}|{resid}|{clf}", f"{rep}|pca|{d}|{resid}|{clf}"
                        if ka not in oof0 or kb not in oof0:
                            continue
                        res = group_bootstrap_delta(y, groups, oof0[ka], oof0[kb], n_boot=n_boot, seed=0)
                        per_seed = [balanced_accuracy_score(y, oofs[s][ka]) - balanced_accuracy_score(y, oofs[s][kb])
                                    for s in seeds]
                        deltas.append({"rep": rep, "le_mode": mode, "d": d, "resid": resid, "clf": clf,
                                       "delta_ba4_seed0": res["delta"], "ci95_lo": res["ci_lo"],
                                       "ci95_hi": res["ci_hi"], "delta_ba4_median_seeds": float(np.median(per_seed)),
                                       "ba4_le_converged_only": float(balanced_accuracy_score(y[keep], oof0[ka][keep])),
                                       "ba4_pca_converged_only": float(balanced_accuracy_score(y[keep], oof0[kb][keep]))})
    pd.DataFrame(deltas).to_csv(out_dir / "le_minus_pca.csv", index=False)
    base = {}
    for name in ("meanF", "meanF_taueff"):
        for clf in ("logreg", "rf"):
            k = f"scalar|{name}|{1 if name == 'meanF' else 2}|0|{clf}"
            base[(name, clf)] = float(balanced_accuracy_score(y, oof0[k]))
    head = []
    for k, p in oof0.items():
        rep, method, d, resid, clf = k.split("|")
        if rep == "scalar":
            continue
        ba = float(balanced_accuracy_score(y, p))
        head.append({"rep": rep, "method": method, "d": int(d), "resid": int(resid), "clf": clf,
                     "ba4_seed0": ba, "headroom_vs_meanF_taueff": ba - base[("meanF_taueff", clf)]})
    pd.DataFrame(head).sort_values("ba4_seed0", ascending=False).to_csv(out_dir / "headroom_vs_scalar.csv", index=False)
    (out_dir / "scalar_baselines.json").write_text(
        __import__("json").dumps({f"{a}|{b}": v for (a, b), v in base.items()}, indent=2))
    log(f"[ev-2] summary written; converged-only groups kept: {keep.mean():.4f}")
