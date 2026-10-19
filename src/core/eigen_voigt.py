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
    """[<F>, <tau>_clip] per sightline.

    Total equivalent width is affine in <F> (to the ~1.8% lambda-span weighting of the
    log-lambda grid), and the literature tau_eff = -ln<F> is monotone in it, so the second
    scalar is the pixel-mean optical depth <-ln F> with tau clipped at 30. The clip keeps
    it a property of the absorption profile rather than a count of underflowed pixels
    (flux underflows above tau ~ 745).
    """
    meanF = flux.mean(axis=1)
    meantau = (-np.log(np.clip(flux, 1e-300, None))).clip(max=30.0).mean(axis=1)
    return np.column_stack([meanF, meantau])                 # shape: (n, 2)


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
BOX_KPC = 60000.0             # Sherwood box, kpc/h (axis.npy coordinates span 0-60000)
N_BLOCK_CELLS = 8             # 7.5 Mpc/h transverse cells per projection axis -> 192 blocks
VOIGT_DETECT_SENSITIVITY = (12.5, 13.0)


def build_representations(variant: str, catalog_dir: Path, rows: Optional[np.ndarray],
                          factors: Optional[List[float]], knn_k: int = 64,
                          log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Raw flux, Voigt line statistics and P_F(k) for all 4 classes (class-major), plus
    scalars, labels, skewer ids, spatial blocks, Voigt convergence flags and kNN tables.

    Voigt features keep NaN for undefined entries (imputed per training fold by the
    harness); the kNN metric uses a label-free pooled-median fill.
    """
    from src.core.eigen_embedding import knn_table
    from src.core.eigen_eval import spatial_blocks
    from src.core.voigt import line_statistics_features
    from sklearn.preprocessing import StandardScaler

    wave, vel = load_grid()
    dv = float(vel[1] - vel[0])
    span = wave.size * dv
    fluxes, conv, cats = [], [], []
    for c in range(1, N_CLASSES + 1):
        F = load_flux_variant(variant, c, rows=rows, factors=factors)          # shape: (n, 2048)
        cat = dict(np.load(Path(catalog_dir) / f"class{c}.npz"))
        if cat["diag"].shape[0] != F.shape[0]:
            raise ValueError(f"build_representations: class {c} catalog has {cat['diag'].shape[0]} "
                             f"sightlines, flux has {F.shape[0]}")
        keys = list(cat["diag_keys"])
        conv.append(cat["diag"][:, keys.index("converged")])
        fluxes.append(F)
        cats.append(cat)
    n = fluxes[0].shape[0]
    y, skewers = stacked_labels_groups(n)

    def voigt_feats(detect: float) -> np.ndarray:
        return np.vstack([line_statistics_features(cat["sightline"], cat["v_kms"], cat["b_kms"], cat["logN"],
                                                   n_sightlines=n, span_kms=span, logN_detect=detect,
                                                   impute=False) for cat in cats])

    axis = np.load(Path(FLUX_BASE) / "1" / "axis.npy")
    if rows is not None:
        axis = axis[rows]
    blocks = np.tile(spatial_blocks(axis, BOX_KPC, N_BLOCK_CELLS), N_CLASSES)
    raw = np.vstack(fluxes).astype(np.float32)                                # shape: (4n, 2048)
    scal = scalar_baselines(np.vstack(fluxes))
    reps: Dict[str, Dict[str, np.ndarray]] = {"raw": {"X": raw}, "voigt": {"X": voigt_feats(12.0)},
                                              "pk": {"X": pk_features(fluxes, dv)}}
    for name, R in reps.items():
        Xk = R["X"]
        if name != "raw":
            Xk = np.where(np.isnan(Xk), np.nanmedian(Xk, axis=0), Xk)
            Xk = StandardScaler().fit_transform(Xk)
        R["knn_idx"], R["knn_dist"] = knn_table(Xk, knn_k)
        if log:
            log(f"  kNN table {name}: X {R['X'].shape}, NaN frac {float(np.isnan(R['X']).mean()):.4f}")
    for det in VOIGT_DETECT_SENSITIVITY:
        reps[f"voigt_det{det:g}"] = {"X": voigt_feats(det), "direct_only": True}
    return {"reps": reps, "scalars": scal, "y": y, "skewers": skewers, "blocks": blocks,
            "voigt_converged": np.concatenate(conv), "n_per_class": n}


def run_evaluation(data: Dict[str, object], out_dir: Path, seeds: Sequence[int], n_folds: int = 5,
                   n_jobs: int = -1, raw_rf_seeds: int = 3, log: Callable[[str], None] = print) -> None:
    """Spatially-blocked CV; resumable per (seed, fold). Writes split_metrics.csv (with
    n_train/n_test), split_diags.jsonl, oof_seed<S>.npz (labels per config) and, for the
    first seed, oof_proba_seed<S>.npz (float16 probabilities)."""
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
    y, blocks, skewers = data["y"], data["blocks"], data["skewers"]
    for seed in seeds:
        keep_proba = seed == seeds[0]
        oof_path = out_dir / f"oof_seed{seed}.npz"
        prob_path = out_dir / f"oof_proba_seed{seed}.npz"
        oof: Dict[str, np.ndarray] = dict(np.load(oof_path)) if oof_path.exists() else {}
        prob: Dict[str, np.ndarray] = dict(np.load(prob_path)) if (keep_proba and prob_path.exists()) else {}
        for fold, (train, test) in enumerate(group_splits(blocks, n_folds, seed)):
            if (seed, fold) in done:
                continue
            t0 = time.time()
            rows, preds, probas, diags = evaluate_split(
                data["reps"], data["scalars"], y, skewers, train, test, seed=seed, n_jobs=n_jobs,
                raw_direct_rf=(seed in seeds[:raw_rf_seeds]), keep_proba=keep_proba)
            for key, p in preds.items():
                oof.setdefault(key, np.zeros(y.size, dtype=np.int8))[test] = p
            for key, p in probas.items():
                prob.setdefault(key, np.zeros((y.size, 4), dtype=np.float16))[test] = p
            np.savez_compressed(oof_path, **oof)
            if keep_proba:
                np.savez_compressed(prob_path, **prob)
            write_header = not metrics_path.exists()
            with open(metrics_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["seed", "fold", "n_train", "n_test"] + list(rows[0].keys()))
                if write_header:
                    w.writeheader()
                for r in rows:
                    w.writerow({"seed": seed, "fold": fold, "n_train": train.size, "n_test": test.size, **r})
            with open(out_dir / "split_diags.jsonl", "a") as fh:
                fh.write(json.dumps({"seed": seed, "fold": fold, **diags}) + "\n")
            log(f"[ev-2] seed {seed} fold {fold}: {len(rows)} configs in {time.time() - t0:.0f}s")


def summarise(data: Dict[str, object], out_dir: Path, seeds: Sequence[int], n_boot: int = 2000,
              log: Callable[[str], None] = print) -> None:
    """Pre-registered statistics ([D-04] as amended by [D-07]).

    * summary_by_config.csv : split median / p16 / p84 of every score.
    * le_minus_pca.csv      : per (input, LE mode, d, resid, clf): seed-averaged-correctness
      BA delta (eigenmap - PCA), block-bootstrap 95% CI, simultaneous max-T band over all
      delta rows, Nadeau-Bengio corrected t on the per-split BA differences, and the
      primary-cell flag.
    * headroom_vs_scalar.csv: BA of every config minus the [<F>, <tau>_clip] baseline
      (same classifier), seed-averaged, with block CIs.
    * diagnostics.json      : Voigt-converged-only BA + <F> of kept vs dropped skewers;
      BA within <F> deciles for the primary cells.
    """
    import json
    import pandas as pd
    from src.core.eigen_eval import (DIMS, LE_MODES, PRIMARY, RESID_MODES, block_bootstrap,
                                     corrected_resampled_t, max_t_band)

    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "split_metrics.csv")
    score_cols = [c for c in df.columns if c.startswith("ba")]
    keys = ["rep", "method", "d", "resid", "clf"]
    g = df.groupby(keys)[score_cols]
    summ = g.median().add_suffix("_median").join(g.quantile(0.16).add_suffix("_p16")) \
        .join(g.quantile(0.84).add_suffix("_p84")).join(g.size().rename("n_splits")).reset_index()
    summ.to_csv(out_dir / "summary_by_config.csv", index=False)

    y, blocks = data["y"], data["blocks"]
    oofs = {s: dict(np.load(out_dir / f"oof_seed{s}.npz")) for s in seeds}
    all_keys = sorted(set.intersection(*[set(o) for o in oofs.values()]))
    correct = {k: np.mean([(oofs[s][k] == y) for s in seeds], axis=0) for k in all_keys}
    n_tr, n_te = int(df["n_train"].median()), int(df["n_test"].median())

    def split_scores(key: str) -> pd.Series:
        rep, method, d, resid, clf = key.split("|")
        m = (df.rep == rep) & (df.method == method) & (df.d == int(d)) & (df.resid == int(resid)) & (df.clf == clf)
        return df[m].set_index(["seed", "fold"])["ba4"]

    pairs = []
    for rep in ("raw", "voigt", "pk"):
        for mode in LE_MODES:
            for d in DIMS:
                for resid in RESID_MODES:
                    for clf in ("logreg", "rf"):
                        ka, kb = f"{rep}|{mode}|{d}|{resid}|{clf}", f"{rep}|pca|{d}|{resid}|{clf}"
                        if ka in correct and kb in correct:
                            pairs.append((rep, mode, d, resid, clf, ka, kb))
    D = np.stack([correct[ka] - correct[kb] for *_, ka, kb in pairs])
    point, boot = block_bootstrap(y, blocks, D, n_boot=n_boot, seed=0)
    lo_s, hi_s = max_t_band(point, boot)
    out = []
    for i, (rep, mode, d, resid, clf, ka, kb) in enumerate(pairs):
        sa, sb = split_scores(ka), split_scores(kb)
        common = sa.index.intersection(sb.index)
        nb = corrected_resampled_t((sa[common] - sb[common]).to_numpy(), n_tr, n_te)
        out.append({"rep": rep, "le_mode": mode, "d": d, "resid": resid, "clf": clf,
                    "delta_ba4": float(point[i]),
                    "ci95_lo": float(np.percentile(boot[:, i], 2.5)), "ci95_hi": float(np.percentile(boot[:, i], 97.5)),
                    "simul95_lo": float(lo_s[i]), "simul95_hi": float(hi_s[i]),
                    "nb_t": nb["t"], "nb_p_one_sided": nb["p_one_sided"], "n_splits": nb["n_splits"],
                    "primary_cell": bool(mode == PRIMARY["le_mode"] and clf == PRIMARY["clf"] and d == PRIMARY["d"])})
    pd.DataFrame(out).to_csv(out_dir / "le_minus_pca.csv", index=False)

    base_key = {clf: f"scalar|meanF_meantau|2|0|{clf}" for clf in ("logreg", "rf")}
    hkeys = [k for k in all_keys if not k.startswith("scalar|")]
    H = np.stack([correct[k] - correct[base_key[k.split("|")[-1]]] for k in hkeys])
    hp, hb = block_bootstrap(y, blocks, H, n_boot=n_boot, seed=1)
    bp, _ = block_bootstrap(y, blocks, np.stack([correct[k] for k in hkeys]), n_boot=1, seed=2)
    head = [{"rep": k.split("|")[0], "method": k.split("|")[1], "d": int(k.split("|")[2]),
             "resid": int(k.split("|")[3]), "clf": k.split("|")[4], "ba4_seedavg": float(bp[i]),
             "headroom": float(hp[i]), "headroom_ci95_lo": float(np.percentile(hb[:, i], 2.5)),
             "headroom_ci95_hi": float(np.percentile(hb[:, i], 97.5))} for i, k in enumerate(hkeys)]
    pd.DataFrame(head).sort_values("ba4_seedavg", ascending=False).to_csv(out_dir / "headroom_vs_scalar.csv", index=False)

    conv_sk = np.ones(data["skewers"].max() + 1, dtype=bool)
    np.logical_and.at(conv_sk, data["skewers"], data["voigt_converged"] > 0.5)
    keep = conv_sk[data["skewers"]]
    meanF = data["scalars"][:, 0]
    diag: Dict[str, object] = {"converged_skewer_frac": float(keep.mean()),
                               "meanF_kept": float(meanF[keep].mean()),
                               "meanF_dropped": float(meanF[~keep].mean()) if (~keep).any() else None,
                               "scalar_baselines_seedavg": {k: float(np.mean([(oofs[s][v] == y).mean() for s in seeds]))
                                                            for k, v in base_key.items()}}
    dec = np.digitize(meanF, np.quantile(meanF, np.linspace(0, 1, 11)[1:-1]))
    for k in all_keys:
        parts = k.split("|")
        is_primary = len(parts) == 5 and parts[1] in ("pca", PRIMARY["le_mode"]) and parts[2] == str(PRIMARY["d"]) \
            and parts[4] == PRIMARY["clf"] and parts[3] == "0"
        if is_primary or k.startswith("scalar|") or parts[1] == "direct":
            from sklearn.metrics import balanced_accuracy_score
            o = oofs[seeds[0]][k]
            diag[f"converged_only_ba4|{k}"] = float(balanced_accuracy_score(y[keep], o[keep]))
            diag[f"ba4_by_meanF_decile|{k}"] = [float(balanced_accuracy_score(y[dec == q], o[dec == q])) for q in range(10)]
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    log(f"[ev-2] summary written; converged-only skewers kept: {keep.mean():.4f}")


# ---------------------------------------------------------------------------
# Stage EV-3 post-hoc checks (catalog-only; defense-panel #3, #9)
# ---------------------------------------------------------------------------
def catalog_posthoc(art_dir: Path, mode: str = "full") -> Dict[str, object]:
    """Per-class catalog statistics and per-feature class separability, fiducial vs
    tau-rescaled.

    For every Voigt line-statistic feature, the one-vs-one ROC AUC for the class pairs
    (reported as |AUC - 0.5|, so direction-free) in each variant. Panel #3: tau-rescaling
    shifts every log N of a class by log10 A_c, so a feature whose separability RISES
    under rescaling flags a rescaling artefact rather than a surviving feedback signal.
    """
    from sklearn.metrics import roc_auc_score
    from src.core.eigen_eval import PAIRS
    from src.core.voigt import line_feature_names, line_statistics_features

    wave, vel = load_grid()
    span = wave.size * float(vel[1] - vel[0])
    names = line_feature_names()
    out: Dict[str, object] = {"per_class": {}, "auc_abs_dev": {}}
    feats = {}
    for variant in VARIANTS:
        per = []
        for c in range(1, N_CLASSES + 1):
            cat = np.load(Path(art_dir) / f"catalog_{variant}_{mode}" / f"class{c}.npz")
            n = cat["diag"].shape[0]
            keys = list(cat["diag_keys"])
            det = cat["logN"] >= 12.0
            out["per_class"][f"{variant}|{c}"] = {
                "n_sightlines": int(n),
                "components_per_sightline": float(cat["logN"].size / n),
                "lines_logN12_per_sightline": float(det.sum() / n),
                "lines_logN13_per_sightline": float((cat["logN"] >= 13.0).sum() / n),
                "b_median_logN12": float(np.median(cat["b_kms"][det])),
                "b_q90_logN12": float(np.percentile(cat["b_kms"][det], 90)),
                "logN_median_logN12": float(np.median(cat["logN"][det])),
                "converged_frac": float(cat["diag"][:, keys.index("converged")].mean()),
                "crashed": int(np.sum(cat["diag"][:, keys.index("n_fit_errors")] < 0)),
                "max_resid_median": float(np.median(cat["diag"][:, keys.index("max_resid")])),
            }
            per.append(line_statistics_features(cat["sightline"], cat["v_kms"], cat["b_kms"], cat["logN"],
                                                n_sightlines=n, span_kms=span, impute=False))
        feats[variant] = per
    for variant in VARIANTS:
        for a, b in PAIRS:
            Xa, Xb = feats[variant][a - 1], feats[variant][b - 1]
            for j, name in enumerate(names):
                va, vb = Xa[:, j], Xb[:, j]
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                if va.size < 50 or vb.size < 50 or np.ptp(np.concatenate([va, vb])) == 0:
                    continue
                auc = roc_auc_score(np.r_[np.zeros(va.size), np.ones(vb.size)], np.r_[va, vb])
                out["auc_abs_dev"][f"{variant}|{a}{b}|{name}"] = float(abs(auc - 0.5))
    return out
