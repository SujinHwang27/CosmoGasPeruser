"""Confusability-residual probe (thin exploratory, NON-paper).

Scientific question (the entire scope — see
``experiments/confusability-residual/SCOPING.md``):

    After regressing out per-sightline MEAN FLUX and LINE DENSITY, does RF
    ``predict_proba`` confusability retain structure that correlates with any
    third physical quantity at |Spearman| >= 0.20?

Design (ONE script, ONE figure, ONE D-XX — anti-scope-creep boundary, binding):

1. Reproduce the recorded RF_Raw fit exactly (the cross-check anchor
   ``export_episode4_rf_confusion_matrices`` on ``service/data-export``): raw
   flux, single 80/20 stratified holdout, seed 42, recorded hyperparameters.
   Read ``predict_proba`` on the held-out test sightlines (the only sightlines
   with an out-of-sample probability vector).
2. Confusability score per test sightline = Shannon entropy of the 4-vector
   (primary) and top-two margin (secondary).
3. Per-sightline covariates via the eda Tier-1 extractor (``scripts/eda_sherwood``),
   recomputed PER SIGHTLINE (the ``results/eda/`` CSVs are class-level aggregates,
   so this re-runs the SAME Tier-1 method per sightline — reuse, not new feature
   extraction):
       - removed confounds : mean_flux, line_density
       - candidate "third quantities" (pre-registered, dispersion/shape measures
         that are NOT linear re-encodings of the two confounds):
           depth_mean, depth_std, depth_max, ew_mean, ew_std, gap_std
       - re-encoding diagnostics (EXCLUDED from PASS by argument, reported only
         to justify exclusion): total_ew (~ linear in mean_flux),
         gap_mean (~ 1/line_density)
4. Partial Spearman of confusability vs each candidate, controlling for
   [mean_flux, line_density]. PASS iff any non-re-encoding candidate has
   |partial rho| >= 0.20; otherwise NULL (the pre-committed, expected outcome).

Pre-committed criteria are symmetric (rule-5) and NULL is a valid end state
(rule-7) — this script reports the partial-correlation table as observed and
does not spin it. It is NOT a paper trigger under any outcome.

Outputs (under ``results/confusability_residual/``):
    partial_correlations.csv   - tidy table: candidate, partial_rho, p, n, is_reencoding
    verdict.txt                - PASS/NULL decision + RF holdout-accuracy cross-check
    figs/residual_partial_spearman.png  - the single figure (|partial rho| bars + 0.20 line)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.core.data import SignalClusteringData

# --- Recorded RF_Raw fit, mirrored from export_episode4_rf_confusion_matrices ---
_RF_RECORDED_HP = dict(
    n_estimators=100,
    max_depth=25,
    max_features="sqrt",
    min_samples_leaf=50,
    min_samples_split=100,
    random_state=42,
    n_jobs=-1,
)
_RF_RAW_RECORDED_ACC = 0.4514  # recorded 10-fold accuracy, RF_Raw (cross-check anchor)
_SEED = 42
_CLASSES = (1, 2, 3, 4)

# Pre-registered candidate "third quantities" (NOT re-encodings of the two confounds).
_CANDIDATES = ("depth_mean", "depth_std", "depth_max", "ew_mean", "ew_std", "gap_std")
# Computed-and-reported only to justify their exclusion (linear re-encodings).
_REENCODINGS = ("total_ew", "gap_mean")
_PASS_THRESHOLD = 0.20


def _load_eda_extractor():
    """Import the eda Tier-1 extractor functions by path (scripts/ is not a package)."""
    eda_path = Path(__file__).resolve().parent / "eda_sherwood.py"
    spec = importlib.util.spec_from_file_location("eda_sherwood", eda_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load eda extractor from {eda_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs top-level imports only; __main__ block is skipped
    return mod


def fit_rf_raw_and_proba(
    X_raw: np.ndarray,  # shape: (n_sightlines, 2048)
    y: np.ndarray,  # shape: (n_sightlines,)
    seed: int = _SEED,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reproduce the recorded RF_Raw holdout fit; return (test_idx, proba, holdout_acc).

    Splits the INDEX array with the same stratified call so test sightlines can be
    mapped back to original positions for the covariate join.
    """
    n = X_raw.shape[0]
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(**_RF_RECORDED_HP)
    clf.fit(X_raw[idx_tr], y[idx_tr])
    y_pred = clf.predict(X_raw[idx_te])
    holdout_acc = float((y_pred == y[idx_te]).mean())
    proba = clf.predict_proba(X_raw[idx_te])  # shape: (n_test, 4)
    return idx_te, proba, holdout_acc


def confusability_scores(proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-sightline confusability from the class-probability 4-vector.

    entropy : Shannon entropy (nats); higher = more confusable.
    margin  : 1 - (p_top1 - p_top2); higher = more confusable.
    """
    p = np.clip(proba, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)  # shape: (n_test,)
    psorted = np.sort(proba, axis=1)
    margin = 1.0 - (psorted[:, -1] - psorted[:, -2])  # shape: (n_test,)
    return {"entropy": entropy, "margin": margin}


def per_sightline_covariates(
    flux_te: np.ndarray,  # shape: (n_test, 2048)
    eda,
) -> pd.DataFrame:
    """Recompute Tier-1 per-sightline covariates with the eda extractor.

    Wavelength grid: pixel index (arange). Spearman is rank-based, so the constant
    wavelength scaling does not change any reported partial correlation; line
    density and gap statistics are rescaled monotonically.
    """
    n, npix = flux_te.shape
    lam = np.arange(npix, dtype=np.float64)
    cols = ["mean_flux", "line_density", *_CANDIDATES, *_REENCODINGS]
    out = {c: np.full(n, np.nan, dtype=np.float64) for c in cols}
    for i in range(n):
        f = flux_te[i]
        minima = eda.detect_local_minima(f)
        depths = eda.absorption_depths(f, minima)
        gaps = eda.gap_statistics(lam, minima)
        local_ews = eda.local_equivalent_widths(lam, f, minima)
        out["mean_flux"][i] = float(f.mean())
        out["line_density"][i] = float(eda.line_density(minima, lam))
        out["depth_mean"][i] = float(np.mean(depths)) if depths.size else 0.0
        out["depth_std"][i] = float(np.std(depths)) if depths.size else 0.0
        out["depth_max"][i] = float(np.max(depths)) if depths.size else 0.0
        out["ew_mean"][i] = float(np.mean(local_ews)) if local_ews.size else 0.0
        out["ew_std"][i] = float(np.std(local_ews)) if local_ews.size else 0.0
        out["gap_std"][i] = float(gaps["gap_std"])
        out["gap_mean"][i] = float(gaps["gap_mean"])
        out["total_ew"][i] = float(eda.total_equivalent_width(lam, f))
    df = pd.DataFrame(out)
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("non-finite covariate produced; investigate before testing")
    return df


def _rank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x)


def _control_basis(controls: np.ndarray, nonlinear: bool) -> np.ndarray:
    """Rank-space design matrix for the controls.

    linear    : [1, r1, r2]
    nonlinear : [1, r1, r2, r1^2, r2^2, r1*r2] — removes the confounds' quadratic
                and interaction structure, not just their linear part. The faithful
                reading of "regress out mean flux AND line density" is to strip
                their FULL association, so the verdict is based on this basis.
    """
    n = controls.shape[0]
    rctrl = [_rank(controls[:, j]) for j in range(controls.shape[1])]
    cols = [np.ones(n), *rctrl]
    if nonlinear:
        r1, r2 = rctrl[0], rctrl[1]
        cols += [r1 * r1, r2 * r2, r1 * r2]
    return np.column_stack(cols)


def partial_spearman(
    c: np.ndarray, q: np.ndarray, controls: np.ndarray, nonlinear: bool = False
) -> Tuple[float, float]:
    """Spearman partial correlation of c and q controlling for `controls` (n, k).

    Rank-transform all variables, residualize rank(c) and rank(q) on the control
    basis via OLS, then Pearson-correlate the residuals. `nonlinear` adds quadratic
    + interaction terms of the controls (see :func:`_control_basis`).
    """
    rc, rq = _rank(c), _rank(q)
    A = _control_basis(controls, nonlinear)
    res_c = rc - A @ np.linalg.lstsq(A, rc, rcond=None)[0]
    res_q = rq - A @ np.linalg.lstsq(A, rq, rcond=None)[0]
    rho, p = stats.pearsonr(res_c, res_q)
    return float(rho), float(p)


def main(out_dir: Path = Path("results/confusability_residual")) -> None:
    out_dir = Path(out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    print("[1/5] loading flux per class ...", flush=True)
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()
    X_raw = np.vstack(
        [np.asarray(flux_per_class[c - 1], dtype=np.float64) for c in _CLASSES]
    )  # shape: (65536, 2048)
    y = np.concatenate(
        [np.full(flux_per_class[c - 1].shape[0], c, dtype=np.int64) for c in _CLASSES]
    )  # shape: (65536,)
    print(f"      X_raw {X_raw.shape}, y {y.shape}", flush=True)

    print("[2/5] fitting RF_Raw (recorded HP, seed 42) + predict_proba ...", flush=True)
    idx_te, proba, holdout_acc = fit_rf_raw_and_proba(X_raw, y, seed=_SEED)
    acc_delta = round(holdout_acc - _RF_RAW_RECORDED_ACC, 4)
    print(
        f"      holdout_acc={holdout_acc:.4f} (recorded {_RF_RAW_RECORDED_ACC}, "
        f"delta {acc_delta:+.4f}); n_test={idx_te.size}",
        flush=True,
    )

    conf = confusability_scores(proba)

    print("[3/5] recomputing Tier-1 per-sightline covariates (test set) ...", flush=True)
    eda = _load_eda_extractor()
    cov = per_sightline_covariates(X_raw[idx_te], eda)

    print("[4/5] partial Spearman (control: mean_flux, line_density; "
          "linear + nonlinear) ...", flush=True)
    controls = cov[["mean_flux", "line_density"]].to_numpy()
    # Persist the per-sightline analysis frame for auditability (idx + covariates + scores).
    analysis = cov.copy()
    analysis.insert(0, "orig_index", idx_te)
    for score_name, c in conf.items():
        analysis[f"confusability_{score_name}"] = c
    analysis.to_csv(out_dir / "per_sightline_analysis.csv", index=False)

    rows: List[Dict[str, object]] = []
    for score_name, c in conf.items():
        for q_name in (*_CANDIDATES, *_REENCODINGS):
            q = cov[q_name].to_numpy()
            rho_lin, p_lin = partial_spearman(c, q, controls, nonlinear=False)
            rho_nl, p_nl = partial_spearman(c, q, controls, nonlinear=True)
            rows.append({
                "confusability": score_name,
                "candidate": q_name,
                "partial_rho_linear": round(rho_lin, 4),
                "partial_rho_nonlinear": round(rho_nl, 4),
                "abs_rho": round(abs(rho_nl), 4),  # verdict uses the conservative NL control
                "p_value_nonlinear": p_nl,
                "n": int(c.shape[0]),
                "is_reencoding": q_name in _REENCODINGS,
            })
    # Diagnostic: raw Spearman of each re-encoding vs the confound it re-encodes.
    reenc_diag = {
        "total_ew~mean_flux": stats.spearmanr(cov["total_ew"], cov["mean_flux"]).statistic,
        "gap_mean~line_density": stats.spearmanr(cov["gap_mean"], cov["line_density"]).statistic,
    }
    table = pd.DataFrame(rows).sort_values(["confusability", "abs_rho"], ascending=[True, False])
    csv_path = out_dir / "partial_correlations.csv"
    table.to_csv(csv_path, index=False)

    # Decision: PASS iff any non-re-encoding candidate clears the threshold under the
    # conservative NONLINEAR control (abs_rho). Score-fragility is reported explicitly:
    # a candidate that clears 0.20 under one confusability score but not the other is
    # demoted by the pre-committed rule-4 fragility (author-defined score).
    non_reenc = table[~table["is_reencoding"]]
    max_abs = float(non_reenc["abs_rho"].max())
    passed = max_abs >= _PASS_THRESHOLD
    top = non_reenc.loc[non_reenc["abs_rho"].idxmax()]
    # Both-scores check on the strongest candidate (rule-4 fragility test).
    same_cand = non_reenc[non_reenc["candidate"] == top["candidate"]]
    clears_both = bool((same_cand["abs_rho"] >= _PASS_THRESHOLD).all()) and len(same_cand) == 2
    robust = passed and clears_both
    verdict = "PASS" if robust else ("PASS-FRAGILE" if passed else "NULL")

    print("[5/5] writing verdict + figure ...", flush=True)
    with open(out_dir / "verdict.txt", "w") as fh:
        fh.write(f"VERDICT: {verdict}\n")
        fh.write(f"max |partial rho| (non-re-encoding, NONLINEAR control) = {max_abs:.4f} "
                 f"(threshold {_PASS_THRESHOLD})\n")
        fh.write(f"strongest candidate: {top['candidate']} via {top['confusability']} "
                 f"(rho_nl={top['partial_rho_nonlinear']:.4f}, "
                 f"rho_lin={top['partial_rho_linear']:.4f}, "
                 f"p={top['p_value_nonlinear']:.3g}, n={int(top['n'])})\n")
        fh.write("  same candidate under both confusability scores (rule-4 fragility):\n")
        for _, r in same_cand.iterrows():
            fh.write(f"    {r['confusability']:>8}: rho_nl={r['partial_rho_nonlinear']:+.4f} "
                     f"(linear {r['partial_rho_linear']:+.4f})  "
                     f"{'>= ' if r['abs_rho'] >= _PASS_THRESHOLD else '<  '}{_PASS_THRESHOLD}\n")
        fh.write(f"\nRF_Raw holdout accuracy cross-check: {holdout_acc:.4f} "
                 f"(recorded {_RF_RAW_RECORDED_ACC}, delta {acc_delta:+.4f})\n\n")
        fh.write("re-encoding diagnostics (raw Spearman vs the confound re-encoded):\n")
        for k, v in reenc_diag.items():
            fh.write(f"  {k}: rho={v:.4f}\n")
        fh.write("\nInterpretation (rule-7, honest framing):\n")
        if verdict == "PASS":
            fh.write(
                f"  Residual confusability retains structure: {top['candidate']} clears "
                f"|rho|>={_PASS_THRESHOLD} under BOTH scores and the nonlinear control,\n"
                f"  after removing mean_flux + line_density. Exploratory finding only —\n"
                f"  NOT a paper trigger (SCOPING guardrail 2); promotion needs fresh PI review.\n"
            )
        elif verdict == "PASS-FRAGILE":
            fh.write(
                f"  Borderline: {top['candidate']} clears {_PASS_THRESHOLD} under "
                f"'{top['confusability']}' but NOT the other confusability score.\n"
                f"  Per the pre-committed rule-4 fragility (author-defined score), this is\n"
                f"  DEMOTED to no-headline; it does not meet the PASS bar as a finding.\n"
                f"  Effectively NULL for reporting. NOT a paper trigger.\n"
            )
        else:
            fh.write(
                "  Confusability re-derives the known mean-flux / line-density confound;\n"
                "  no orthogonal residual structure above the pre-committed 0.20 bar.\n"
                "  Expected NULL (SCOPING). NOT a paper trigger.\n"
            )

    # --- the single figure: nonlinear-control |partial rho| per candidate, BOTH scores ---
    # Grouped bars (entropy vs margin) make the rule-4 score-fragility visible at a glance.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (
        table[table["confusability"] == "margin"]
        .sort_values("abs_rho", ascending=True)["candidate"].tolist()
    )
    ent = table[table["confusability"] == "entropy"].set_index("candidate").loc[order]
    mar = table[table["confusability"] == "margin"].set_index("candidate").loc[order]
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(ypos + 0.2, mar["abs_rho"], height=0.4, color="#1f77b4", label="margin score")
    ax.barh(ypos - 0.2, ent["abs_rho"], height=0.4, color="#ff7f0e", label="entropy score")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{c}*" if mar.loc[c, "is_reencoding"] else c for c in order])
    ax.axvline(_PASS_THRESHOLD, color="crimson", ls="--", lw=1.5,
               label=f"PASS threshold {_PASS_THRESHOLD}")
    ax.set_xlabel("|partial Spearman|, nonlinear control (mean_flux + line_density + quad + interaction)")
    ax.set_title(f"Confusability-residual probe — VERDICT: {verdict}\n"
                 "(* = re-encoding of a confound, excluded from PASS)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "residual_partial_spearman.png", dpi=150)
    plt.close(fig)

    print(f"\nDONE — VERDICT: {verdict}  (max |partial rho| = {max_abs:.4f})", flush=True)
    print(f"  table:   {csv_path}", flush=True)
    print(f"  verdict: {out_dir / 'verdict.txt'}", flush=True)
    print(f"  figure:  {out_dir / 'figs' / 'residual_partial_spearman.png'}", flush=True)


if __name__ == "__main__":
    main()
