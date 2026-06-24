"""
signed-response-embedding probe — control-first de-risking.

Scientific purpose
------------------
The predecessor probe (`feedback_response.py`, [D-01]) showed the response
MAGNITUDE ||R_c|| = ||F_c - F_1|| restates total absorption (NULL on magnitude),
but that the SIGN of the response is recipe-dependent and physical (C-PASS:
winds ADD absorption, strong-AGN REMOVES it). The sign is the one degree of
freedom every magnitude/content representation provably discards.

This probe asks the one question that decides whether the sign is EMBEDDABLE:
at fixed total absorption, WITHIN a single feedback recipe, do different
sightlines respond in different directions — i.e. is the signed response a
per-sightline AXIS, or merely (i) a per-recipe global CONSTANT (a class label),
and/or (ii) total absorption restated under a sign?

Candidate per-sightline feature: net signed response s_c[i] = mean_x R_c[i,x]
(positive => absorption REMOVED). Gating runs on c=4 (StrongAGN — strongest
directional signal, the most favorable case).

Controls (pre-committed; see experiments/signed-response-embedding/SCOPING.md):
  F  within-recipe spread : SR = IQR(s_4) / |mean(s_4) - mean(s_2)|. The
     per-recipe-constant firewall and THE decider. PASS SR>=0.5, NULL SR<0.25.
  E  absorption orthogonality : |Spearman(s_4, total_abs)| < 0.5 AND the
     absorption-residual spread SR_resid = IQR(resid) / sep_between >= 0.4.
     The [D-13] / magnitude-restated firewall.
  C' cross-recipe ordering : per-sightline means reproduce the global C-PASS
     (mean s_2 < mean s_3 < mean s_4 and s_2 < 0 < s_4). KILL if violated.

Data is NOISELESS (S/N=inf): R_c is the exact physical feedback response.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.core.data import SignalClusteringData

_SEED = 42
_HIRESP_PX = 0.05      # |R| pixel threshold for a "responding" pixel (p_c only)
_ORTHO_BAR = 0.5       # |Spearman| below this => orthogonal-enough
_ORTHO_NULL = 0.7      # |Spearman| at/above this => absorption restated (NULL)
_SR_PASS = 0.5         # spread ratio at/above => per-sightline axis
_SR_NULL = 0.25        # spread ratio below => per-recipe constant
_SR_RESID_PASS = 0.4   # absorption-residual spread at/above => real residual axis
_SR_RESID_NULL = 0.2   # absorption-residual spread below => no residual axis


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _iqr(a: np.ndarray) -> float:
    q75, q25 = np.percentile(a, [75, 25])
    return float(q75 - q25)


def run_signed_response_probe(out_dir: Path, seed: int = _SEED) -> Dict[str, object]:
    """Run the control-first de-risking; write figure + result.json; return result."""
    out_dir = Path(out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    _ = np.random.default_rng(seed)  # determinism contract; no randomness used here

    loader = SignalClusteringData()
    fpc, _y = loader.load_flux_per_class()                 # list of 4 x (16384, 2048)
    F = {c: np.asarray(fpc[c - 1], dtype=np.float64) for c in (1, 2, 3, 4)}
    n = F[1].shape[0]

    # Response field R_c = F_c - F_1 (positive => absorption REMOVED).
    R = {c: F[c] - F[1] for c in (2, 3, 4)}

    # Primary per-sightline signed feature: net signed response (threshold-free).
    s = {c: R[c].mean(axis=1) for c in (2, 3, 4)}          # each shape (16384,)
    mean_s = {c: float(s[c].mean()) for c in (2, 3, 4)}

    # Secondary (reported only): responding-pixel sign fraction, NaN-guarded.
    p = {}
    p_nmasked = {}
    for c in (2, 3, 4):
        mask = np.abs(R[c]) > _HIRESP_PX                   # shape (16384, 2048)
        cnt = mask.sum(axis=1)
        pos = (R[c] > 0) & mask
        with np.errstate(invalid="ignore", divide="ignore"):
            pc = np.where(cnt >= 5, pos.sum(axis=1) / np.maximum(cnt, 1), np.nan)
        p[c] = pc
        p_nmasked[c] = int(np.sum(cnt < 5))

    total_abs = (1.0 - F[1]).mean(axis=1)                  # baseline mean absorption

    # --- Control C': cross-recipe directional ordering (KILL trigger) ---
    ordering_ok = (mean_s[2] < mean_s[3] < mean_s[4]) and (mean_s[2] < 0.0 < mean_s[4])
    if not ordering_ok:
        result = {
            "verdict": "KILL",
            "reading": (
                "C' FAILED — per-sightline recipe means do not reproduce the prior "
                f"global C-PASS (mean s_2={mean_s[2]:.5f}, s_3={mean_s[3]:.5f}, "
                f"s_4={mean_s[4]:.5f}; need s_2<s_3<s_4 and s_2<0<s_4). Feature/load "
                "inconsistent with feedback-response-field [D-01]; emitting no graded "
                "verdict — surface to PI."
            ),
            "n_sightlines": n,
            "C_prime_ordering": {str(c): round(mean_s[c], 6) for c in (2, 3, 4)},
        }
        with open(out_dir / "result.json", "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    # --- Control F: within-recipe spread (the per-recipe-constant firewall) ---
    spread_within = _iqr(s[4])
    sep_between = abs(mean_s[4] - mean_s[2])
    SR = spread_within / sep_between if sep_between > 0 else float("inf")
    F_pass = SR >= _SR_PASS
    F_null = SR < _SR_NULL
    # two-sided minority cross-check (non-gating)
    frac_pos_s4 = float(np.mean(s[4] > 0.0))
    two_sided = 0.05 < frac_pos_s4 < 0.95

    # --- Control E: absorption orthogonality + absorption-residual spread ---
    rho_E = _spearman(s[4], total_abs)
    slope, intercept = np.polyfit(total_abs, s[4], 1)
    resid = s[4] - (slope * total_abs + intercept)
    SR_resid = _iqr(resid) / sep_between if sep_between > 0 else float("inf")
    E_pass = (abs(rho_E) < _ORTHO_BAR) and (SR_resid >= _SR_RESID_PASS)
    E_null = (abs(rho_E) >= _ORTHO_NULL) or (SR_resid < _SR_RESID_NULL)

    # Reported cross-check: orthogonality to the separability-vector norm.
    rho_sepnorm: Optional[float] = None
    sep_norm = None
    try:
        fp = np.load("data/feature_discovery/fingerprints_wavelet.npy")  # (16384, 24)
        sep_norm = np.linalg.norm(fp, axis=1)
        rho_sepnorm = _spearman(s[4], sep_norm)
    except Exception:
        pass

    # --- verdict (pre-committed; PASS = F_pass AND E_pass) ---
    verdict = "PASS" if (F_pass and E_pass) else "NULL"
    if verdict == "PASS":
        reading = (
            "The SIGNED feedback response is a per-sightline directional axis: "
            f"within StrongAGN, sightline-to-sightline spread (SR={SR:.2f}) is "
            "comparable to the between-recipe shift (NOT a per-recipe constant), "
            f"and a real residual axis survives absorption control "
            f"(SR_resid={SR_resid:.2f}, |Spearman(s_4, total_abs)|={abs(rho_E):.2f} "
            "< 0.5). Worth scoping a future signed-response embedding track "
            "(separately, defense-panel-gated). CEILING: characterizes the DIRECTION "
            "of feedback action per sightline; does NOT beat the ~0.45 4-class "
            "classification ceiling and claims nothing about it."
        )
    else:
        why = []
        if F_null:
            why.append(
                f"within-recipe spread is tiny vs the between-recipe shift "
                f"(SR={SR:.2f} < {_SR_NULL}) — the signed response is a PER-RECIPE "
                "CONSTANT (a class label), not a per-sightline axis"
            )
        elif not F_pass:
            why.append(
                f"within-recipe spread is ambiguous (SR={SR:.2f}, in the "
                f"[{_SR_NULL}, {_SR_PASS}) NULL-leaning band) — does not clear the "
                "per-sightline-axis bar"
            )
        if E_null:
            why.append(
                f"the sign tracks total absorption (|Spearman|={abs(rho_E):.2f}) "
                f"and/or no residual axis survives absorption control "
                f"(SR_resid={SR_resid:.2f}) — magnitude restated under a sign "
                "([D-13] deep-line trap, one level deeper)"
            )
        elif not E_pass:
            why.append(
                f"absorption-orthogonality is not clean enough "
                f"(|Spearman(s_4, total_abs)|={abs(rho_E):.2f}, SR_resid={SR_resid:.2f})"
            )
        reading = (
            "The signed feedback response does NOT clear the bar: " + "; ".join(why)
            + ". The SIGN, like the MAGNITUDE before it (feedback-response-field "
            "[D-01]), is not a distinct per-sightline embedding axis at this stage."
        )

    result = {
        "verdict": verdict,
        "reading": reading,
        "n_sightlines": n,
        "noiseless_data": "S/N=inf — R_c is the exact physical feedback response",
        "C_prime_ordering": {
            "mean_s": {str(c): round(mean_s[c], 6) for c in (2, 3, 4)},
            "reproduces_global_C_pass": bool(ordering_ok),
        },
        "F_within_recipe_spread": {
            "spread_within_IQR_s4": round(spread_within, 6),
            "sep_between_meanS4_minus_meanS2": round(sep_between, 6),
            "spread_ratio_SR": round(SR, 4),
            "frac_s4_positive": round(frac_pos_s4, 4),
            "two_sided_minority": bool(two_sided),
            "pass_bar": _SR_PASS, "null_bar": _SR_NULL,
            "pass": bool(F_pass), "null": bool(F_null),
        },
        "E_absorption_orthogonality": {
            "spearman_s4_vs_total_absorption": round(rho_E, 4),
            "spearman_s4_vs_separability_norm": (
                round(rho_sepnorm, 4) if rho_sepnorm is not None else None
            ),
            "SR_resid_after_absorption": round(SR_resid, 4),
            "ortho_bar": _ORTHO_BAR, "ortho_null": _ORTHO_NULL,
            "sr_resid_pass": _SR_RESID_PASS, "sr_resid_null": _SR_RESID_NULL,
            "pass": bool(E_pass), "null": bool(E_null),
        },
        "secondary_posfrac_per_sightline": {
            "median_p_c": {
                str(c): (round(float(np.nanmedian(p[c])), 4)) for c in (2, 3, 4)
            },
            "n_sightlines_masked_lt5_responding_px": {
                str(c): p_nmasked[c] for c in (2, 3, 4)
            },
            "note": "p_c is threshold-dependent (0.05 px cut) -> reported, NOT gating",
        },
    }
    _write_figure(out_dir / "figs" / "signed_response.png",
                  s, resid, total_abs, mean_s, result)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def verify_signed_response_pass(out_dir: Path, seed: int = _SEED) -> Dict[str, object]:
    """Adversarial verification of the [D-01] PASS against the defense-panel attacks.

    Answers, numbers-first (the verdict may DOWNGRADE the PASS):
      #1  SR-bar calibration: sign-permutation null (destroy within-sightline sign
          coherence) + bootstrap CI on SR.
      #4  few-pixel domination: participation ratio (effective # pixels) + top-5
          pixel share of the signed sum, for the bulk and the extreme tail.
      #2/#5 variance decomposition vs total absorption (linear AND nonlinear/binned)
          + conditional spread by absorption decile (the 'fan').
      #7  cross-recipe: coherence + absorption coupling for c in {2,3,4}.
      #8  magnitude recompute: Spearman(||R_4||, total_abs) in-session (expect ~0.778).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    loader = SignalClusteringData()
    fpc, _y = loader.load_flux_per_class()
    F = {c: np.asarray(fpc[c - 1], dtype=np.float64) for c in (1, 2, 3, 4)}
    n, npix = F[1].shape
    R = {c: F[c] - F[1] for c in (2, 3, 4)}
    s = {c: R[c].mean(axis=1) for c in (2, 3, 4)}
    total_abs = (1.0 - F[1]).mean(axis=1)
    mean_s = {c: float(s[c].mean()) for c in (2, 3, 4)}
    sep = abs(mean_s[4] - mean_s[2])
    s4 = s[4]

    def _iqr_(a):
        q75, q25 = np.percentile(a, [75, 25])
        return float(q75 - q25)

    out: Dict[str, object] = {"n_sightlines": n, "n_pixels": npix}

    # --- amplitude of s4 + the extreme negative tail the panel flagged (#3/#4) ---
    ext = np.argsort(s4)[:100]                       # 100 most-negative s4
    out["amplitude_s4"] = {
        "mean": round(float(s4.mean()), 6), "iqr": round(_iqr_(s4), 6),
        "std": round(float(s4.std()), 6),
        "min": round(float(s4.min()), 4), "max": round(float(s4.max()), 4),
        "frac_abs_gt_0p05": round(float(np.mean(np.abs(s4) > 0.05)), 5),
        "frac_abs_gt_0p1": round(float(np.mean(np.abs(s4) > 0.1)), 5),
        "frac_abs_gt_0p3": round(float(np.mean(np.abs(s4) > 0.3)), 6),
        "n_abs_gt_0p3": int(np.sum(np.abs(s4) > 0.3)),
    }
    out["extreme_negative_tail_top100"] = {
        "mean_s4": round(float(s4[ext].mean()), 4),
        "mean_total_abs_of_tail": round(float(total_abs[ext].mean()), 4),
        "overall_mean_total_abs": round(float(total_abs.mean()), 4),
        "note": "if tail total_abs >> overall, the big signed responses live in HIGH-"
                "absorption gas (physical); if << overall, they live in the diffuse "
                "bulk (suspicious per Nasir+2017)",
    }

    # --- #4 few-pixel domination: participation ratio + top-5 signed share ---
    absR = np.abs(R[4])
    pr = (absR.sum(axis=1) ** 2) / np.maximum((R[4] ** 2).sum(axis=1), 1e-30)
    signed_sum = R[4].sum(axis=1)                    # = s4 * npix
    top5_idx = np.argsort(absR, axis=1)[:, -5:]
    top5_signed = np.take_along_axis(R[4], top5_idx, axis=1).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_top5 = np.where(np.abs(signed_sum) > 1e-9, top5_signed / signed_sum, np.nan)
    out["participation_4"] = {
        "median_effective_pixels_all": round(float(np.median(pr)), 1),
        "median_effective_pixels_extreme_tail": round(float(np.median(pr[ext])), 1),
        "p10_effective_pixels_all": round(float(np.percentile(pr, 10)), 1),
        "median_top5px_share_of_signed_sum_all": round(float(np.nanmedian(frac_top5)), 3),
        "median_top5px_share_extreme_tail": round(float(np.nanmedian(frac_top5[ext])), 3),
        "note": "effective pixels >> 5 and top-5 share small => NOT a few-pixel artifact",
    }

    # --- #1 SR calibration: sign-permutation null + bootstrap CI ---
    signs = rng.integers(0, 2, size=R[4].shape) * 2 - 1
    s4_signperm = (absR * signs).mean(axis=1)        # incoherent-sign null
    iqr_obs, iqr_null = _iqr_(s4), _iqr_(s4_signperm)
    coherence_ratio = iqr_obs / iqr_null if iqr_null > 0 else float("inf")
    boot = np.empty(1000)
    for b in range(1000):
        bi = rng.integers(0, n, n)
        sepb = abs(s4[bi].mean() - s[2][bi].mean())
        boot[b] = _iqr_(s4[bi]) / sepb if sepb > 0 else np.nan
    out["SR_calibration_1"] = {
        "SR_observed": round(iqr_obs / sep, 3),
        "iqr_observed": round(iqr_obs, 6),
        "iqr_signperm_null": round(iqr_null, 6),
        "sign_coherence_ratio": round(coherence_ratio, 2),
        "SR_bootstrap_mean": round(float(np.nanmean(boot)), 3),
        "SR_bootstrap_ci95": [round(float(np.nanpercentile(boot, 2.5)), 3),
                              round(float(np.nanpercentile(boot, 97.5)), 3)],
        "SR_ci_excludes_0p5": bool(np.nanpercentile(boot, 2.5) > 0.5),
        "note": "coherence_ratio >> 1 => within-sightline pixel signs are COHERENT "
                "(real direction), not random cancellation; ~1 => signed mean is noise",
    }

    # --- #2/#5 variance decomposition: linear + nonlinear(binned) + the fan ---
    slope, intercept = np.polyfit(total_abs, s4, 1)
    r2_lin = 1.0 - np.var(s4 - (slope * total_abs + intercept)) / np.var(s4)
    edges = np.percentile(total_abs, np.linspace(0, 100, 11))
    binidx = np.clip(np.digitize(total_abs, edges[1:-1]), 0, 9)
    cond_mean = np.array([s4[binidx == b].mean() for b in range(10)])
    cond_iqr = np.array([_iqr_(s4[binidx == b]) for b in range(10)])
    r2_bin = 1.0 - np.var(s4 - cond_mean[binidx]) / np.var(s4)
    out["variance_decomposition_2_5"] = {
        "r2_linear_total_abs": round(float(r2_lin), 3),
        "r2_binned_nonlinear_total_abs": round(float(r2_bin), 3),
        "residual_var_fraction_after_nonlinear": round(float(1 - r2_bin), 3),
        "spearman_s4_total_abs": round(_spearman(s4, total_abs), 3),
        "cond_iqr_by_absorption_decile_lowtohigh": [round(float(x), 5) for x in cond_iqr],
        "note": "if residual_var_fraction is large AND cond_iqr does not vanish across "
                "deciles, the spread is NOT just an absorption fan",
    }

    # --- #7 cross-recipe coherence + coupling ---
    cross = {}
    for c in (2, 3, 4):
        sp = (np.abs(R[c]) * (rng.integers(0, 2, size=R[c].shape) * 2 - 1)).mean(axis=1)
        cr = _iqr_(s[c]) / _iqr_(sp) if _iqr_(sp) > 0 else float("inf")
        cross[str(c)] = {
            "iqr": round(_iqr_(s[c]), 6),
            "sign_coherence_ratio": round(cr, 2),
            "spearman_vs_total_abs": round(_spearman(s[c], total_abs), 3),
            "mean": round(mean_s[c], 6),
        }
    out["cross_recipe_7"] = cross

    # --- #8 magnitude recompute (in-session) ---
    rho4_mag = np.linalg.norm(R[4], axis=1)
    out["magnitude_recompute_8"] = {
        "spearman_normR4_vs_total_abs": round(_spearman(rho4_mag, total_abs), 3),
        "expected_from_predecessor_ledger": 0.778,
    }

    with open(out_dir / "verification.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def _write_figure(path, s, resid, total_abs, mean_s, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = result["F_within_recipe_spread"]
    E = result["E_absorption_orthogonality"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # F: within-recipe s_4 distribution with IQR band + recipe means.
    ax[0, 0].hist(s[4], bins=120, color="C3", alpha=0.8)
    q25, q75 = np.percentile(s[4], [25, 75])
    ax[0, 0].axvspan(q25, q75, color="k", alpha=0.12, label=f"IQR (spread={F['spread_within_IQR_s4']:.4f})")
    for c, col in zip((2, 3, 4), ("C2", "C1", "C3")):
        ax[0, 0].axvline(mean_s[c], color=col, ls="--", lw=1.5, label=f"mean s_{c}={mean_s[c]:.4f}")
    ax[0, 0].axvline(0.0, color="k", lw=0.8)
    ax[0, 0].set_title(f"F: within-recipe s_4 — SR={F['spread_ratio_SR']:.2f} "
                       f"({'PASS' if F['pass'] else 'NULL' if F['null'] else 'AMBIG'})")
    ax[0, 0].set_xlabel("s_4 = mean signed response (>0 = absorption removed)")
    ax[0, 0].legend(fontsize=7)

    # E: absorption-residual distribution.
    ax[0, 1].hist(resid, bins=120, color="C0", alpha=0.8)
    ax[0, 1].axvline(0.0, color="k", lw=0.8)
    ax[0, 1].set_title(f"E: s_4 residual after absorption — SR_resid={E['SR_resid_after_absorption']:.2f}")
    ax[0, 1].set_xlabel("s_4 - linfit(total_abs)")

    # E: s_4 vs total absorption scatter.
    ax[1, 0].scatter(total_abs, s[4], s=2, alpha=0.15)
    ax[1, 0].axhline(0.0, color="k", lw=0.8)
    ax[1, 0].set_title(f"E: s_4 vs total absorption (Spearman={E['spearman_s4_vs_total_absorption']})")
    ax[1, 0].set_xlabel("mean(1 - F1) baseline absorption"); ax[1, 0].set_ylabel("s_4")

    # C': three per-recipe s_c distributions overlaid.
    for c, col in zip((2, 3, 4), ("C2", "C1", "C3")):
        ax[1, 1].hist(s[c], bins=100, alpha=0.45, color=col,
                      label=f"s_{c} (mean {mean_s[c]:.4f})")
    ax[1, 1].axvline(0.0, color="k", lw=0.8)
    ax[1, 1].set_title("C': per-recipe net signed response s_c (2=Wind 3=WindAGN 4=StrongAGN)")
    ax[1, 1].set_xlabel("s_c"); ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"signed-response-embedding probe — VERDICT: {result['verdict']}  "
                 "(CEILING: direction of feedback action; does NOT beat ~0.45 4-class ceiling)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)
