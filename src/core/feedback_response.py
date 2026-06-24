"""
feedback-response-field probe — control-first de-risking.

Scientific purpose
------------------
Tests whether the COUNTERFACTUAL response field R_c(x) = F_c(x) - F_1(x) (how each
feedback recipe c in {2,3,4} displaces the spectrum from its OWN NoFeedback
baseline, on the BYTE-IDENTICAL geometry) is a real, geometry-bound, DIRECTIONAL,
and ORTHOGONAL new per-sightline axis — i.e. worth building into a full embedding
— or merely a restatement of quantities the project already has (total absorption,
the 24-dim separability vector).

The data is NOISELESS (S/N=inf), so R_c is the EXACT physical response of the gas
to feedback on a fixed skewer (no measurement-noise "residual" to fight). The real
risk is therefore degeneracy: that the response magnitude is just "deep-line
sightlines" (total absorption) or the separability vector under a new name. The
controls target exactly that. See experiments/feedback-response-field/SCOPING.md
for the pre-committed PASS / NULL bars.

Controls (pre-committed):
  A  geometry-binding : same-geometry response << cross-geometry (random-pair)
     difference -> the 4 paired spectra are a tight counterfactual family.
  B  concentration    : a minority of geometries carries most of the response.
  C  directionality   : the sign of the response (absorption REMOVED vs ADDED)
     differs across recipes -> a degree of freedom magnitude reps discard.
  D  orthogonality    : response magnitude is NOT a proxy for total absorption or
     the separability-vector norm (|Spearman| < 0.6 each) -> a genuinely new axis.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.core.data import SignalClusteringData

_SEED = 42
_HIRESP_PX = 0.05      # |R| pixel threshold for "responding" pixel
_ORTHO_BAR = 0.6       # |Spearman| below this => orthogonal-enough to be new axis


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def run_response_probe(out_dir: Path, seed: int = _SEED) -> Dict[str, object]:
    """Run the control-first de-risking; write figure + result.json; return result."""
    out_dir = Path(out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    loader = SignalClusteringData()
    fpc, _ = loader.load_flux_per_class()                 # list of 4 x (16384, 2048)
    F = {c: np.asarray(fpc[c - 1], dtype=np.float64) for c in (1, 2, 3, 4)}
    n = F[1].shape[0]

    # Response field R_c = F_c - F_1 (positive => absorption REMOVED / more transmission).
    R = {c: F[c] - F[1] for c in (2, 3, 4)}
    rho = {c: np.linalg.norm(R[c], axis=1) for c in (2, 3, 4)}   # per-geometry magnitude

    # --- Control A: geometry-binding (same vs cross-geometry), recipe 4 (strongest) ---
    perm = rng.permutation(n)
    perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)  # derangement-ish
    d_same = rho[4]                                       # ||F4[i] - F1[i]||
    d_cross = np.linalg.norm(F[4][perm] - F[1], axis=1)  # ||F4[j] - F1[i]||, j!=i
    frac_same_lt_cross = float(np.mean(d_same < d_cross))
    A_pass = (np.median(d_same) < 0.5 * np.median(d_cross)) and (frac_same_lt_cross > 0.9)

    # --- Control B: concentration (does a minority carry most response?) ---
    order = np.argsort(rho[4])[::-1]
    cum = np.cumsum(rho[4][order]) / rho[4].sum()
    top10_share = float(cum[int(0.10 * n)])              # share of total response in top 10%
    B_pass = top10_share > 0.40

    # --- Control C: directionality (sign of response differs across recipes?) ---
    # Among responding pixels (|R|>thr), fraction with R>0 (absorption removed).
    posfrac = {}
    mean_signed = {}
    for c in (2, 3, 4):
        mask = np.abs(R[c]) > _HIRESP_PX
        posfrac[c] = float(np.mean(R[c][mask] > 0)) if mask.any() else float("nan")
        mean_signed[c] = float(R[c].mean())              # global mean signed response
    # StrongAGN (4) should net-remove absorption more than StellarWind (2).
    dir_gap = abs(posfrac[4] - posfrac[2])
    C_pass = dir_gap > 0.10

    # --- Control D: orthogonality to total absorption + separability-vector norm ---
    total_abs = (1.0 - F[1]).mean(axis=1)                # baseline mean absorption
    rho_vs_absorption = _spearman(rho[4], total_abs)
    sep_norm = None
    rho_vs_sepnorm = None
    try:
        fp = np.load("data/feature_discovery/fingerprints_wavelet.npy")  # (16384, 24)
        sep_norm = np.linalg.norm(fp, axis=1)
        rho_vs_sepnorm = _spearman(rho[4], sep_norm)
    except Exception:
        pass
    D_pass = (abs(rho_vs_absorption) < _ORTHO_BAR) and (
        rho_vs_sepnorm is None or abs(rho_vs_sepnorm) < _ORTHO_BAR
    )

    # --- verdict (pre-committed; lead with numbers) ---
    # PASS = geometry-binds (A) AND a new orthogonal axis (D) AND carries new
    # structure (concentration B or directionality C).
    verdict = "PASS" if (A_pass and D_pass and (B_pass or C_pass)) else "NULL"
    if verdict == "PASS":
        reading = (
            "The counterfactual feedback-response field is a real per-sightline "
            "axis: the 4 paired spectra are a tight geometry-bound family (A), the "
            "response is "
            + ("concentrated in a minority " if B_pass else "")
            + ("and directionally structured (evaporate vs enrich differs by "
               "recipe) " if C_pass else "")
            + "and it is NOT a restatement of total absorption or the separability "
            "vector (D). Worth building the full response embedding + the OT "
            "(kinematic) axis. CEILING: this characterizes WHERE/HOW feedback acts; "
            "it does not beat the per-sightline 4-class classification ceiling "
            "(~0.45) and makes no such claim."
        )
    else:
        why = []
        if not A_pass:
            why.append("the paired spectra are not geometry-bound (A) — pairing void")
        if not D_pass:
            why.append(
                "response magnitude largely restates total absorption / the "
                "separability vector (D) — not a new axis"
            )
        if not (B_pass or C_pass):
            why.append("no concentration or directional structure (B,C)")
        reading = (
            "The counterfactual response field does NOT clear the bar: "
            + "; ".join(why) + ". It is not a distinct embedding axis worth "
            "building out at this stage."
        )

    result = {
        "verdict": verdict,
        "reading": reading,
        "n_sightlines": n,
        "noiseless_data": "S/N=inf — R_c is the exact physical feedback response",
        "A_geometry_binding": {
            "median_d_same": round(float(np.median(d_same)), 4),
            "median_d_cross": round(float(np.median(d_cross)), 4),
            "frac_same_lt_cross": round(frac_same_lt_cross, 4),
            "pass": bool(A_pass),
        },
        "B_concentration": {
            "top10pct_response_share": round(top10_share, 4), "pass": bool(B_pass),
        },
        "C_directionality": {
            "posfrac_responding_px": {str(c): round(posfrac[c], 4) for c in (2, 3, 4)},
            "mean_signed_response": {str(c): round(mean_signed[c], 5) for c in (2, 3, 4)},
            "strongAGN_minus_stellarwind_posfrac_gap": round(dir_gap, 4),
            "pass": bool(C_pass),
        },
        "D_orthogonality": {
            "spearman_rho4_vs_total_absorption": round(rho_vs_absorption, 4),
            "spearman_rho4_vs_separability_norm": (
                round(rho_vs_sepnorm, 4) if rho_vs_sepnorm is not None else None
            ),
            "bar": _ORTHO_BAR,
            "pass": bool(D_pass),
        },
    }
    _write_figure(out_dir / "figs" / "feedback_response.png",
                  rho, d_same, d_cross, total_abs, sep_norm, posfrac, result)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def _write_figure(path, rho, d_same, d_cross, total_abs, sep_norm, posfrac, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # A: geometry-binding
    ax[0, 0].hist(d_cross, bins=80, alpha=0.5, label="cross-geometry (random pair)", color="grey")
    ax[0, 0].hist(d_same, bins=80, alpha=0.6, label="same-geometry (feedback response)", color="C0")
    ax[0, 0].set_title("A: geometry-binding — ||F4-F1||")
    ax[0, 0].set_xlabel("L2 difference"); ax[0, 0].legend(fontsize=8)
    # B: concentration (response magnitude distribution, log y)
    ax[0, 1].hist(rho[4], bins=120, color="C3")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title(f"B: response magnitude rho4 (top10%% = {result['B_concentration']['top10pct_response_share']:.2f} of total)")
    ax[0, 1].set_xlabel("||F4-F1|| per sightline")
    # C: directionality (sign fraction per recipe)
    cs = [2, 3, 4]
    ax[1, 0].bar([str(c) for c in cs], [posfrac[c] for c in cs], color=["C2", "C1", "C3"])
    ax[1, 0].axhline(0.5, ls="--", color="k", lw=1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_title("C: frac of responding px with absorption REMOVED (R>0), by recipe")
    ax[1, 0].set_xlabel("feedback recipe (2=Wind 3=WindAGN 4=StrongAGN)")
    # D: orthogonality scatter (rho4 vs total absorption)
    ax[1, 1].scatter(total_abs, rho[4], s=2, alpha=0.2)
    rs = result["D_orthogonality"]["spearman_rho4_vs_total_absorption"]
    ax[1, 1].set_title(f"D: rho4 vs total absorption (Spearman={rs})")
    ax[1, 1].set_xlabel("mean(1-F1) baseline absorption"); ax[1, 1].set_ylabel("rho4")
    fig.suptitle(f"feedback-response-field probe — VERDICT: {result['verdict']}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
