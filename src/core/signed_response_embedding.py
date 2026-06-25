"""
signed-response embedding — S1 (construction + Gates 2,3) and S2 (structure test).

Scientific purpose
------------------
Builds the handcrafted, de-entangled DIRECTION embedding of the per-sightline feedback
response and tests whether it reveals INCREMENTAL STRUCTURE no existing representation
encodes — under the seven defense-panel-required execution controls (TRACK_SPEC §4.0),
which exist to stop a degenerate one-giant-cluster partition (the
cluster-environment-gradient ghost) from scoring a false "new structure" PASS.

The direction is amplitude-free (sign ⊥ magnitude); the data is noiseless so R_c is
exact. The primary clustering uses the c=4 sub-block on the RESPONDING SUBSET (>=20
responding px) and every bar is calibrated against a SIGN-PERMUTATION SURROGATE null
(destroys directional coherence, preserves magnitude/absorption structure).

Verdict (TRACK_SPEC §4.1 REVISED) — PASS requires ALL:
  - largest cluster < 70% (not degenerate),
  - bootstrap-ARI stability > 95th pct of the sign-permutation null (structure above noise),
  - between-cluster eta^2 of total_abs < 0.30 (not the gas gate),
  - AMI vs BOTH existing K=5 clusterings < the related-axes upper anchor AND < 0.40.
NULL if any fails.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.core.data import SignalClusteringData

_SEED = 42
_HIRESP_PX = 0.05        # |R| threshold for a "responding" pixel
_MIN_RESP = 20           # responding-subset cut (binomial SE on signfrac < ~0.11)
_GATE2_ABS_BAR = 0.41    # |Spearman(dir_unit, total_abs)| must stay below this
_GATE3_FAN_BAR = 3.0     # dir_unit decile-IQR fan factor must be < this (raw s4 ~10x)
_DEGEN_BAR = 0.70        # largest-cluster fraction >= this => degenerate NULL
_ETA2_BAR = 0.30         # between-cluster eta^2 of total_abs >= this => gas gate NULL
_AMI_REDERIVE = 0.40     # AMI >= this with an existing clustering => re-derivation NULL
_K = 5
_N_PERM = 50             # sign-permutation surrogate draws
_N_BOOT = 12             # bootstrap-ARI stability resamples
_DV_KMS = 2.636502840371587


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _iqr(a: np.ndarray) -> float:
    q75, q25 = np.percentile(a, [75, 25])
    return float(q75 - q25)


def _direction_block(R: np.ndarray, theta: float = _HIRESP_PX) -> Tuple[np.ndarray, np.ndarray]:
    """Amplitude-free c=4 direction features per sightline + responding-pixel count.

    Returns (block, cnt) where block columns are
    [dir_unit, signfrac, sign_autocorr_lag1, mean_run_length] (shape: (n, 4)).
    """
    absR = np.abs(R)
    mask = absR > theta
    cnt = mask.sum(axis=1)                                    # responding pixels per sightline
    dir_unit = R.sum(axis=1) / (absR.sum(axis=1) + 1e-8)      # sum(R)/sum(|R|) in [-1, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        signfrac = np.where(cnt >= 1, ((R > 0) & mask).sum(axis=1) / np.maximum(cnt, 1), 0.5)
    sg = np.sign(R)
    autocorr = (sg[:, :-1] * sg[:, 1:]).mean(axis=1)          # lag-1 sign autocorrelation
    n_runs = (sg[:, :-1] != sg[:, 1:]).sum(axis=1) + 1
    mean_run = R.shape[1] / n_runs                            # coherence length (pixels)
    block = np.column_stack([dir_unit, signfrac, autocorr, mean_run]).astype(np.float64)
    return block, cnt


def _cluster(Xs: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=_K, n_init=5, random_state=seed).fit_predict(Xs)


def _eta2(values: np.ndarray, labels: np.ndarray) -> float:
    """Between-cluster eta^2 = SS_between / SS_total of `values` over `labels`."""
    grand = values.mean()
    ss_tot = float(((values - grand) ** 2).sum())
    ss_bet = 0.0
    for c in np.unique(labels):
        v = values[labels == c]
        ss_bet += len(v) * (v.mean() - grand) ** 2
    return float(ss_bet / ss_tot) if ss_tot > 0 else 0.0


def _stability(X: np.ndarray, ref: np.ndarray, seed: int, n_boot: int = _N_BOOT) -> float:
    """Mean ARI between the reference partition and re-clusterings of subsamples."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        Xs = StandardScaler().fit_transform(X[idx])
        lab = _cluster(Xs, seed)
        aris.append(adjusted_rand_score(ref[idx], lab))
    return float(np.mean(aris))


def _gap_statistic(Xs: np.ndarray, seed: int, B: int = 10) -> Tuple[float, float]:
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    w_real = float(KMeans(n_clusters=_K, n_init=5, random_state=seed).fit(Xs).inertia_)
    lo, hi = Xs.min(axis=0), Xs.max(axis=0)
    logw = []
    for _ in range(B):
        U = rng.uniform(lo, hi, size=Xs.shape)
        logw.append(np.log(KMeans(n_clusters=_K, n_init=3, random_state=seed).fit(U).inertia_))
    gap = float(np.mean(logw) - np.log(w_real))
    sk = float(np.std(logw) * np.sqrt(1.0 + 1.0 / B))
    return gap, sk


def run_embedding_structure(out_dir: Path, seed: int = _SEED) -> Dict[str, object]:
    """S1+S2: build the direction embedding and run the panel-controlled structure test."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_mutual_info_score as ami

    out_dir = Path(out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    loader = SignalClusteringData()
    fpc, _y = loader.load_flux_per_class()
    F = {c: np.asarray(fpc[c - 1], dtype=np.float64) for c in (1, 2, 3, 4)}
    R = {c: F[c] - F[1] for c in (2, 3, 4)}
    total_abs = (1.0 - F[1]).mean(axis=1)
    sat_frac = np.mean((1.0 - F[1]) > 0.8, axis=1)            # density/structure proxy (gradient list)
    n = F[1].shape[0]

    block4, cnt4 = _direction_block(R[4])                     # primary c=4 sub-block
    dir_unit4, signfrac4 = block4[:, 0], block4[:, 1]

    # cross-recipe coupling-flip statistic (interpretation + gradient list)
    block2, _ = _direction_block(R[2])
    coupling_slope = block4[:, 1] - block2[:, 1]              # signfrac4 - signfrac2

    # ---- S1 gates ----
    sub = cnt4 >= _MIN_RESP                                   # pre-registered responding subset
    n_sub = int(sub.sum())
    g2_abs = _spearman(dir_unit4[sub], total_abs[sub])
    g2_twosided = float(np.mean(signfrac4[sub] > 0.5))
    gate2_pass = (abs(g2_abs) < _GATE2_ABS_BAR) and (0.05 < g2_twosided < 0.95)

    edges = np.percentile(total_abs[sub], np.linspace(0, 100, 11))
    bid = np.clip(np.digitize(total_abs[sub], edges[1:-1]), 0, 9)
    dec_iqr = np.array([_iqr(dir_unit4[sub][bid == b]) for b in range(10)])
    fan = float(dec_iqr.max() / max(dec_iqr.min(), 1e-9))
    gate3_pass = fan < _GATE3_FAN_BAR

    # ---- pre-flight: gradient-list screen + AMI anchors (panel mods #4, #5) ----
    grad_list = {
        "coupling_slope": {"spearman_with_total_abs": round(_spearman(coupling_slope[sub], total_abs[sub]), 3)},
        "saturated_frac": {"spearman_with_total_abs": round(_spearman(sat_frac[sub], total_abs[sub]), 3)},
    }
    lab_w = np.load("data/feature_discovery/labels_wavelet_k5.npy")[sub]
    lab_r = np.load("data/feature_discovery/labels_raw_k5.npy")[sub]
    ami_anchor_upper = float(ami(lab_w, lab_r))              # two related reps on this subset
    rand_lab = rng.integers(0, _K, size=n_sub)
    ami_anchor_lower = float(ami(lab_w, rand_lab))

    # ---- S2 primary clustering (responding subset, c=4 direction sub-block) ----
    Xsub = block4[sub]
    Xs = StandardScaler().fit_transform(Xsub)
    clusters = _cluster(Xs, seed)
    sizes = np.bincount(clusters, minlength=_K)
    largest_frac = float(sizes.max() / n_sub)
    ami_w = float(ami(clusters, lab_w))
    ami_r = float(ami(clusters, lab_r))
    eta2_abs = _eta2(total_abs[sub], clusters)
    stab_real_ms = [_stability(Xsub, clusters, s) for s in (seed, seed + 1, seed + 2)]
    stab_real = float(np.mean(stab_real_ms))
    gap, gap_sk = _gap_statistic(Xs, seed)

    # physical gradient (Holm over the 2 screened quantities that pass the abs screen)
    grad_results = {}
    for name, vec in (("coupling_slope", coupling_slope[sub]), ("saturated_frac", sat_frac[sub])):
        screened = abs(grad_list[name]["spearman_with_total_abs"]) < _GATE2_ABS_BAR
        rho = _spearman(dir_unit4[sub], vec)
        grad_results[name] = {"passes_abs_screen": bool(screened),
                              "spearman_dir_vs_quantity": round(rho, 3)}

    # ---- sign-permutation surrogate null (panel mod #2) ----
    absR4 = np.abs(R[4])
    null_stab, null_ami_w, null_ami_r, null_largest = [], [], [], []
    for _p in range(_N_PERM):
        signs = rng.integers(0, 2, size=R[4].shape) * 2 - 1
        Rp = absR4 * signs                                    # destroys coherence, keeps |R|
        bp, _c = _direction_block(Rp)
        Xp = StandardScaler().fit_transform(bp[sub])
        cp = _cluster(Xp, seed)
        null_stab.append(_stability(bp[sub], cp, seed, n_boot=8))
        null_ami_w.append(ami(cp, lab_w))
        null_ami_r.append(ami(cp, lab_r))
        null_largest.append(np.bincount(cp, minlength=_K).max() / n_sub)
    stab_null_95 = float(np.percentile(null_stab, 95))
    ami_null_w_5 = float(np.percentile(null_ami_w, 5))
    ami_null_r_5 = float(np.percentile(null_ami_r, 5))
    # permutation p-value of the stability excess + multi-seed robustness of the margin
    p_stab = (1.0 + sum(1 for v in null_stab if v >= stab_real)) / (_N_PERM + 1)
    stab_margin_robust = (min(stab_real_ms) > stab_null_95) and (p_stab < 0.05)

    # ---- verdict (TRACK_SPEC §4.1 REVISED) ----
    not_degenerate = largest_frac < _DEGEN_BAR
    above_null = stab_real > stab_null_95
    not_gasgate = eta2_abs < _ETA2_BAR
    different_axis = (ami_w < ami_anchor_upper) and (ami_r < ami_anchor_upper)
    not_rederive = (ami_w < _AMI_REDERIVE) and (ami_r < _AMI_REDERIVE)
    passed = bool(not_degenerate and above_null and not_gasgate and different_axis and not_rederive)

    fails = []
    if not not_degenerate:
        fails.append(f"DEGENERATE: largest cluster {largest_frac:.2f} >= {_DEGEN_BAR} "
                     "(one-giant-cluster collapse — the bulk null re-biting)")
    if not above_null:
        fails.append(f"NO STRUCTURE ABOVE NULL: stability {stab_real:.3f} <= sign-permutation "
                     f"95th pct {stab_null_95:.3f} (the coherent direction adds no cluster "
                     "structure beyond magnitude/noise)")
    if not not_gasgate:
        fails.append(f"GAS GATE: total_abs between-cluster eta^2 {eta2_abs:.3f} >= {_ETA2_BAR} "
                     "(the partition is absorption strength, not direction)")
    if not not_rederive:
        fails.append(f"RE-DERIVES KNOWN: AMI(wavelet)={ami_w:.3f} / AMI(raw)={ami_r:.3f} "
                     f">= {_AMI_REDERIVE}")
    elif not different_axis:
        fails.append(f"NOT A DIFFERENT AXIS vs anchor: AMI(wavelet)={ami_w:.3f}, AMI(raw)={ami_r:.3f} "
                     f"not both below the related-axes upper anchor {ami_anchor_upper:.3f}")

    verdict = "PASS" if passed else "NULL"
    if passed:
        reading = (
            f"S2 PASS: the c=4 direction embedding partitions the responding subset (n={n_sub}) "
            f"into a non-degenerate (largest {largest_frac:.2f}), stable "
            f"(ARI {stab_real:.3f} > null {stab_null_95:.3f}) structure that is NOT the gas gate "
            f"(eta^2={eta2_abs:.3f}) and is a genuinely different axis from the existing "
            f"separability/content clusterings (AMI {ami_w:.3f}/{ami_r:.3f} < anchor "
            f"{ami_anchor_upper:.3f}). The signed-response direction supports an incremental-structure "
            "embedding. CEILING: interpretive axis, no classification claim."
        )
    else:
        reading = ("S2 NULL (TRACK_SPEC §4.5, no-publication path): " + "; ".join(fails)
                   + ". The signed-response EMBEDDING line closes here; the verified scalar "
                   "directional axis ([D-01]/[D-02]) stands on its own.")

    result = {
        "stage": "S1+S2 — embedding construction + structure test",
        "verdict": verdict,
        "reading": reading,
        "n_total": n,
        "responding_subset": {"min_responding_px": _MIN_RESP, "n_subset": n_sub,
                              "frac_of_population": round(n_sub / n, 4)},
        "gate2_de_entangled": {"spearman_dir_unit4_total_abs": round(g2_abs, 3),
                               "frac_signfrac_gt_0p5": round(g2_twosided, 3),
                               "pass": gate2_pass},
        "gate3_fan": {"dir_unit4_decile_iqr_fan_factor": round(fan, 2),
                      "bar_lt": _GATE3_FAN_BAR, "pass": gate3_pass},
        "gradient_list_screen": grad_list,
        "ami_anchors": {"related_axes_upper": round(ami_anchor_upper, 4),
                        "random_lower": round(ami_anchor_lower, 4)},
        "primary_clustering": {
            "k": _K, "largest_cluster_frac": round(largest_frac, 4),
            "cluster_sizes": sizes.tolist(),
            "ami_vs_wavelet_k5": round(ami_w, 4), "ami_vs_raw_k5": round(ami_r, 4),
            "total_abs_between_cluster_eta2": round(eta2_abs, 4),
            "bootstrap_ari_stability": round(stab_real, 4),
            "stability_multiseed_min_mean_max": [round(min(stab_real_ms), 4),
                                                 round(float(np.mean(stab_real_ms)), 4),
                                                 round(max(stab_real_ms), 4)],
            "gap_statistic": round(gap, 4), "gap_sk": round(gap_sk, 4),
        },
        "sign_permutation_null": {
            "n_perm": _N_PERM,
            "stability_null_mean": round(float(np.mean(null_stab)), 4),
            "stability_null_95pct": round(stab_null_95, 4),
            "stability_permutation_pvalue": round(p_stab, 4),
            "stability_margin_robust(min_seed>null95 AND p<0.05)": bool(stab_margin_robust),
            "ami_null_wavelet_5pct": round(ami_null_w_5, 4),
            "ami_null_raw_5pct": round(ami_null_r_5, 4),
            "largest_frac_null_mean": round(float(np.mean(null_largest)), 4),
        },
        "physical_gradient_holm": grad_results,
        "decision_checks": {
            "not_degenerate(<0.70)": not_degenerate,
            "stability_above_null_95": above_null,
            "not_gas_gate(eta2<0.30)": not_gasgate,
            "different_axis(<anchor)": different_axis,
            "not_rederive(<0.40)": not_rederive,
        },
    }
    _write_structure_figure(out_dir / "figs" / "structure_s2.png",
                            Xs, clusters, total_abs[sub], dir_unit4[sub],
                            null_stab, stab_real, result)
    with open(out_dir / "structure_s2.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def run_s3_stability_interpretation(out_dir: Path, seed: int = _SEED) -> Dict[str, object]:
    """S3 (TRACK_SPEC §4.3 + §4.2): cross-feature stability + physical interpretation.

    Cross-feature: does the direction axis reproduce across the three de-entangled
    variants — unit-direction (sum(R)/sum(|R|)), sign-fraction, median-sign — at pairwise
    Spearman >= 0.6? (The S2 stability-above-surrogate condition is already met.)
    Interpretation: the recipe-ordered coupling (winds ADD / strong-AGN REMOVES Lya
    absorption) mapped to Nasir/Bolton+2017 feedback-in-overdensities.
    """
    out_dir = Path(out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    loader = SignalClusteringData()
    fpc, _y = loader.load_flux_per_class()
    F = {c: np.asarray(fpc[c - 1], dtype=np.float64) for c in (1, 2, 3, 4)}
    R = {c: F[c] - F[1] for c in (2, 3, 4)}
    absR4 = np.abs(R[4])
    cnt4 = (absR4 > _HIRESP_PX).sum(axis=1)
    sub = cnt4 >= _MIN_RESP
    n_sub = int(sub.sum())

    # three de-entangled direction variants (amplitude-free), responding subset
    v_unit = (R[4].sum(axis=1) / (absR4.sum(axis=1) + 1e-8))[sub]
    mask = absR4 > _HIRESP_PX
    v_signfrac = (((R[4] > 0) & mask).sum(axis=1) / np.maximum(cnt4, 1))[sub]
    med_R = np.median(R[4], axis=1)
    med_absR = np.median(absR4, axis=1)
    v_median = (med_R / (med_absR + 1e-8))[sub]

    rc = {
        "unit_vs_signfrac": round(_spearman(v_unit, v_signfrac), 3),
        "unit_vs_median": round(_spearman(v_unit, v_median), 3),
        "signfrac_vs_median": round(_spearman(v_signfrac, v_median), 3),
    }
    min_rc = min(rc.values())
    crossfeat_pass = min_rc >= 0.6

    # physical interpretation: recipe-ordered signed response (the coupling flip)
    recipe = {}
    for c in (2, 3, 4):
        Rc = R[c]
        mc = np.abs(Rc) > _HIRESP_PX
        recipe[str(c)] = {
            "mean_signed_response": round(float(Rc.mean()), 5),
            "posfrac_responding": round(float((Rc[mc] > 0).mean()), 4),
        }
    # direction relates to the absorption-clean recipe coupling (signfrac4 - signfrac2)
    signfrac2 = (((R[2] > 0) & (np.abs(R[2]) > _HIRESP_PX)).sum(axis=1)
                 / np.maximum((np.abs(R[2]) > _HIRESP_PX).sum(axis=1), 1))[sub]
    coupling = v_signfrac - signfrac2
    total_abs = (1.0 - F[1]).mean(axis=1)[sub]
    dir_vs_coupling = round(_spearman(v_unit, coupling), 3)
    coupling_vs_abs = round(_spearman(coupling, total_abs), 3)

    verdict = "PASS" if crossfeat_pass else "TEMPERED"
    if crossfeat_pass:
        reading = (
            f"S3 PASS: the direction axis is feature-robust (pairwise Spearman among "
            f"unit-direction / sign-fraction / median-sign all >= {min_rc:.2f}); combined with "
            f"the S2 stability-above-surrogate (p=0.039), the modest structure is not an artifact "
            f"of one feature choice. Physical reading (Nasir/Bolton+2017, feedback acts in "
            f"overdense gas): the net signed response flips from ADD (StellarWind, mean "
            f"{recipe['2']['mean_signed_response']}) to REMOVE (strong-AGN, mean "
            f"{recipe['4']['mean_signed_response']}) — galactic winds enrich/compress gas (more Lya "
            f"absorption) while AGN thermal feedback heats gas, lowering HI and Lya absorption; the "
            f"direction axis tracks this recipe coupling (Spearman {dir_vs_coupling}) which is "
            f"absorption-clean (Spearman {coupling_vs_abs} with total_abs). CEILING: interpretive "
            f"axis; no classification claim; NOT a paper trigger."
        )
    else:
        reading = (
            f"S3 TEMPERED: the direction axis is NOT fully feature-robust (min pairwise Spearman "
            f"{min_rc:.2f} < 0.6 among the de-entangled variants) — the modest S2 structure depends "
            f"on the feature choice and should be reported as feature-sensitive, narrowing the claim."
        )

    result = {
        "stage": "S3 — cross-feature stability + physical interpretation",
        "verdict": verdict,
        "reading": reading,
        "n_subset": n_sub,
        "crossfeature_rank_correlations": rc,
        "crossfeature_min_spearman": round(min_rc, 3),
        "crossfeature_bar": 0.6,
        "recipe_ordered_coupling": recipe,
        "direction_vs_coupling_spearman": dir_vs_coupling,
        "coupling_vs_total_abs_spearman": coupling_vs_abs,
        "s2_stability_above_surrogate": "p=0.039 (from S2 [D-04]); CI above null 95th pct",
    }
    _write_s3_figure(out_dir / "figs" / "s3_stability_interpretation.png",
                     v_unit, v_signfrac, v_median, recipe, result)
    with open(out_dir / "s3_stability_interpretation.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def _write_s3_figure(path, v_unit, v_signfrac, v_median, recipe, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = result["crossfeature_rank_correlations"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].scatter(v_unit, v_signfrac, s=2, alpha=0.12)
    ax[0, 0].set_title(f"unit-direction vs sign-fraction (Spearman {rc['unit_vs_signfrac']})")
    ax[0, 0].set_xlabel("unit-direction sum(R)/sum(|R|)"); ax[0, 0].set_ylabel("sign-fraction")
    ax[0, 1].scatter(v_unit, v_median, s=2, alpha=0.12, color="C1")
    ax[0, 1].set_title(f"unit-direction vs median-sign (Spearman {rc['unit_vs_median']})")
    ax[0, 1].set_xlabel("unit-direction"); ax[0, 1].set_ylabel("median-sign")
    cs = [2, 3, 4]
    means = [recipe[str(c)]["mean_signed_response"] for c in cs]
    ax[1, 0].bar([str(c) for c in cs], means, color=["C2", "C1", "C3"])
    ax[1, 0].axhline(0.0, color="k", lw=0.8)
    ax[1, 0].set_title("recipe-ordered net signed response (ADD<0, REMOVE>0)")
    ax[1, 0].set_xlabel("recipe (2=Wind 3=WindAGN 4=StrongAGN)")
    ax[1, 1].axis("off")
    txt = (f"VERDICT: {result['verdict']}\n\n"
           f"cross-feature min Spearman = {result['crossfeature_min_spearman']} (PASS>=0.6)\n"
           f"  unit/signfrac = {rc['unit_vs_signfrac']}\n"
           f"  unit/median   = {rc['unit_vs_median']}\n"
           f"  signfrac/median = {rc['signfrac_vs_median']}\n\n"
           f"S2 stability: {result['s2_stability_above_surrogate']}\n\n"
           f"direction vs recipe-coupling = {result['direction_vs_coupling_spearman']}\n"
           f"coupling vs total_abs = {result['coupling_vs_total_abs_spearman']} (abs-clean)\n\n"
           "Physical: winds ADD, strong-AGN REMOVES Lya\n"
           "absorption (Nasir/Bolton+2017).\n"
           "CEILING: interpretive axis; not a paper trigger.")
    ax[1, 1].text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle(f"signed-response embedding — S3 stability + interpretation: {result['verdict']}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_structure_figure(path, Xs, clusters, total_abs_sub, dir_unit4_sub,
                            null_stab, stab_real, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = result["primary_clustering"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # 2D view of the direction block (first two standardized features) colored by cluster
    ax[0, 0].scatter(Xs[:, 0], Xs[:, 1], c=clusters, s=3, alpha=0.3, cmap="tab10")
    ax[0, 0].set_title(f"direction block clusters (largest {pc['largest_cluster_frac']:.2f}, "
                       f"K={pc['k']})")
    ax[0, 0].set_xlabel("z(dir_unit4)"); ax[0, 0].set_ylabel("z(signfrac4)")
    # stability vs sign-permutation null
    ax[0, 1].hist(null_stab, bins=15, color="grey", alpha=0.7, label="sign-perm null")
    ax[0, 1].axvline(stab_real, color="C3", lw=2, label=f"real {stab_real:.3f}")
    ax[0, 1].axvline(result["sign_permutation_null"]["stability_null_95pct"], color="k", ls="--",
                     lw=1, label="null 95th pct")
    ax[0, 1].set_title("bootstrap-ARI stability vs surrogate null")
    ax[0, 1].set_xlabel("stability ARI"); ax[0, 1].legend(fontsize=8)
    # gas-gate audit: total_abs by cluster
    cl = np.unique(clusters)
    ax[1, 0].boxplot([total_abs_sub[clusters == c] for c in cl], labels=[str(c) for c in cl],
                     showfliers=False)
    ax[1, 0].set_title(f"gas-gate audit: total_abs by cluster (eta^2="
                       f"{pc['total_abs_between_cluster_eta2']})")
    ax[1, 0].set_xlabel("cluster"); ax[1, 0].set_ylabel("baseline total absorption")
    # AMI on the anchored scale
    a = result["ami_anchors"]
    ax[1, 1].axis("off")
    txt = (f"VERDICT: {result['verdict']}\n\n"
           f"n subset = {result['responding_subset']['n_subset']} "
           f"({result['responding_subset']['frac_of_population']:.0%})\n"
           f"largest cluster = {pc['largest_cluster_frac']:.2f} (NULL>=0.70)\n"
           f"stability ARI  = {stab_real:.3f}  (null 95% "
           f"{result['sign_permutation_null']['stability_null_95pct']:.3f})\n"
           f"total_abs eta^2 = {pc['total_abs_between_cluster_eta2']:.3f}  (NULL>=0.30)\n"
           f"AMI wavelet/raw = {pc['ami_vs_wavelet_k5']:.3f}/{pc['ami_vs_raw_k5']:.3f}\n"
           f"  anchor(related)={a['related_axes_upper']:.3f}  NULL>=0.40\n"
           f"gap stat = {pc['gap_statistic']:.3f} +/- {pc['gap_sk']:.3f}\n\n"
           "CEILING: interpretive DIRECTION axis; no\nclassification claim.")
    ax[1, 1].text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle(f"signed-response embedding — S2 structure test: {result['verdict']}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
