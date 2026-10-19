"""
Voigt-profile decomposition of Lyα-forest sightlines (eigen-voigt-embedding track).

Scientific purpose
------------------
Decomposes each Sherwood z=0.3 sightline into a catalog of HI Lyα absorbers, each a
Voigt profile with (line-centre velocity v, Doppler parameter b, column density
log10 N_HI), and condenses the catalog into a fixed-length per-sightline vector of
line statistics (line counts, b and log N distributions, the joint (b, log N)
occupancy, line spacing). b carries thermal + turbulent broadening and N_HI the
neutral-gas content, the two physical channels through which feedback heating and
gas redistribution act on the forest (Nasir+2017; Tillman+2023).

The fitter is the VoigtFit library (Krogager 2018, lmfit least squares), driven by an
automatic, residual-seeded component search: components are seeded at every
optical-depth peak above `tau_min`, fitted jointly, and new components are added at
every remaining residual absorption peak until max |F_model - F| < `tol` or
`max_iter` refinements.

Data facts this module relies on (verified 2026-10-18, see the track LEDGER §4):
  * Sherwood `_inf` flux is NOISELESS with no LSF: flux == roll(exp(-tau), 1) to
    machine precision. Convergence is therefore set by an explicit residual
    tolerance (an analysis choice, recorded per run), not by a noise floor.
  * Sightlines are periodic, so each spectrum is rolled to put its most transmissive
    pixel at the array edge before fitting; component velocities are mapped back.
  * VoigtFit's Tepper-Garcia H(a, x) evaluates 0/0 at x = 0 exactly, so seeded line
    centres are offset by a fixed sub-pixel amount to stay off the (sub)sampled grid.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.special import wofz

# HI Lya constants exactly as in VoigtFit's line list (lambda0 stored there as float32), so
# the fitter, the residual check and the velocity mapping share one convention (NIST
# 1215.6701 Å / f = 0.4164 would shift v by 0.12 km/s and log N by 4e-4 dex).
LYA_REST_A = float(np.float32(1215.6696))   # Å
LYA_F_OSC = 0.416
LYA_GAMMA = 6.265e8           # s^-1
C_CMS = 2.99792458e10
_E_CGS = 4.8032e-10
_ME_G = 9.1094e-28
_TAU0_COEF = 1.497e-15        # tau0 = coef * N f lambda0[Å] / b[km/s]
_TAU_SAT = 5.0                # tau_eff above which a pixel counts as saturated for seeding
_TAU_CLIP = 30.0              # tau_eff clip for the saturated-segment curve-of-growth seed
_SAT_B_GRID = (10.0, 20.0, 30.0, 45.0, 70.0)   # km/s

CATALOG_COLUMNS = ("sightline", "v_kms", "b_kms", "logN")
DIAG_KEYS = ("max_resid", "rms_resid", "n_iter", "converged", "n_fit_errors", "max_resid_region",
             "n_regions")


@dataclass
class VoigtFitConfig:
    """Fitter settings. All are analysis choices recorded with every catalog."""
    tau_min: float = 0.05           # seed threshold on tau_eff = -ln F
    prominence_frac: float = 0.5    # seed peak prominence = prominence_frac * tau_min
    tol: float = 0.01               # convergence: max |F_model - F| over the sightline
    max_iter: int = 15              # residual-refinement rounds
    res_kms: float = 0.0            # VoigtFit LSF FWHM; 0 = point-sampled, no LSF (true for Sherwood)
    logN_min: float = 11.0
    logN_max: float = 23.0          # Sherwood tau reaches ~1e10 (DLA-strength columns)
    b_min: float = 3.0
    b_max: float = 300.0
    z_jitter_px: float = 0.37       # sub-pixel seed offset (Tepper-Garcia x=0 singularity)
    region_thresh: float = 0.005    # 1 - F above which a pixel belongs to an absorption region
    region_pad: int = 5             # pixels added on each side of a region before merging


def tau_voigt(wave: np.ndarray, z: float, b_kms: float, logN: float) -> np.ndarray:
    """Exact HI Lyα optical depth of one Voigt component via the Faddeeva function."""
    b = b_kms * 1e5
    x = (C_CMS / b) * (1.0 - LYA_REST_A / (wave / (1.0 + z)))
    a = LYA_REST_A * 1e-8 * LYA_GAMMA / (4.0 * np.pi * b)
    c_a = np.sqrt(np.pi) * _E_CGS ** 2 * LYA_F_OSC * LYA_REST_A * 1e-8 / _ME_G / C_CMS / b
    return c_a * 10.0 ** logN * wofz(x + 1j * a).real   # shape: (n_pix,)


_X_EXACT = 10.0              # |x| below which the profile is evaluated exactly


def _hybrid_voigt(wl: np.ndarray, l0: float, f: float, N: float, b: float, gam: float,
                  z: float = 0.0) -> np.ndarray:
    """Drop-in for VoigtFit.voigt.Voigt (same signature, b in cm/s).

    VoigtFit's Tepper-Garcia H(a, x) is 0/0 at x = 0 and wrong by factors up to 1e12 for
    |x| < 1e-4 and by 1-3% for 1 < |x| < 5 (measured 2026-10-18). The spikes near line
    centre crash scipy.optimize.leastsq (MINPACK segfault, class-2 sightline 7307). The
    exact Faddeeva function is used for |x| < _X_EXACT; Tepper-Garcia (relative error
    < 4e-4 there) is kept for the far wings, where wofz would cost 2x for no gain.
    """
    x = (C_CMS / b) * (1.0 - l0 / (wl / (z + 1.0)))
    a = l0 * 1e-8 * gam / (4.0 * np.pi * b)
    c_a = np.sqrt(np.pi) * _E_CGS ** 2 * f * l0 * 1e-8 / _ME_G / C_CMS / b
    H = np.empty_like(x)
    core = np.abs(x) < _X_EXACT
    H[core] = wofz(x[core] + 1j * a).real
    xw = x[~core]
    P = xw ** 2
    H0 = np.exp(-P)
    Q = 1.5 / P
    H[~core] = H0 - a / np.sqrt(np.pi) / P * (H0 * H0 * (4.0 * P * P + 7.0 * P + 4.0 + Q) - Q - 1.0)
    return c_a * N * H


def _install_vf_patches() -> None:
    """Patch VoigtFit (idempotent): hybrid profile, and a point-sampled no-LSF path.

    With a float resolution VoigtFit evaluates the model on its own padded log grid
    (3x subsampled) and linearly interpolates back to the data, which shifts sharp lines
    by ~0.13 km/s and triples the cost. Sherwood tau is point-sampled at the pixel
    centres with no LSF, so for res <= 0 the exact forward model is exp(-tau) evaluated
    on the data grid itself.
    """
    import VoigtFit.dataset as vf_dataset
    import VoigtFit.voigt as vf_voigt
    vf_voigt.Voigt = _hybrid_voigt
    if getattr(vf_dataset.evaluate_profile, "_cgp_point_sampled", False):
        return
    # VoigtFit builds Minimizer(..., nan_policy='omit'): dropping non-finite residuals
    # shrinks the array MINPACK sized its work buffers for, and scipy's C MINPACK then
    # segfaults (reproduced on class-2 sightline 7307). 'raise' turns it into a
    # ValueError that the component search handles.
    base_minimizer = vf_dataset.Minimizer

    class _RaisingMinimizer(base_minimizer):
        def __init__(self, *args, **kwargs):
            kwargs["nan_policy"] = "raise"
            super().__init__(*args, **kwargs)

    vf_dataset.Minimizer = _RaisingMinimizer
    original = vf_dataset.evaluate_profile

    def evaluate_profile(x, pars, lines, kernel, z_sys=None, sampling=3, kernel_nsub=1):
        if isinstance(kernel, float) and kernel <= 0.0:
            return np.exp(-vf_voigt.evaluate_optical_depth(x, pars, lines, z_sys=z_sys))
        return original(x, pars, lines, kernel, z_sys=z_sys, sampling=sampling, kernel_nsub=kernel_nsub)

    evaluate_profile._cgp_point_sampled = True
    vf_dataset.evaluate_profile = evaluate_profile


def model_flux(wave: np.ndarray, comps: List[Tuple[float, float, float]]) -> np.ndarray:
    """Transmitted flux of a list of (z, b_kms, logN) components."""
    tau = np.zeros_like(wave, dtype=np.float64)
    for z, b, logN in comps:
        tau += tau_voigt(wave, z, b, logN)
    return np.exp(-tau)


class VoigtSightlineFitter:
    """Automatic VoigtFit decomposition of one periodic, noiseless sightline.

    The rolled sightline is split into independent absorption regions — maximal runs of
    pixels with 1 - F > `region_thresh`, padded by `region_pad` pixels and merged — and
    each region is decomposed on its own (VPFIT/AutoVP practice): absorbers separated by
    transmitted flux do not interact, and joint fits of ~80 components over the whole
    sightline both cost O(n_par^2) and drove scipy's MINPACK into segfaults.
    """

    def __init__(self, wave: np.ndarray, vel: np.ndarray,
                 config: Optional[VoigtFitConfig] = None) -> None:
        wave = np.asarray(wave, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        if wave.ndim != 1 or wave.shape != vel.shape or not np.all(np.diff(wave) > 0):
            raise ValueError("VoigtSightlineFitter: wave/vel must be matching increasing 1-D grids")
        self.wave = wave                                    # shape: (n_pix,)
        self.vel = vel                                      # shape: (n_pix,)
        self.cfg = config or VoigtFitConfig()
        self.n_pix = wave.size
        self.dv = float(vel[1] - vel[0])
        self.dlnw = float(np.log(wave[1] / wave[0]))
        z_mid = float(wave[self.n_pix // 2] / LYA_REST_A - 1.0)
        self._zjit = self.cfg.z_jitter_px * self.dlnw * (1.0 + z_mid)

    def _z_of_pixel(self, p: int) -> float:
        return float(self.wave[p] / LYA_REST_A - 1.0) + self._zjit

    def _regions(self, f: np.ndarray) -> List[Tuple[int, int]]:
        cfg = self.cfg
        absorbed = (1.0 - f) > cfg.region_thresh
        if not absorbed.any():
            return []
        k = np.ones(2 * cfg.region_pad + 1)
        grown = np.convolve(absorbed.astype(np.float64), k, mode="same") > 0
        edges = np.flatnonzero(np.diff(np.concatenate([[0], grown.astype(np.int8), [0]])))
        return [(int(lo), int(hi)) for lo, hi in zip(edges[::2], edges[1::2])]

    def _seed(self, tau_eff: np.ndarray, lo: int, hi: int) -> List[Tuple[float, float, float]]:
        cfg = self.cfg
        t = tau_eff[lo:hi]
        idx, _ = find_peaks(t, height=cfg.tau_min, prominence=cfg.prominence_frac * cfg.tau_min)
        sat = t > _TAU_SAT
        seeds = []
        if idx.size:
            widths = peak_widths(t, idx, rel_height=0.5)[0]
            for p, w_px in zip(idx, widths):
                if sat[p]:
                    continue                                # saturated cores are seeded per segment
                b = float(np.clip(w_px * self.dv / 1.665, 8.0, 150.0))      # FWHM = 2 sqrt(ln2) b
                n_col = t[p] * b / (_TAU0_COEF * LYA_F_OSC * LYA_REST_A)
                logN = float(np.clip(np.log10(max(n_col, 1e11)), cfg.logN_min + 0.5, cfg.logN_max - 1.0))
                seeds.append((self._z_of_pixel(lo + int(p)), b, logN))
        # one seed per contiguous saturated segment: the flat noiseless floor has no usable
        # peak structure, so (b, log N) come from a curve-of-growth grid search matching the
        # region's tau_eff clipped at _TAU_CLIP (covers Doppler- and damping-dominated widths)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], sat.astype(np.int8), [0]])))
        wr = self.wave[lo:hi]
        t_clip = np.minimum(t, _TAU_CLIP)
        for s0, s1 in zip(edges[::2], edges[1::2]):
            z0 = self._z_of_pixel(lo + int((s0 + s1 - 1) // 2))
            best, best_err = (25.0, 15.0), np.inf
            for b in _SAT_B_GRID:
                for logN in np.arange(13.0, cfg.logN_max - 0.25, 0.1):
                    m = np.minimum(tau_voigt(wr, z0, b, logN), _TAU_CLIP)
                    err = float(np.sum((m - t_clip) ** 2 * (m > t_clip)) + np.sum((m - t_clip)[s0:s1] ** 2))
                    if err < best_err:
                        best, best_err = (b, float(logN)), err
            seeds.append((z0, best[0], best[1]))
        if not seeds:                                       # absorbed but no peak above tau_min
            p = int(np.argmax(t))
            seeds.append((self._z_of_pixel(lo + p), 15.0, 12.0))
        return seeds

    def _refine_seeds(self, comps: List[Tuple[float, float, float]], r: np.ndarray, lo: int,
                      single: bool) -> List[Tuple[float, float, float]]:
        """Add components at residual absorption peaks (all peaks, or only the worst pixel)."""
        cfg = self.cfg
        if single:
            pk = np.array([int(np.argmax(r))])
        else:
            pk, _ = find_peaks(np.clip(r, 0.0, None), height=cfg.tol)
            if pk.size == 0:
                pk = np.array([int(np.argmax(np.abs(r)))])
        trial = list(comps)
        for p in pk:
            n_col = max(float(r[p]), 1e-3) * 15.0 / (_TAU0_COEF * LYA_F_OSC * LYA_REST_A)
            trial.append((self._z_of_pixel(lo + int(p)), 15.0,
                          float(np.clip(np.log10(n_col), cfg.logN_min + 0.5, cfg.logN_max - 1.0))))
        return trial

    def _voigtfit(self, f: np.ndarray, lo: int, hi: int, comps: List[Tuple[float, float, float]]
                  ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """One joint VoigtFit fit of region [lo, hi); returns fitted components and
        VoigtFit's own model-minus-data flux residual over the region."""
        import VoigtFit                                     # heavy import; worker-local
        _install_vf_patches()
        cfg = self.cfg
        wr, fr = self.wave[lo:hi], f[lo:hi]
        half_px = 0.5 * self.dlnw                           # bounds off the pixel grid
        z_lo = float(wr[0] * np.exp(half_px) / LYA_REST_A - 1.0)
        z_hi = float(wr[-1] * np.exp(-half_px) / LYA_REST_A - 1.0)
        ds = VoigtFit.DataSet(float(np.sqrt(wr[0] * wr[-1]) / LYA_REST_A - 1.0))
        ds.verbose = False
        ds.add_data(wr, fr, cfg.res_kms, err=np.full_like(fr, cfg.tol), normalized=True)
        ds.add_line("HI_1215", velspan=0.5 * (hi - lo) * self.dv + 3.0 * self.dv)
        for z, b, logN in comps:
            ds.add_component("HI", float(np.clip(z, z_lo, z_hi)), b, logN)
        ds.prepare_dataset(norm=False, mask=False, verbose=False)
        for k in range(len(comps)):
            ds.pars[f"logN{k}_HI"].set(min=cfg.logN_min, max=cfg.logN_max)
            ds.pars[f"b{k}_HI"].set(min=cfg.b_min, max=cfg.b_max)
            ds.pars[f"z{k}_HI"].set(min=z_lo, max=z_hi)
        popt, _ = ds.fit(verbose=False)
        p = ds.best_fit
        fitted = [(p[f"z{k}_HI"].value, p[f"b{k}_HI"].value, p[f"logN{k}_HI"].value)
                  for k in range(len(comps))]
        resid = -np.asarray(popt.residual, dtype=np.float64) * cfg.tol    # chi = (data - model) / err
        if resid.shape != fr.shape or not np.isfinite(resid).all():
            resid = model_flux(wr, fitted) - fr
        return fitted, resid

    def _search(self, f: np.ndarray, lo: int, hi: int, trial: List[Tuple[float, float, float]],
                st: Dict[str, object]) -> None:
        """Greedy residual-seeded component search on one region; updates `st` in place.

        A refinement is kept only if it lowers max |residual|; when adding a component at
        every residual peak fails, one component at the worst pixel is tried before
        stopping."""
        cfg = self.cfg
        single = False
        n_err = 0
        for _ in range(cfg.max_iter):
            st["n_iter"] += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fitted, r = self._voigtfit(f, lo, hi, trial)   # r > 0: absorption missing
                except (ValueError, OverflowError, FloatingPointError):
                    n_err += 1
                    st["n_err"] += 1
                    if n_err > 2 or len(trial) <= max(len(st["comps"]), 1):
                        return
                    trial = trial[:-1]
                    continue
            new_max = float(np.abs(r).max())
            if st["comps"] and new_max >= st["max_res"]:
                if single:
                    return
                single = True
                trial = self._refine_seeds(st["comps"], st["best_r"], lo, single=True)
                continue
            st["comps"], st["max_res"], st["best_r"] = fitted, new_max, r
            single = False
            if st["max_res"] < cfg.tol:
                return
            trial = self._refine_seeds(st["comps"], st["best_r"], lo, single=False)

    def fit(self, flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Decompose one sightline.

        Returns
        -------
        comps : (n_comp, 3) array of [v_kms, b_kms, logN] in the ORIGINAL (unrolled) frame.
        diag  : DIAG_KEYS. Residuals are in flux units over the whole sightline from the
                exact model of all components; converged = max_resid < tol.
        """
        flux = np.asarray(flux, dtype=np.float64)
        if flux.shape != (self.n_pix,) or not np.isfinite(flux).all():
            raise ValueError("VoigtSightlineFitter.fit: flux must be finite with shape (n_pix,)")
        if flux.min() < -1e-12 or flux.max() > 1.0 + 1e-9:     # stored flux has ~-5e-16 round-off
            raise ValueError("VoigtSightlineFitter.fit: flux outside [0, 1]")
        flux = np.clip(flux, 0.0, 1.0)
        cfg = self.cfg
        shift = int(np.argmax(flux))
        f = np.roll(flux, -shift)
        tau_eff = -np.log(np.clip(f, 1e-300, None))
        comps: List[Tuple[float, float, float]] = []
        n_err = n_iter = 0
        regions = self._regions(f)
        worst_region = 0.0
        for lo, hi in regions:
            st = {"comps": [], "max_res": float(np.max(1.0 - f[lo:hi])), "best_r": None,
                  "n_err": 0, "n_iter": 0}
            self._search(f, lo, hi, self._seed(tau_eff, lo, hi), st)
            comps.extend(st["comps"])
            n_err += st["n_err"]
            n_iter += st["n_iter"]
            worst_region = max(worst_region, st["max_res"])
        r_all = model_flux(self.wave, comps) - f if comps else 1.0 - f
        max_res = float(np.abs(r_all).max())
        out = np.empty((len(comps), 3), dtype=np.float64)   # shape: (n_comp, 3)
        span = self.n_pix * self.dv
        for i, (z, b, logN) in enumerate(comps):
            pix = np.log((1.0 + z) * LYA_REST_A / self.wave[0]) / self.dlnw
            out[i] = (((pix + shift) * self.dv) % span, b, logN)
        diag = {"max_resid": max_res, "rms_resid": float(np.sqrt(np.mean(r_all ** 2))),
                "n_iter": float(n_iter), "converged": float(max_res < cfg.tol),
                "n_fit_errors": float(n_err), "max_resid_region": worst_region,
                "n_regions": float(len(regions))}
        return out, diag


_WORKER: Optional[VoigtSightlineFitter] = None


def _init_worker(wave: np.ndarray, vel: np.ndarray, cfg_dict: Dict[str, float]) -> None:
    global _WORKER
    os.environ.setdefault("MPLBACKEND", "Agg")
    _WORKER = VoigtSightlineFitter(wave, vel, VoigtFitConfig(**cfg_dict))


def _fit_block(args: Tuple[int, np.ndarray]) -> Tuple[int, List[np.ndarray], List[Dict[str, float]]]:
    start, block = args
    comps, diags = [], []
    for row in block:
        c, d = _WORKER.fit(row)
        comps.append(c)
        diags.append(d)
    return start, comps, diags


def _crash_row(flux_row: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Placeholder for a sightline whose fit killed its process (n_fit_errors = -1)."""
    d = {k: 0.0 for k in DIAG_KEYS}
    d["max_resid"] = d["max_resid_region"] = float(1.0 - np.min(flux_row))
    d["n_fit_errors"] = -1.0
    return np.zeros((0, 3), dtype=np.float64), d


def _fit_isolated(row: np.ndarray, initargs: Tuple) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit one sightline in its own process so a native crash cannot take others with it."""
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker, initargs=initargs) as one:
        try:
            _, comps, diags = one.submit(_fit_block, (0, row[None, :])).result()
            return comps[0], diags[0]
        except BrokenProcessPool:
            return _crash_row(row)


def _run_chunk(jobs: Dict[int, np.ndarray], initargs: Tuple, n_jobs: int,
               pool_box: List[ProcessPoolExecutor]) -> Dict[int, Tuple[np.ndarray, Dict[str, float]]]:
    """Fit all blocks of one chunk, surviving worker crashes.

    On a broken pool the pool is rebuilt and only unfinished sightlines are re-queued,
    one per task; after three breaks the remainder is fitted one process per sightline.
    """
    rows: Dict[int, Tuple[np.ndarray, Dict[str, float]]] = {}
    pending = dict(jobs)                                    # start index -> (m, n_pix) block
    breaks = 0
    while pending:
        if breaks >= 3:
            for s, blk in sorted(pending.items()):
                for j, row in enumerate(blk):
                    rows[s + j] = _fit_isolated(row, initargs)
            return rows
        futs = {pool_box[0].submit(_fit_block, (s, blk)): s for s, blk in pending.items()}
        try:
            for fut in as_completed(futs):
                start, comps, diags = fut.result()
                for j, (c, d) in enumerate(zip(comps, diags)):
                    rows[start + j] = (c, d)
                pending.pop(start)
        except BrokenProcessPool:
            breaks += 1
            for fut, s in futs.items():
                if s in pending and fut.done() and not fut.cancelled() and fut.exception() is None:
                    start, comps, diags = fut.result()
                    for j, (c, d) in enumerate(zip(comps, diags)):
                        rows[start + j] = (c, d)
                    pending.pop(start)
            pending = {s + j: blk[j:j + 1] for s, blk in pending.items() for j in range(blk.shape[0])}
            pool_box[0].shutdown(wait=False, cancel_futures=True)
            pool_box[0] = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=initargs)
            print(f"[voigt] worker crash #{breaks}; re-queued {len(pending)} sightlines", flush=True)
    return rows


def fit_catalog(flux: np.ndarray, wave: np.ndarray, vel: np.ndarray, out_path: Path,
                config: Optional[VoigtFitConfig] = None, n_jobs: int = 8,
                chunk: int = 512, block: int = 16) -> Dict[str, np.ndarray]:
    """Decompose every sightline of `flux` (n, n_pix) and write a resumable catalog.

    Chunks of `chunk` sightlines are checkpointed as `<out_path>.partNNNN.npz`, so an
    interrupted run resumes; the merged catalog is written to `out_path` (.npz) with
    columns CATALOG_COLUMNS plus per-sightline diagnostics (DIAG_KEYS).
    """
    flux = np.asarray(flux)                                 # shape: (n_sightlines, n_pix)
    if flux.ndim != 2 or not np.isfinite(flux).all():
        raise ValueError("fit_catalog: flux must be a finite 2-D array")
    cfg = config or VoigtFitConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = flux.shape[0]
    starts = list(range(0, n, chunk))
    initargs = (np.asarray(wave), np.asarray(vel), asdict(cfg))
    pool_box = [ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=initargs)]
    try:
        for ci, s0 in enumerate(starts):
            part = out_path.with_suffix(f".part{ci:04d}.npz")
            if part.exists():
                continue
            s1 = min(s0 + chunk, n)
            jobs = {s: np.asarray(flux[s:min(s + block, s1)], dtype=np.float64) for s in range(s0, s1, block)}
            rows = _run_chunk(jobs, initargs, n_jobs, pool_box)
            _save_part(part, rows, s0, s1)
            print(f"[voigt] {out_path.name}: sightlines {s0}-{s1 - 1} done ({ci + 1}/{len(starts)})", flush=True)
    finally:
        pool_box[0].shutdown(wait=True)
    return merge_parts(out_path, n_parts=len(starts), config=cfg)


def _save_part(part: Path, rows: Dict[int, Tuple[np.ndarray, Dict[str, float]]], s0: int, s1: int) -> None:
    idx, comps = [], []
    diag_keys = DIAG_KEYS
    diag = np.zeros((s1 - s0, len(diag_keys)), dtype=np.float64)
    for i in range(s0, s1):
        c, d = rows[i]
        idx.append(np.full(len(c), i, dtype=np.int64))
        comps.append(c)
        diag[i - s0] = [d[k] for k in diag_keys]
    np.savez(part, sightline=np.concatenate(idx) if idx else np.zeros(0, np.int64),
             comps=np.vstack(comps) if comps else np.zeros((0, 3)), diag=diag,
             diag_keys=np.array(diag_keys), s0=s0, s1=s1)


def merge_parts(out_path: Path, n_parts: int, config: VoigtFitConfig) -> Dict[str, np.ndarray]:
    """Merge checkpoint parts into `out_path` and remove the parts."""
    out_path = Path(out_path)
    parts = [np.load(out_path.with_suffix(f".part{ci:04d}.npz")) for ci in range(n_parts)]
    sl = np.concatenate([p["sightline"] for p in parts])
    comps = np.vstack([p["comps"] for p in parts])          # shape: (n_comp_total, 3)
    diag = np.vstack([p["diag"] for p in parts])            # shape: (n_sightlines, len(DIAG_KEYS))
    cat = {"sightline": sl, "v_kms": comps[:, 0], "b_kms": comps[:, 1], "logN": comps[:, 2],
           "diag": diag, "diag_keys": parts[0]["diag_keys"],
           "config_keys": np.array(list(asdict(config).keys())),
           "config_vals": np.array(list(asdict(config).values()), dtype=np.float64)}
    np.savez_compressed(out_path, **cat)
    for ci in range(n_parts):
        out_path.with_suffix(f".part{ci:04d}.npz").unlink()
    return cat


# ---------------------------------------------------------------------------
# Per-sightline line-statistics features
# ---------------------------------------------------------------------------
_B_EDGES = (0.0, 20.0, 35.0, 60.0, np.inf)
_N_EDGES = (12.0, 12.75, 13.5, np.inf)


def line_feature_names() -> List[str]:
    names = ["n_lines_12", "n_lines_13", "n_lines_14", "logNtot",
             "b_q10", "b_q25", "b_q50", "b_q75", "b_q90", "b_Nweighted",
             "frac_b_gt40", "frac_b_lt15",
             "logN_q25", "logN_q50", "logN_q75", "logN_q90",
             "bN_slope", "dv_nn_median", "has_lines"]
    for i in range(len(_B_EDGES) - 1):
        for j in range(len(_N_EDGES) - 1):
            names.append(f"occ_b{i}_N{j}")
    return names


def line_statistics_features(sightline: np.ndarray, v_kms: np.ndarray, b_kms: np.ndarray,
                             logN: np.ndarray, n_sightlines: int, span_kms: float,
                             logN_detect: float = 12.0) -> np.ndarray:
    """Fixed-length line statistics per sightline from a flattened absorber catalog.

    Only components with logN >= `logN_detect` enter the distribution features (weaker
    components are residual-fill terms of the decomposition); logNtot sums ALL
    components. Sightlines with < 1 (quantiles) or < 3 (b-N slope) detected lines get
    the population median of that feature among sightlines that have it, flagged by
    `has_lines` = 0 — a documented imputation, never NaN.
    """
    names = line_feature_names()
    X = np.full((n_sightlines, len(names)), np.nan, dtype=np.float64)   # shape: (n, n_feat)
    order = np.argsort(sightline, kind="stable")
    sl, v, b, N = sightline[order], v_kms[order], b_kms[order], logN[order]
    bounds = np.searchsorted(sl, np.arange(n_sightlines + 1))
    col = {k: i for i, k in enumerate(names)}
    for i in range(n_sightlines):
        lo, hi = bounds[i], bounds[i + 1]
        Ni_all = N[lo:hi]
        X[i, col["logNtot"]] = np.log10(np.sum(10.0 ** Ni_all)) if hi > lo else 0.0
        m = Ni_all >= logN_detect
        bi, Ni, vi = b[lo:hi][m], Ni_all[m], v[lo:hi][m]
        X[i, col["n_lines_12"]] = m.sum()
        X[i, col["n_lines_13"]] = np.sum(Ni_all >= 13.0)
        X[i, col["n_lines_14"]] = np.sum(Ni_all >= 14.0)
        X[i, col["has_lines"]] = float(bi.size > 0)
        for jb in range(len(_B_EDGES) - 1):
            for jn in range(len(_N_EDGES) - 1):
                X[i, col[f"occ_b{jb}_N{jn}"]] = np.sum((bi >= _B_EDGES[jb]) & (bi < _B_EDGES[jb + 1])
                                                      & (Ni >= _N_EDGES[jn]) & (Ni < _N_EDGES[jn + 1]))
        if bi.size == 0:
            continue
        X[i, [col["b_q10"], col["b_q25"], col["b_q50"], col["b_q75"], col["b_q90"]]] = \
            np.percentile(bi, [10, 25, 50, 75, 90])
        w = 10.0 ** Ni
        X[i, col["b_Nweighted"]] = np.sum(w * bi) / np.sum(w)
        X[i, col["frac_b_gt40"]] = np.mean(bi > 40.0)
        X[i, col["frac_b_lt15"]] = np.mean(bi < 15.0)
        X[i, [col["logN_q25"], col["logN_q50"], col["logN_q75"], col["logN_q90"]]] = \
            np.percentile(Ni, [25, 50, 75, 90])
        if bi.size >= 2:
            vs = np.sort(vi)
            gaps = np.diff(np.concatenate([vs, [vs[0] + span_kms]]))   # periodic sightline
            X[i, col["dv_nn_median"]] = np.median(gaps)
        if bi.size >= 3 and np.ptp(Ni) > 0:
            X[i, col["bN_slope"]] = np.polyfit(Ni, bi, 1)[0]
    for k in range(X.shape[1]):
        miss = ~np.isfinite(X[:, k])
        if miss.any():
            fill = np.median(X[~miss, k]) if (~miss).any() else 0.0
            X[miss, k] = fill
    return X
