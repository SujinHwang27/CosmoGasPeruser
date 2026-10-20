"""
Forward-modelled (COS-like) Lyman-alpha spectra and their Voigt decomposition (lit-claims-audit T2).

Scientific purpose
------------------
Tests whether the noiseless line-width differences between feedback classes survive an
observation-like forward model, which is the regime published low-z feedback claims are
made in (Khaire+2024a: perfect-skewer b-N differences "wash out" in forward models;
Nasir+2017: forward modelling moves median b by ~8 km/s). Operator, applied to the native
noiseless flux F (2048 px, 2.6365 km/s, periodic):

  1. periodic convolution with the line-spread function (tabulated COS LSF or Gaussian),
     optionally an extra Gaussian 'b_turb' broadening drawn per sightline (nuisance);
  2. roll so the most transmissive pixel of the convolved spectrum is at the edge (no
     absorber straddles the cut), trim to 2046 px, rebin by 3 -> 682 px (7.91 km/s);
  3. multiplicative continuum error 1 + sum_{k<=2} c_k P_k(x), c_k ~ N(0, cont_sigma);
  4. Gaussian noise, sigma = 1/SNR (flux-independent) or sqrt(F)/SNR.

The fitter evaluates exactly the same operator inside VoigtFit (fine-grid tau -> exp ->
LSF -> rebin) through the evaluate_profile patch in src/core/voigt.py, with error-weighted
chi^2, noise-significance seeding/regions and AICc component acceptance.
"""

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_widths

from src.core.voigt import (LYA_F_OSC, LYA_REST_A, _TAU0_COEF, _install_vf_patches, model_flux, tau_voigt)

C_KMS = 299792.458


def gaussian_lsf(fwhm_kms: float, dv_kms: float) -> np.ndarray:
    """Normalised Gaussian kernel on the native pixel grid (odd length, +-4 sigma)."""
    sig = fwhm_kms / 2.3548 / dv_kms
    h = int(np.ceil(4 * sig))
    x = np.arange(-h, h + 1)
    k = np.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum()


def tabulated_lsf(path: Path, dv_kms: float, lam_obs: float = 1595.0, cos_pixel_A: float = 0.00997) -> np.ndarray:
    """COS LSF table -> kernel on the native velocity grid.

    The on-disk table (COS_LSF1_cw1327_75) has two columns: pixel offset (COS pixels,
    -159..159) and the LSF value. Offsets are mapped to velocity with the G130M dispersion
    `cos_pixel_A` at the observed wavelength (z = 0.3 Lya ~ 1595 A), exactly as the Sherwood
    spectrum generator (data/raw/Sherwood/src/utils.py) does; the kernel is interpolated onto
    multiples of dv_kms and normalised. (z = 0.3 Lya would fall in G160M; that table is not on
    disk, which LEDGER records as a forward-model caveat.)
    """
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"tabulated_lsf: expected (offset, value) columns, got shape {raw.shape}")
    off_pix, prof = raw[:, 0], raw[:, 1]
    v = off_pix * C_KMS * cos_pixel_A / lam_obs
    h = int(np.floor(np.abs(v).max() / dv_kms))
    grid = np.arange(-h, h + 1) * dv_kms
    k = np.interp(grid, v, prof, left=0.0, right=0.0)
    return k / k.sum()


@dataclass
class ObsConfig:
    snr: float = 30.0                 # per rebinned pixel; np.inf = noiseless (LSF + rebin only)
    rebin: int = 3
    n_keep: int = 2046
    noise: str = "const"              # const | sqrtF
    cont_sigma: float = 0.01          # std of each Legendre continuum coefficient (k = 0, 1, 2)
    b_turb_max: float = 0.0           # if > 0: extra Gaussian broadening b ~ U(0, b_turb_max) km/s
    b_extra: float = 0.0              # if > 0: fixed extra Gaussian broadening (b, km/s), unknown to the fitter


class ForwardModel:
    """Observation operator on the native periodic grid (see module docstring)."""

    def __init__(self, wave: np.ndarray, dv_kms: float, lsf: np.ndarray, cfg: ObsConfig) -> None:
        if lsf.ndim != 1 or lsf.size % 2 == 0 or not np.isclose(lsf.sum(), 1.0):
            raise ValueError("ForwardModel: LSF kernel must be 1-D, odd length, normalised")
        self.wave = np.asarray(wave, dtype=np.float64)
        self.dv = dv_kms
        self.lsf = lsf
        self.cfg = cfg
        n = cfg.n_keep
        self.fine_wave = self.wave[:n]                     # native grid of the kept pixels
        self.obs_wave = self.fine_wave.reshape(-1, cfg.rebin).mean(axis=1)   # shape: (682,)

    def _convolve_periodic(self, flux: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        n = flux.size
        kp = np.zeros(n)
        h = kernel.size // 2
        kp[:h + 1] = kernel[h:]
        kp[-h:] = kernel[:h]
        return np.real(np.fft.ifft(np.fft.fft(flux) * np.fft.fft(kp)))

    def observe(self, flux: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        f = self._convolve_periodic(np.asarray(flux, dtype=np.float64), self.lsf)
        if cfg.b_extra > 0:
            f = self._convolve_periodic(f, gaussian_lsf(cfg.b_extra * 2.3548 / np.sqrt(2.0), self.dv))
        if cfg.b_turb_max > 0:
            bt = rng.uniform(0.0, cfg.b_turb_max)
            if bt > 0.5:
                f = self._convolve_periodic(f, gaussian_lsf(bt * 2.3548 / np.sqrt(2.0), self.dv))
        shift = int(np.argmax(f))
        f = np.roll(f, -shift)[:cfg.n_keep]
        f = f.reshape(-1, cfg.rebin).mean(axis=1)
        x = np.linspace(-1, 1, f.size)
        cont = 1.0 + sum(rng.normal(0.0, cfg.cont_sigma) * np.polynomial.legendre.Legendre.basis(k)(x) for k in range(3))
        f_true = f.copy()
        f = f * cont
        if np.isfinite(cfg.snr):
            sig = (np.full_like(f, 1.0 / cfg.snr) if cfg.noise == "const"
                   else np.sqrt(np.clip(f_true, 1e-3, None)) / cfg.snr)
            f = f + rng.normal(0.0, 1.0, f.size) * sig
        else:
            sig = np.full_like(f, 1e-3)
        return {"flux": f, "sigma": sig, "shift": shift}


class _FMKernel:
    """VoigtFit kernel object: evaluates the forward operator for a region of obs pixels."""

    def __init__(self, fm: ForwardModel) -> None:
        self.fm = fm
        self.pad = fm.lsf.size // 2 + fm.cfg.rebin

    def evaluate(self, x: np.ndarray, pars, lines, z_sys) -> np.ndarray:
        import VoigtFit.voigt as vf_voigt
        fm, r = self.fm, self.fm.cfg.rebin
        j0 = int(np.argmin(np.abs(fm.obs_wave - x[0])))
        j1 = int(np.argmin(np.abs(fm.obs_wave - x[-1])))
        i0, i1 = max(j0 * r - self.pad, 0), min((j1 + 1) * r + self.pad, fm.fine_wave.size)
        fw = fm.fine_wave[i0:i1]
        prof = np.exp(-vf_voigt.evaluate_optical_depth(fw, pars, lines, z_sys=z_sys))
        conv = np.convolve(np.pad(prof, fm.lsf.size // 2, mode="edge"), fm.lsf, mode="valid")
        a = j0 * r - i0
        seg = conv[a:a + (j1 - j0 + 1) * r]
        return seg.reshape(-1, r).mean(axis=1)


def _install_fm_patch() -> None:
    import VoigtFit.dataset as vf_dataset
    _install_vf_patches()
    current = vf_dataset.evaluate_profile
    if getattr(current, "_cgp_fm", False):
        return

    def evaluate_profile(x, pars, lines, kernel, z_sys=None, sampling=3, kernel_nsub=1):
        if isinstance(kernel, _FMKernel):
            return kernel.evaluate(x, pars, lines, z_sys)
        return current(x, pars, lines, kernel, z_sys=z_sys, sampling=sampling, kernel_nsub=kernel_nsub)

    evaluate_profile._cgp_fm = True
    evaluate_profile._cgp_point_sampled = True
    vf_dataset.evaluate_profile = evaluate_profile


@dataclass
class NoisyFitConfig:
    seed_sig: float = 4.0         # seed: smoothed absorption peak >= seed_sig sigma
    region_sig: float = 2.5       # region: smoothed absorption >= region_sig sigma
    region_pad: int = 2
    resid_sig: float = 3.0        # refinement: smoothed residual absorption peak >= resid_sig sigma
    d_aicc: float = -2.0          # accept a refinement only if AICc changes by < d_aicc
    max_iter: int = 10
    logN_min: float = 11.0
    logN_max: float = 23.0
    b_min: float = 3.0
    b_max: float = 300.0


class NoisyVoigtFitter:
    """VoigtFit decomposition of a forward-modelled spectrum (error-weighted, AICc search)."""

    def __init__(self, fm: ForwardModel, cfg: Optional[NoisyFitConfig] = None) -> None:
        self.fm = fm
        self.cfg = cfg or NoisyFitConfig()
        self.kernel = _FMKernel(fm)
        self.w = fm.obs_wave
        self.dlnw = float(np.log(self.w[1] / self.w[0]))
        self.dv_obs = fm.dv * fm.cfg.rebin

    def _z(self, p: float) -> float:
        return float(self.w[0] * np.exp((p + 0.37) * self.dlnw) / LYA_REST_A - 1.0)

    def _smooth(self, a: np.ndarray) -> np.ndarray:
        return np.convolve(np.pad(a, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")

    def _regions(self, f: np.ndarray, sig: np.ndarray) -> List[Tuple[int, int]]:
        s = self._smooth(1.0 - f) / (sig / np.sqrt(3.0))
        absorbed = s > self.cfg.region_sig
        if not absorbed.any():
            return []
        k = np.ones(2 * self.cfg.region_pad + 1)
        grown = np.convolve(absorbed.astype(float), k, mode="same") > 0
        e = np.flatnonzero(np.diff(np.concatenate([[0], grown.astype(np.int8), [0]])))
        return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2]) if b - a >= 3]

    def _seed(self, f: np.ndarray, sig: np.ndarray, lo: int, hi: int) -> List[Tuple[float, float, float]]:
        cfg = self.cfg
        fs = self._smooth(f)[lo:hi]
        sg = (sig / np.sqrt(3.0))[lo:hi]
        snr_abs = (1.0 - fs) / sg
        floor = np.clip(2 * sg, 1e-3, None)
        t = -np.log(np.clip(fs, floor, None))
        idx, _ = find_peaks(snr_abs, height=cfg.seed_sig, prominence=2.0)
        seeds = []
        if idx.size:
            wpx = peak_widths(t, idx, rel_height=0.5)[0]
            for p, wp in zip(idx, wpx):
                b = float(np.clip(wp * self.dv_obs / 1.665, 10.0, 150.0))
                N = t[p] * b / (_TAU0_COEF * LYA_F_OSC * LYA_REST_A)
                seeds.append((self._z(lo + p), b, float(np.clip(np.log10(max(N, 1e11)), 12.0, 21.0))))
        if not seeds:
            p = int(np.argmax(snr_abs))
            seeds.append((self._z(lo + p), 20.0, 12.8))
        return seeds

    def _fit(self, f, sig, lo, hi, comps):
        import VoigtFit
        _install_fm_patch()
        cfg = self.cfg
        wr, fr, er = self.w[lo:hi], f[lo:hi], sig[lo:hi]
        half = 0.5 * self.dlnw
        z_lo = float(wr[0] * np.exp(half) / LYA_REST_A - 1.0)
        z_hi = float(wr[-1] * np.exp(-half) / LYA_REST_A - 1.0)
        ds = VoigtFit.DataSet(float(np.sqrt(wr[0] * wr[-1]) / LYA_REST_A - 1.0))
        ds.verbose = False
        ds.add_data(wr, fr, 0.0, err=er, normalized=True)
        ds.add_line("HI_1215", velspan=0.5 * (hi - lo) * self.dv_obs + 3.0 * self.dv_obs)
        for z, b, N in comps:
            ds.add_component("HI", float(np.clip(z, z_lo, z_hi)), b, N)
        ds.prepare_dataset(norm=False, mask=False, verbose=False)
        for reg in ds.regions:
            reg.kernel = self.kernel
        for k in range(len(comps)):
            ds.pars[f"logN{k}_HI"].set(min=cfg.logN_min, max=cfg.logN_max)
            ds.pars[f"b{k}_HI"].set(min=cfg.b_min, max=cfg.b_max)
            ds.pars[f"z{k}_HI"].set(min=z_lo, max=z_hi)
        popt, chi2 = ds.fit(verbose=False)
        p = ds.best_fit
        fitted = [(p[f"z{k}_HI"].value, p[f"b{k}_HI"].value, p[f"logN{k}_HI"].value) for k in range(len(comps))]
        resid = -np.asarray(popt.residual, dtype=np.float64) * er    # model - data
        return fitted, resid, float(np.sum(popt.residual ** 2))

    @staticmethod
    def _aicc(chi2: float, k: int, n: int) -> float:
        return chi2 + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)

    def fit(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.cfg
        f, sig, shift = obs["flux"], obs["sigma"], obs["shift"]
        comps_all, chi2_tot, n_tot, n_err = [], 0.0, 0, 0
        for lo, hi in self._regions(f, sig):
            n = hi - lo
            max_comp = max((n - 2) // 3, 1)                 # leastsq needs n_params < n_pix
            trial = sorted(self._seed(f, sig, lo, hi), key=lambda c: -c[2])[:max_comp]
            best, best_aicc, best_r, best_chi2 = None, np.inf, None, None
            for _ in range(cfg.max_iter):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        fitted, r, chi2 = self._fit(f, sig, lo, hi, trial)
                    except (ValueError, TypeError, OverflowError, FloatingPointError):
                        n_err += 1
                        if len(trial) <= 1 or best is not None:
                            break
                        trial = trial[:-1]
                        continue
                a = self._aicc(chi2, 3 * len(fitted), n)
                if best is not None and a - best_aicc >= cfg.d_aicc:
                    break
                best, best_aicc, best_r, best_chi2 = fitted, a, r, chi2
                rs = self._smooth(np.clip(best_r, 0, None)) / (sig[lo:hi] / np.sqrt(3.0))
                pk, _ = find_peaks(rs, height=cfg.resid_sig)
                if pk.size == 0 or len(best) + 1 > max_comp:
                    break
                p = int(pk[np.argmax(rs[pk])])
                trial = list(best) + [(self._z(lo + p), 20.0, 12.8)]
            chi2_null = float(np.sum(((1.0 - f[lo:hi]) / sig[lo:hi]) ** 2))
            if best is not None and best_aicc - chi2_null >= cfg.d_aicc:   # no better than no absorption
                best = None
            if best is not None:
                comps_all.extend(best)
                chi2_tot += best_chi2
                n_tot += n
        out = np.empty((len(comps_all), 3))
        span = 2048 * self.fm.dv
        for i, (z, b, N) in enumerate(comps_all):
            pix_fine = np.log((1 + z) * LYA_REST_A / self.fm.fine_wave[0]) / np.log(self.fm.wave[1] / self.fm.wave[0])
            out[i] = (((pix_fine + shift) * self.fm.dv) % span, b, N)
        k_tot = 3 * len(comps_all)
        diag = {"chi2_nu": float(chi2_tot / max(n_tot - k_tot, 1)), "n_regions_pix": float(n_tot),
                "n_comp": float(len(comps_all)), "n_fit_errors": float(n_err)}
        return out, diag


# ---------------------------------------------------------------------------
# batch catalog writer (same resumable chunked layout as voigt.fit_catalog)
# ---------------------------------------------------------------------------
_W: Dict[str, object] = {}


def _init(wave, dv, lsf, ocfg: Dict, fcfg: Dict) -> None:
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    fm = ForwardModel(wave, dv, lsf, ObsConfig(**ocfg))
    _W["fm"], _W["fit"] = fm, NoisyVoigtFitter(fm, NoisyFitConfig(**fcfg))


def _work(args: Tuple[int, np.ndarray, int]):
    start, block, seed = args
    comps, diags = [], []
    for j, row in enumerate(block):
        rng = np.random.default_rng([seed, start + j])
        obs = _W["fm"].observe(row, rng)
        try:
            c, d = _W["fit"].fit(obs)
        except Exception:                                      # flagged (n_fit_errors=-2), counted in analysis
            c, d = np.zeros((0, 3)), {k: (-2.0 if k == "n_fit_errors" else 0.0) for k in DIAG_KEYS_NOISY}
        comps.append(c)
        diags.append(d)
    return start, comps, diags


DIAG_KEYS_NOISY = ("chi2_nu", "n_regions_pix", "n_comp", "n_fit_errors")


def fit_noisy_catalog(flux: np.ndarray, wave: np.ndarray, dv: float, lsf: np.ndarray, ocfg: ObsConfig,
                      fcfg: NoisyFitConfig, noise_seed: int, out_path: Path, n_jobs: int = 9,
                      chunk: int = 1024, block: int = 32) -> None:
    """Forward-model + fit every sightline; checkpointed per chunk, merged into out_path."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = flux.shape[0]
    starts = list(range(0, n, chunk))
    initargs = (np.asarray(wave), dv, lsf, asdict(ocfg), asdict(fcfg))
    pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=initargs)
    try:
        for ci, s0 in enumerate(starts):
            part = out_path.with_suffix(f".part{ci:04d}.npz")
            if part.exists():
                continue
            s1 = min(s0 + chunk, n)
            jobs = [(s, np.asarray(flux[s:min(s + block, s1)], dtype=np.float64), noise_seed) for s in range(s0, s1, block)]
            rows = {}
            pending = list(jobs)
            for attempt in range(3):
                try:
                    futs = [pool.submit(_work, j) for j in pending]
                    for fu in as_completed(futs):
                        st, cs, ds = fu.result()
                        for k, (c, d) in enumerate(zip(cs, ds)):
                            rows[st + k] = (c, d)
                    break
                except BrokenProcessPool:
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=initargs)
                    pending = [(s + k, b[k:k + 1], noise_seed) for s, b, _ in pending for k in range(b.shape[0])
                               if (s + k) not in rows]
            for i in range(s0, s1):                            # crashed rows -> empty, flagged
                rows.setdefault(i, (np.zeros((0, 3)), {k: (-1.0 if k == "n_fit_errors" else 0.0) for k in DIAG_KEYS_NOISY}))
            sl = np.concatenate([np.full(len(rows[i][0]), i) for i in range(s0, s1)])
            cp = np.vstack([rows[i][0] for i in range(s0, s1)]) if s1 > s0 else np.zeros((0, 3))
            dg = np.array([[rows[i][1][k] for k in DIAG_KEYS_NOISY] for i in range(s0, s1)])
            np.savez(part, sightline=sl, comps=cp, diag=dg)
            print(f"[noisy] {out_path.name}: {s1}/{n}", flush=True)
    finally:
        pool.shutdown(wait=True)
    parts = [np.load(out_path.with_suffix(f".part{ci:04d}.npz")) for ci in range(len(starts))]
    comps = np.vstack([p["comps"] for p in parts])
    np.savez_compressed(out_path, sightline=np.concatenate([p["sightline"] for p in parts]).astype(np.int64),
                        v_kms=comps[:, 0], b_kms=comps[:, 1], logN=comps[:, 2],
                        diag=np.vstack([p["diag"] for p in parts]), diag_keys=np.array(DIAG_KEYS_NOISY),
                        obs_config=np.array(str(asdict(ocfg))), fit_config=np.array(str(asdict(fcfg))))
    for ci in range(len(starts)):
        out_path.with_suffix(f".part{ci:04d}.npz").unlink()
