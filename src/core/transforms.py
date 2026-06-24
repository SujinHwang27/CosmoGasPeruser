import numpy as np
from typing import Optional, List, Union, Tuple
from scipy.fftpack import dct
import pywt
from src.core.base import BaseTransformer


class FluxPowerSpectrumTransform(BaseTransformer):
    """Per-sightline flux power spectrum P_F(k), the canonical Lyα-forest sufficient
    statistic feedback modifies ([D-13]).

    Scientific purpose
    ------------------
    Computes, per sightline, P_F(k) = |rFFT(delta_F)|^2 on the flux contrast
    delta_F = F / <F>_global - 1, then log-bins it geometrically in k. <F>_global is
    the GLOBAL mean flux over the whole input array (a single scalar) — NOT a per-class
    or per-sightline mean: a per-class mean-flux would leak the feedback-recipe label
    (the [D-13]/confounder), so any cross-recipe use must pass one recipe's flux at a
    time and normalize by that recipe's own global mean.

    A power spectrum is phase/sign-insensitive by construction, so P_F(k) cannot encode
    the SIGN of a per-pixel flux change — which is exactly why the signed-response axis
    is, in principle, orthogonal to this basis (the Gate-1 test measures whether that
    holds empirically).

    Output is log10(mean power per geometric k-bin); only populated bins are emitted
    (k=0 / DC is dropped). Set `dv_kms` to attach a physical k-axis (rad / (km/s)).
    """

    def __init__(self, n_bins: int = 64, dv_kms: Optional[float] = None,
                 log_power: bool = True) -> None:
        self.n_bins = n_bins
        self.dv_kms = dv_kms
        self.log_power = log_power
        self.k_bin_centers_: Optional[np.ndarray] = None   # populated by fit_transform

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)                # shape: (n_sightlines, n_pix)
        if not np.isfinite(X).all():
            raise ValueError("FluxPowerSpectrumTransform: X contains NaN/Inf")
        n, npix = X.shape
        mean_F = float(X.mean())                           # <F>_global, a scalar
        if mean_F == 0.0:
            raise ValueError("FluxPowerSpectrumTransform: global mean flux is zero")
        delta = X / mean_F - 1.0                            # flux contrast
        power = np.abs(np.fft.rfft(delta, axis=1)) ** 2     # shape: (n, npix//2 + 1)
        power = power[:, 1:]                                # drop k=0 (DC)
        kidx = np.arange(1, power.shape[1] + 1)             # integer wavenumbers 1..npix/2

        edges = np.geomspace(1.0, float(power.shape[1]), self.n_bins + 1)
        binid = np.clip(np.digitize(kidx, edges) - 1, 0, self.n_bins - 1)
        cols, centers = [], []
        for b in range(self.n_bins):
            m = binid == b
            if not m.any():
                continue
            cols.append(power[:, m].mean(axis=1))           # mean power in the bin
            centers.append(float(kidx[m].mean()))
        out = np.column_stack(cols)                         # shape: (n, n_populated_bins)
        centers_arr = np.asarray(centers, dtype=np.float64)
        if self.dv_kms is not None:                         # physical k = 2*pi*idx/(npix*dv)
            centers_arr = 2.0 * np.pi * centers_arr / (npix * self.dv_kms)
        self.k_bin_centers_ = centers_arr
        return np.log10(out + 1e-30) if self.log_power else out

class PCATransform(BaseTransformer):
    def __init__(self, n_components: int = 2, centered: bool = False):
        self.n_components = n_components
        self.centered = centered

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.centered:
            X = X - np.mean(X, axis=0)
        
        gram_matrix = np.dot(X.T, X)
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
        
        # Sort in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_indices = sorted_indices[:self.n_components]
        principal_components = eigenvectors[:, top_indices]
        
        return np.dot(X, principal_components)

class DCTTransform(BaseTransformer):
    def __init__(self, n_coefficients: Optional[int] = None, mode: str = "full"):
        self.n_coefficients = n_coefficients
        self.mode = mode # "full", "dominant", "high_freq"

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        dct_data = dct(X, type=2, norm='ortho', axis=1)
        
        if self.mode == "full":
            return dct_data
        elif self.mode == "dominant":
            return dct_data[:, :self.n_coefficients]
        elif self.mode == "high_freq":
            return dct_data[:, -self.n_coefficients:]
        else:
            raise ValueError(f"Unknown DCT mode: {self.mode}")

class WindowedDCTTransform(BaseTransformer):
    """
    Implements Joint Position-Frequency analysis using overlapping windows and DCT-II.
    As specified in docs/feature_extraction_plan.md.
    """
    def __init__(self, window_len: int = 256, hop_size: int = 128, n_coeffs: int = 32):
        self.window_len = window_len
        self.hop_size = hop_size
        self.n_coeffs = n_coeffs
        self.window_func = np.hanning(window_len)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        n_samples, n_features = X.shape
        n_windows = ((n_features - self.window_len) // self.hop_size) + 1
        
        # Output shape: (n_samples, n_windows * n_coeffs)
        output = np.zeros((n_samples, n_windows * self.n_coeffs))
        
        for i in range(n_windows):
            start = i * self.hop_size
            end = start + self.window_len
            
            # Apply window function and DCT
            windowed_segment = X[:, start:end] * self.window_func
            dct_segment = dct(windowed_segment, type=2, norm='ortho', axis=1)
            
            # Truncate to first n_coeffs
            output[:, i*self.n_coeffs : (i+1)*self.n_coeffs] = dct_segment[:, :self.n_coeffs]
            
        return output

class WaveletTransform(BaseTransformer):
    """
    Implements multi-resolution analysis using Discrete Wavelet Transform.
    As specified in docs/feature_extraction_plan.md.
    """
    def __init__(self, wavelet: str = 'db8', level: int = 6, drop_levels: List[int] = []):
        self.wavelet = wavelet
        self.level = level
        self.drop_levels = drop_levels # e.g., [1, 2] to drop D1, D2

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        transformed_list = []
        
        for i in range(len(X)):
            coeffs = pywt.wavedec(X[i], self.wavelet, mode='periodization', level=self.level)
            # coeffs is [cA_n, cD_n, cD_{n-1}, ..., cD_1]
            # Our plan says: [D1, D2, D3, D4, D5, D6, A6]
            
            cA6 = coeffs[0]
            cD_list = coeffs[1:][::-1] # [cD1, cD2, cD3, cD4, cD5, cD6]
            
            # Filter dropped levels
            # Level 1 is cD1, etc.
            final_coeffs = []
            for l in range(1, self.level + 1):
                if l not in self.drop_levels:
                    final_coeffs.append(cD_list[l-1])
            
            # A6 usually kept unless explicitely dropped (not standard but possible)
            if 'approximation' not in self.drop_levels and 'A6' not in self.drop_levels:
                final_coeffs.append(cA6)
            
            transformed_list.append(np.concatenate(final_coeffs))
            
        return np.vstack(transformed_list)

class FisherTransform(BaseTransformer):
    """
    Fisher Feature Selection logic.
    Note: Current implementation calculates scores rather than reducing dimensions.
    """
    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Simplified Fisher Score across classes
        classes = np.unique(y)
        means = np.array([np.mean(X[y == c], axis=0) for c in classes])
        
        # S_b: Between-class scatter
        S_b = np.var(means, axis=0)
        # S_w: Within-class scatter (simplified)
        S_w = np.array([np.var(X[y == c], axis=0) for c in classes]).mean(axis=0) + 1e-6
        
        scores = S_b / S_w
        
        if self.top_k:
            top_indices = np.argsort(scores)[-self.top_k:]
            return X[:, top_indices]
        
        return scores
