import numpy as np
from typing import Optional, List, Union, Tuple
from scipy.fftpack import dct
import pywt
from src.core.base import BaseTransformer

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

class FluxPowerSpectrum(BaseTransformer):
    """
    1D Lyman-alpha flux power spectrum P_F(k) transform.

    Scientific purpose
    ------------------
    Computes the per-sightline 1D flux power spectrum P_F(k) in velocity space
    (s/km units, the conventional Lya basis), then log-bins to a low-dimensional
    feature for downstream supervised classification of physics-feedback labels.
    This is the canonical sufficient statistic for the thermal/pressure-smoothing
    structure that feedback modifies (see experiments/pk-feedback-classifier
    [D-01], [D-04]).

    Pipeline
    --------
    1.  Clip ~1e-16 float underflow: F = clip(X, 0, None).
    2.  Mean-flux normalization:
          - 'per_sightline' (PRIMARY): delta_F = F / <F>_los - 1.
            Forces classifier onto P(k) *shape* (mean-flux is divided out).
          - 'global' (SECONDARY confounder probe per [D-02]/[D-03]):
            delta_F = F / <F>_global - 1; the k=0 bin is DROPPED from the
            output (output has n_kbins - 1 columns) because the global
            normalization preserves the inter-sightline mean-flux variance
            as a DC channel which is exactly the leakage we want to expose
            via the importance map, not via a flat zero column.
    3.  Apply Hann window then numpy rfft along axis=1; |.|^2 gives the raw
        positive-frequency power.
    4.  Power-spectrum normalization: P(k) = |rfft|^2 * (delta_v / N) where
        N is the number of pixels (2048). This is the discrete equivalent of
        the continuous P(k) integral and has units of (delta_F)^2 * (km/s).
        With pre-Hann windowing this is approximate (no 1/sum(w^2) correction
        is applied — the windowing constant cancels uniformly across all
        sightlines and so is irrelevant for classification).
    5.  k-axis: k = 2*pi*rfftfreq(N, d=delta_v). At delta_v=2.6365 km/s,
        Nyquist ~ 1.19 s/km and k_min ~ 1.16e-3 s/km.
    6.  Log-bin the 1024 raw positive-frequency P(k) values into n_kbins
        log-spaced k-bins (mean within each bin). Bin edges/centers are
        stored as attributes for later mapping to physical k.

    Parameters
    ----------
    norm : {'per_sightline', 'global'}, default 'per_sightline'
    n_kbins : int, default 20
        Number of log-spaced k-bins. The first bin always contains k=0 in
        'per_sightline' mode; for 'global' mode the k=0 bin is dropped and
        the returned width is n_kbins - 1.
    vel_path : Optional[str]
        Path to a 'vel.npy' file from which delta_v is derived (uniform-spacing
        is asserted). Either vel_path or delta_v must be set before
        fit_transform; delta_v takes precedence if both are given.
    delta_v : Optional[float]
        Direct velocity spacing in km/s (override; useful for tests).

    Attributes (set by fit_transform)
    ---------------------------------
    delta_v_ : float
    k_raw_ : np.ndarray   # shape (N//2 + 1,)
    k_bin_edges_ : np.ndarray   # shape (n_kbins + 1,)
    k_bin_centers_ : np.ndarray   # shape (n_kbins,) or (n_kbins - 1,) after drop
    dropped_k0_ : bool
    """

    def __init__(
        self,
        norm: str = "per_sightline",
        n_kbins: int = 20,
        vel_path: Optional[str] = None,
        delta_v: Optional[float] = None,
    ):
        if norm not in ("per_sightline", "global"):
            raise ValueError(f"norm must be 'per_sightline' or 'global', got {norm!r}")
        self.norm = norm
        self.n_kbins = int(n_kbins)
        self.vel_path = vel_path
        self.delta_v = delta_v

        # Attributes populated at fit_transform time.
        self.delta_v_: Optional[float] = None
        self.k_raw_: Optional[np.ndarray] = None
        self.k_bin_edges_: Optional[np.ndarray] = None
        self.k_bin_centers_: Optional[np.ndarray] = None
        self.dropped_k0_: bool = False

    def _resolve_delta_v(self) -> float:
        """Resolve delta_v from explicit override or vel.npy file."""
        if self.delta_v is not None:
            return float(self.delta_v)
        if self.vel_path is None:
            raise ValueError(
                "FluxPowerSpectrum requires either delta_v or vel_path to be set."
            )
        vel = np.load(self.vel_path)
        diffs = np.diff(vel)
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"vel.npy at {self.vel_path} is not uniformly spaced "
                f"(min diff {diffs.min()}, max diff {diffs.max()})."
            )
        return float(diffs[0])

    def _build_kbins(self, k_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Log-spaced k-bin edges/centers spanning the positive raw k range."""
        # Smallest positive k is k_raw[1] (k_raw[0] = 0 is the DC bin).
        k_min_pos = k_raw[1]
        k_max = k_raw[-1]
        edges = np.logspace(
            np.log10(k_min_pos), np.log10(k_max), self.n_kbins + 1
        )
        # Geometric centers.
        centers = np.sqrt(edges[:-1] * edges[1:])
        return edges, centers

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, n_pixels), got shape {X.shape}.")
        n_samples, n_pixels = X.shape  # shape: (n_samples, n_pixels)

        # Cast to float64 just in case caller passed float32; spec is float64.
        F = np.asarray(X, dtype=np.float64)
        # Clip ~1e-16 float underflow negatives to 0 BEFORE any normalization.
        F = np.clip(F, 0.0, None)

        # Mean-flux normalization.
        if self.norm == "per_sightline":
            mean_F = F.mean(axis=1, keepdims=True)  # shape: (n_samples, 1)
            # Guard zero-mean rows (would be a fully-saturated sightline).
            mean_F = np.where(mean_F > 0.0, mean_F, 1.0)
            delta_F = F / mean_F - 1.0
        else:  # 'global'
            mean_F_global = F.mean()
            if mean_F_global <= 0.0:
                mean_F_global = 1.0
            delta_F = F / mean_F_global - 1.0

        # Hann window + rfft + |.|^2.
        window = np.hanning(n_pixels)  # shape: (n_pixels,)
        windowed = delta_F * window  # shape: (n_samples, n_pixels)
        spectrum = np.fft.rfft(windowed, axis=1)  # shape: (n_samples, n_pixels//2+1)
        power = np.abs(spectrum) ** 2  # shape: (n_samples, n_pixels//2+1)

        # Resolve delta_v then build physical k-axis and apply P(k) normalization.
        delta_v = self._resolve_delta_v()
        self.delta_v_ = delta_v
        power *= delta_v / n_pixels  # standard P(k) discrete normalization.

        k_raw = 2.0 * np.pi * np.fft.rfftfreq(n_pixels, d=delta_v)  # shape: (n_pixels//2+1,)
        self.k_raw_ = k_raw

        # Build log-spaced k-bin edges/centers across the positive-k range.
        edges, centers = self._build_kbins(k_raw)
        self.k_bin_edges_ = edges
        self.k_bin_centers_ = centers

        # Assign each raw k to a bin (1..n_kbins for in-range; 0 for k=0 / underflow;
        # n_kbins+1 only for the rare overflow at exactly the top edge -- clip it back).
        bin_idx = np.digitize(k_raw, edges, right=False)  # shape: (n_pixels//2+1,)
        bin_idx = np.clip(bin_idx, 0, self.n_kbins)  # collapse the top-edge overflow into the last bin.

        # Average raw P(k) within each of the n_kbins (1..n_kbins). Use boolean masks
        # rather than a Python loop over samples; vectorize over samples.
        binned = np.zeros((n_samples, self.n_kbins), dtype=np.float64)  # shape: (n_samples, n_kbins)
        for b in range(1, self.n_kbins + 1):
            mask = bin_idx == b
            if mask.any():
                binned[:, b - 1] = power[:, mask].mean(axis=1)
            # If a log-bin happens to be empty (n_kbins too large for n_pixels), leave it at 0.

        if self.norm == "global":
            # Drop the k=0 bin explicitly so we cannot rely on a label-correlated
            # DC channel for the SECONDARY regime. k=0 raw value sits in
            # bin_idx == 0 (below the first log-edge), so we already excluded it
            # from `binned`; however, the smallest positive k (k_raw[1]) IS the
            # bin-1 lower edge. To "drop k=0" we keep the same n_kbins-1 columns
            # but explicitly state the first log-bin's lowest contributor.
            # We drop bin index 0 (== bin 1 in 1-based) only if it is dominated
            # by k_raw[1] -- but per [D-02] the intent is to remove the DC
            # channel itself, which is bin_idx == 0 in our digitize convention
            # and is already excluded. To be defensive and match the spec text,
            # we ALSO drop the first log-bin (which contains k_raw[1], the
            # lowest non-zero k) to ensure no mean-flux residue leaks via the
            # adjacent lowest-k log-bin.
            binned = binned[:, 1:]  # shape: (n_samples, n_kbins - 1)
            self.k_bin_centers_ = centers[1:]
            self.dropped_k0_ = True
        else:
            self.dropped_k0_ = False

        return binned


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
