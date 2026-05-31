import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any, Literal
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class DataSummary:
    """Summary statistics for loaded data."""
    shape: Tuple[int, ...]
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    has_nan: bool
    has_inf: bool


class DataIngestor:
    """
    Standardizes data ingestion for Sherwood-style datasets.
    """
    def __init__(self, base_path: str, filename: str = "flux.npy", num_classes: int = 4):
        self.base_path = base_path
        self.filename = filename
        self.num_classes = num_classes

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and concatenates data from class-specific directories.
        """
        X_list = []
        y_list = []

        for i in range(1, self.num_classes + 1):
            file_path = os.path.join(self.base_path, str(i), self.filename)
            if not os.path.exists(file_path):
                # Fallback check if the user provided generic path but file exists as 'data.npy'
                if self.filename == "flux.npy" and os.path.exists(os.path.join(self.base_path, str(i), "data.npy")):
                    file_path = os.path.join(self.base_path, str(i), "data.npy")
                else:
                    raise FileNotFoundError(f"Data file not found at {file_path}")

            data = np.load(file_path)
            X_list.append(data)
            y_list.append(np.full(len(data), i))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    def load_per_class(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Loads data as list of per-class arrays (useful for micro-probing).
        Returns: (list of 4 arrays, class labels array)
        """
        X_list = []
        y_list = []

        for i in range(1, self.num_classes + 1):
            file_path = os.path.join(self.base_path, str(i), self.filename)
            if not os.path.exists(file_path):
                if self.filename == "flux.npy" and os.path.exists(os.path.join(self.base_path, str(i), "data.npy")):
                    file_path = os.path.join(self.base_path, str(i), "data.npy")
                else:
                    raise FileNotFoundError(f"Data file not found at {file_path}")

            data = np.load(file_path, mmap_mode='r')
            X_list.append(data)
            y_list.append(np.full(len(data), i))

        y = np.concatenate(y_list)
        return X_list, y


class SignalClusteringData:
    """
    Data loader for signal clustering analysis.
    Loads wavelet features and computes absorption from flux.
    """
    # Default paths
    FLUX_BASE = "data/preprocessed/Sherwood_z0.3_inf"
    WAVELET_BASE = "data/processed/wavelet_db8_l6_d12"

    def __init__(self, flux_path: str = None, wavelet_path: str = None):
        self.flux_path = flux_path or self.FLUX_BASE
        self.wavelet_path = wavelet_path or self.WAVELET_BASE

    def load_flux(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw flux data from per-class directories.
        Returns: (flux array, class labels)
        """
        ingestor = DataIngestor(self.flux_path, filename="flux.npy")
        flux, y = ingestor.load()
        return flux, y

    def load_absorption(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load flux and compute absorption A = 1 - F, clipped to [0, 1].
        Returns: (absorption array, class labels)
        """
        flux, y = self.load_flux()
        absorption = np.clip(1.0 - flux, 0.0, 1.0)
        return absorption, y

    def load_wavelet(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load wavelet features from per-class directories.
        Returns: (wavelet array, class labels)
        """
        ingestor = DataIngestor(self.wavelet_path, filename="data.npy")
        wavelet, y = ingestor.load()
        return wavelet, y

    def load_wavelet_per_class(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load wavelet features as list of per-class arrays.
        Returns: (list of 4 wavelet arrays, class labels)
        """
        ingestor = DataIngestor(self.wavelet_path, filename="data.npy")
        X_list, y = ingestor.load_per_class()
        return X_list, y

    @staticmethod
    def normalize_wavelet_per_level(X: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization per wavelet level (D1-D6 + A6).

        Wavelet level boundaries for db8 with 6 levels:
        - D1: 0-1024
        - D2: 1024-1536
        - D3: 1536-1792
        - D4: 1792-1920
        - D5: 1920-1984
        - D6: 1984-2016
        - A6: 2016-2048

        Args:
            X: Wavelet array of shape (n_sightlines, 2048)

        Returns:
            Z-score normalized wavelet array
        """
        level_boundaries = {
            'D1': (0, 1024),
            'D2': (1024, 1536),
            'D3': (1536, 1792),
            'D4': (1792, 1920),
            'D5': (1920, 1984),
            'D6': (1984, 2016),
            'A6': (2016, 2048)
        }

        X_normalized = np.zeros_like(X)

        for level_name, (start, end) in level_boundaries.items():
            level_data = X[:, start:end]
            mean = np.mean(level_data)
            std = np.std(level_data)
            if std > 0:
                X_normalized[:, start:end] = (level_data - mean) / std
            else:
                X_normalized[:, start:end] = level_data - mean

        return X_normalized

    def load_wavelet_per_class_normalized(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load wavelet features as list of per-class arrays with z-score normalization per level.
        Returns: (list of 4 normalized wavelet arrays, class labels)
        """
        X_list, y = self.load_wavelet_per_class()
        X_normalized = [self.normalize_wavelet_per_level(X) for X in X_list]
        return X_normalized, y

    def load_wavelet_global_normalized(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load wavelet features as list of per-class arrays with global z-score normalization per level.
        Normalization is computed across ALL data (all classes combined) before splitting per class.
        Returns: (list of 4 normalized wavelet arrays, class labels)
        """
        X_list, y = self.load_wavelet_per_class()
        # Stack all classes to compute global statistics
        X_all = np.vstack(X_list)
        # Normalize using global statistics per level
        X_all_normalized = self.normalize_wavelet_per_level(X_all)
        # Split back into per-class
        X_normalized = []
        start = 0
        for X in X_list:
            end = start + X.shape[0]
            X_normalized.append(X_all_normalized[start:end])
            start = end
        return X_normalized, y

    def load_flux_per_class(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load flux as list of per-class arrays.
        Returns: (list of 4 flux arrays, class labels)
        """
        ingestor = DataIngestor(self.flux_path, filename="flux.npy")
        X_list, y = ingestor.load_per_class()
        return X_list, y

    def get_summary(self, X: np.ndarray, name: str = "data") -> DataSummary:
        """
        Compute summary statistics for an array.
        """
        return DataSummary(
            shape=X.shape,
            dtype=str(X.dtype),
            min=float(np.nanmin(X)),
            max=float(np.nanmax(X)),
            mean=float(np.nanmean(X)),
            std=float(np.nanstd(X)),
            has_nan=bool(np.any(np.isnan(X))),
            has_inf=bool(np.any(np.isinf(X)))
        )

    def validate(self, X_wavelet: np.ndarray, X_raw: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run sanity checks on loaded data.
        Returns dict with validation results.
        """
        results = {}

        # Check wavelet
        results['wavelet'] = {
            'shape': X_wavelet.shape,
            'has_nan': bool(np.any(np.isnan(X_wavelet))),
            'has_inf': bool(np.any(np.isinf(X_wavelet))),
            'min': float(np.nanmin(X_wavelet)),
            'max': float(np.nanmax(X_wavelet)),
            'valid': not (np.any(np.isnan(X_wavelet)) or np.any(np.isinf(X_wavelet)))
        }

        # Check absorption
        results['absorption'] = {
            'shape': X_raw.shape,
            'in_range': bool(np.all((X_raw >= 0) & (X_raw <= 1))),
            'has_negative': bool(np.any(X_raw < 0)),
            'mean': float(np.nanmean(X_raw)),
            'valid': bool(np.all((X_raw >= 0) & (X_raw <= 1)))
        }

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        results['class_distribution'] = {int(k): int(v) for k, v in zip(unique, counts)}
        results['class_valid'] = all(v == 16384 for v in counts)

        results['all_valid'] = all([
            results['wavelet']['valid'],
            results['absorption']['valid'],
            results['class_valid']
        ])

        return results


def group_indices_within_class(
    n_sightlines: int,
    M: int,
    *,
    seed: int = 42,
) -> List[np.ndarray]:
    """Random disjoint within-class grouping of sightline indices into groups of size M.

    Returns a list of integer-index arrays (each shape ``(M,)``) covering a random
    permutation of ``range(n_sightlines)``. Any tail of size < M (sightlines that do
    not fit into a complete group) is dropped — this is documented behaviour. With
    the canonical Sherwood shape ``N=16384`` and ``M ∈ {1, 4, 16, 64, 256}``, every
    one of those M divides N exactly, so the tail is empty in practice; the drop
    is defensive against off-spec callers.

    The caller is responsible for looping over classes — this helper operates on a
    single class at a time. It is the SNR-correct counterpart to
    :func:`stack_flux_within_class` for P(k) stacking: compute P(k) per-sightline,
    then average per group using these index arrays (so the noise floor falls
    ~1/M for independent sightlines).

    Args:
        n_sightlines: Number of sightlines available in the class (axis-0 size).
        M: Group size (stack depth). Must be >= 1.
        seed: RNG seed for reproducible disjoint grouping.

    Returns:
        List of length ``n_sightlines // M``, each entry a ``(M,)`` int64 index array.
        The union of all entries is a subset of ``range(n_sightlines)`` with no
        duplicates; entries are pairwise disjoint.

    Raises:
        ValueError: if ``M < 1`` or ``n_sightlines < 1``.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    if n_sightlines < 1:
        raise ValueError(f"n_sightlines must be >= 1, got {n_sightlines}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sightlines)
    n_groups = n_sightlines // M
    # Drop the tail (perm[n_groups * M :]) so every group is exactly size M.
    perm = perm[: n_groups * M]
    # shape: (n_groups, M)
    groups = perm.reshape(n_groups, M)
    return [groups[g] for g in range(n_groups)]


def stack_flux_within_class(
    flux: np.ndarray,
    M: int,
    *,
    seed: int = 42,
    mode: Literal["flux", "pk_passthrough"] = "flux",
) -> np.ndarray:
    """Average flux within random disjoint groups of size M (one class at a time).

    Caller loops over classes; this operates on a single ``(N, 2048)`` per-class
    flux block. Returns the per-group MEAN flux — i.e. the "stack the spectra,
    then compute P(k) downstream" path.

    Tail handling: any sightlines that do not fit into a complete group of M are
    DROPPED. With Sherwood ``N=16384`` and ``M ∈ {1, 4, 16, 64, 256}`` the tail
    is always empty; the drop is defensive against off-spec callers.

    For the SNR-correct stacking pattern (compute P(k) per-sightline, then average
    per group), use :func:`group_indices_within_class` instead — that returns the
    index arrays so the caller can run ``FluxPowerSpectrum`` per-sightline and
    average within groups. This helper deliberately implements ONLY the
    flux-averaging path; the ``pk_passthrough`` mode is intentionally NOT
    supported here (it would require the helper to know about P(k), violating
    separation of concerns) — pass ``mode="pk_passthrough"`` and you get a
    ``NotImplementedError`` pointing you at the sibling helper.

    Args:
        flux: Per-class flux array, shape ``(N_sightlines, 2048)``.
        M: Stack depth (group size). Must be >= 1.
        seed: RNG seed for reproducible disjoint grouping.
        mode: ``"flux"`` (default) averages flux per group and returns the
            ``(N//M, 2048)`` block. ``"pk_passthrough"`` is reserved and
            raises ``NotImplementedError`` — use :func:`group_indices_within_class`.

    Returns:
        Stacked flux array, shape ``(N_sightlines // M, 2048)``, dtype ``float64``.

    Raises:
        ValueError: if ``flux`` is not 2D or ``M < 1``.
        NotImplementedError: if ``mode == "pk_passthrough"``.
    """
    if mode == "pk_passthrough":
        raise NotImplementedError(
            "mode='pk_passthrough' is not implemented in stack_flux_within_class "
            "(would require P(k) knowledge here). Use group_indices_within_class() "
            "to get the grouping, compute P(k) per-sightline, then average per group."
        )
    if mode != "flux":
        raise ValueError(f"mode must be 'flux' or 'pk_passthrough', got {mode!r}")
    if flux.ndim != 2:
        raise ValueError(f"flux must be 2D (N_sightlines, 2048), got shape {flux.shape}")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    n_sightlines = flux.shape[0]
    index_groups = group_indices_within_class(n_sightlines, M, seed=seed)
    n_groups = len(index_groups)

    # shape: (n_groups, 2048)
    out = np.empty((n_groups, flux.shape[1]), dtype=np.float64)
    for g, idx in enumerate(index_groups):
        out[g] = flux[idx].mean(axis=0)
    return out


def load_velocity_axis(
    class_id: int,
    flux_base: str = SignalClusteringData.FLUX_BASE,
) -> np.ndarray:
    """Load the per-pixel velocity axis ``vel.npy`` for one Sherwood class.

    Args:
        class_id: Physics class id ∈ {1, 2, 3, 4} (1=NoFeedback, 2=StellarWind,
            3=WindAGN, 4=WindStrongAGN).
        flux_base: Base directory under which ``{class_id}/vel.npy`` lives.
            Defaults to :attr:`SignalClusteringData.FLUX_BASE`.

    Returns:
        ``(2048,) float64`` velocity-axis array in km/s.

    Raises:
        FileNotFoundError: if ``vel.npy`` is absent for the requested class.
        ValueError: if the loaded axis is not 1D length 2048, or contains NaN/Inf.
    """
    if class_id not in (1, 2, 3, 4):
        raise ValueError(f"class_id must be in {{1,2,3,4}}, got {class_id}")
    path = os.path.join(flux_base, str(class_id), "vel.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"vel.npy not found at {path}")
    vel = np.load(path).astype(np.float64, copy=False)
    if vel.ndim != 1 or vel.shape[0] != 2048:
        raise ValueError(
            f"vel.npy for class {class_id} has shape {vel.shape}, expected (2048,)"
        )
    if not np.all(np.isfinite(vel)):
        raise ValueError(f"vel.npy for class {class_id} contains non-finite values")
    return vel


def assert_velocity_axes_uniform(
    tol: float = 1e-9,
    flux_base: str = SignalClusteringData.FLUX_BASE,
) -> float:
    """Assert all 4 classes share an identical, uniformly-spaced velocity axis.

    Loads ``vel.npy`` for classes 1–4, asserts they are pairwise identical within
    ``tol`` (max-abs-difference) AND that the per-pixel spacing is uniform within
    ``tol`` (max - min of ``diff(vel)`` <= tol). On success, returns the scalar
    Δv (km/s/pixel) derived from class 1.

    The probe script in ``experiments/pk-feedback-classifier/`` calls this once
    on startup so the physical k-axis (``k = 2π · rfftfreq(2048, d=Δv)``) is
    derived from disk rather than hardcoded ([D-04]).

    Args:
        tol: Absolute tolerance for both the inter-class identity check and the
            intra-axis uniform-spacing check. Default ``1e-9`` km/s.
        flux_base: Base directory; passed through to :func:`load_velocity_axis`.

    Returns:
        Δv (km/s/pixel) — the uniform per-pixel velocity step. Expected ≈ 2.6365.

    Raises:
        AssertionError: if any two class velocity axes differ by > tol, or if
            the spacing is non-uniform beyond tol.
    """
    axes = [load_velocity_axis(c, flux_base=flux_base) for c in (1, 2, 3, 4)]
    ref = axes[0]
    for c, ax in zip((2, 3, 4), axes[1:]):
        diff = float(np.max(np.abs(ax - ref)))
        assert diff <= tol, (
            f"vel.npy for class {c} differs from class 1 by max-abs={diff:.3e} > tol={tol:.3e}"
        )
    spacing = np.diff(ref)
    spread = float(spacing.max() - spacing.min())
    assert spread <= tol, (
        f"vel.npy spacing is non-uniform: max-min(diff)={spread:.3e} > tol={tol:.3e}"
    )
    delta_v = float(spacing.mean())
    return delta_v


class DatasetFactory:
    """
    Factory for retrieving pre-split or raw datasets.
    """
    @staticmethod
    def get_dataset(name: str, base_data_dir: str = "data", filename: str = "flux.npy") -> Tuple[np.ndarray, np.ndarray]:
        # Implementation for specific dataset names if needed
        # For now, default to checking the data directory structure
        dataset_path = os.path.join(base_data_dir, "processed", name)
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(base_data_dir, name) # check top level data dir

        ingestor = DataIngestor(dataset_path, filename=filename)
        return ingestor.load()
