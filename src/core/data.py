import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any
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
