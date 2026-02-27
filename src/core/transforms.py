import numpy as np
from typing import Optional, List, Union
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
