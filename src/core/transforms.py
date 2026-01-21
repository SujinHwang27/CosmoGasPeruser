import numpy as np
from scipy.fftpack import dct
from typing import Optional, Protocol

class Transform(Protocol):
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        ...

class PCATransform:
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

class DCTTransform:
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

class FisherTransform:
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
