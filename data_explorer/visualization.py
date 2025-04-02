"""
Visualization functionality for spectra and analysis results.
"""

from pathlib import Path
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import Dict, List, Tuple
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_spectrum(flux: np.ndarray, 
                 wavelength: np.ndarray = None,
                 title: str = "Spectrum Plot",
                 figsize: Tuple[int, int] = (20, 4)) -> None:
    """
    Plot a single spectrum.
    
    Args:
        flux: Array of flux values
        wavelength: Array of wavelength values (optional)
        title: Plot title
        figsize: Figure size
    """
    if wavelength is None:
        wavelength = np.arange(len(flux))
        
    plt.figure(figsize=figsize)
    plt.plot(wavelength, flux, label='Spectrum', color='black', lw=1.5)
    plt.xlabel('Wavelength index', fontsize=12)
    plt.ylabel('Flux', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

def plot_spectrum_with_minima(flux: np.ndarray,
                            minima_indices: np.ndarray,
                            minima_values: np.ndarray,
                            title: str = "Spectrum with Local Minima",
                            figsize: Tuple[int, int] = (14, 4)) -> None:
    """
    Plot spectrum with highlighted local minima.
    
    Args:
        flux: Array of flux values
        minima_indices: Indices of local minima
        minima_values: Values at local minima
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(flux, label="Flux", color="blue", linewidth=1)
    plt.scatter(minima_indices, minima_values, 
               color="red", label="Local Minima", zorder=5)
    plt.title(title)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Flux Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_grayscale_spectra(spectra: np.ndarray,
                          num_samples: int = 20,
                          title: str = "",
                          figsize: Tuple[int, int] = (20, 4)) -> None:
    """
    Plot multiple spectra as grayscale images.
    
    Args:
        spectra: Array of spectra
        num_samples: Number of spectra to plot
        title: Plot title
        figsize: Figure size
    """
    indices = np.random.choice(len(spectra), 
                             size=min(num_samples, len(spectra)), 
                             replace=False)
    sample_spectra = spectra[indices]
    
    plt.figure(figsize=figsize)
    plt.imshow(sample_spectra, cmap="gray", aspect="auto", interpolation='none')
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_local_minima_distribution(minima_counts: List[int],
                                 title: str = "Distribution of Local Minima Counts",
                                 figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot distribution of local minima counts.
    
    Args:
        minima_counts: List of local minima counts
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.hist(minima_counts, bins='auto', color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Number of Local Minima")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_minima_locations(all_minima_indices: List[int],
                         title: str = "Local Minima Locations",
                         figsize: Tuple[int, int] = (20, 2)) -> None:
    """
    Plot locations of local minima across wavelength indices.
    
    Args:
        all_minima_indices: List of all local minima indices
        title: Plot title
        figsize: Figure size
    """
    from collections import Counter
    index_frequencies = Counter(all_minima_indices)
    indices = np.arange(max(all_minima_indices) + 1)
    frequencies = np.zeros_like(indices, dtype=float)
    
    for index, freq in index_frequencies.items():
        frequencies[index] = freq
    
    plt.figure(figsize=figsize)
    plt.scatter(indices, np.zeros_like(indices), 
               c=frequencies, cmap='viridis', 
               norm=plt.Normalize(vmin=frequencies.min(), vmax=frequencies.max()),
               s=10)
    
    plt.colorbar(label="Frequency of Local Minima")
    plt.title(title)
    plt.xlabel("Wavelength Index")
    plt.yticks([])
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

def plot_pca_variance(pca: PCA,
                     title: str = "PCA Explained Variance") -> None:
    """
    Plot PCA explained variance ratio.
    
    Args:
        pca_results: Dictionary of variance ratios
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')


    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_pca_reconstruction(original: np.ndarray,
                          reconstructed: np.ndarray,
                          title: str = "Original vs PCA Reconstructed",
                          figsize: Tuple[int, int] = (15, 4)) -> None:
    """
    Plot original spectrum against its PCA reconstruction.
    
    Args:
        original: Original spectrum
        reconstructed: Reconstructed spectrum
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(original, label='Original', color='blue', alpha=0.7)
    plt.plot(reconstructed, label='Reconstructed', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Wavelength Index')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_pca_loadings(pca: PCA, 
                        n_components: int = 20,
                        feature_names: List[str] = None) -> pd.DataFrame:
    """
    Analyze and visualize PCA loadings (feature contributions).
    
    Args:
        pca: Fitted PCA object
        n_components: Number of components to analyze
        feature_names: List of feature names (optional)
        
    Returns:
        DataFrame containing PCA loadings
    """
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(pca.components_.shape[1])]
    
    # Create component names
    component_names = [f"PC{i+1}" for i in range(n_components)]
    
    # Create DataFrame of loadings
    loadings_df = pd.DataFrame(
        pca.components_[:n_components],
        columns=feature_names,
        index=component_names
    )
    
    # Print top features for each component
    top_features_per_pc = loadings_df.abs().idxmax(axis=1)
    logger.info("\nMost important feature for each principal component:")
    logger.info(top_features_per_pc)
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(loadings_df, cmap="coolwarm", annot=False, center=0)
    plt.title("PCA Loadings (Feature Contributions to Components)")
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    plt.tight_layout()
    plt.show()
    
    return loadings_df

def plot_pca_projection(X: np.ndarray,
                       y: np.ndarray,
                       n_components: int = 2) -> None:
    """
    Plot PCA projection of the dataset in 2D space.
    
    Args:
        X: Feature matrix
        y: Labels
        n_components: Number of components to use for projection (default: 2)
        title: Plot title
        figsize: Figure size
    """

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"PCA Projection of the Dataset with {n_components} principal components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

def plot_tsne(X: np.ndarray, y:np.ndarray):

    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization of Dataset")
    plt.colorbar(label="Class Label")
    plt.show()


