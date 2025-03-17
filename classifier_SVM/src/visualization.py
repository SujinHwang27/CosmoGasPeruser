"""
Visualization functionality for spectra and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

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

def plot_pca_variance(pca_results: Dict[str, float],
                     title: str = "PCA Explained Variance",
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot PCA explained variance ratio.
    
    Args:
        pca_results: Dictionary of variance ratios
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(list(pca_results.values()), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(title)
    plt.grid(True)
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
