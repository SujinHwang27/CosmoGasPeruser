"""
Feature extraction functionality including PCA and local minima analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
import random
import logging
from pathlib import Path
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config using absolute import
from classifier_SVM.config.config import EXPLAINED_VARIANCE, NCOMP

logger = logging.getLogger(__name__)


def perform_pca_analysis(spectra: np.ndarray) -> Dict[str, int]:
    """
    Perform PCA analysis to determine optimal number of components.
    
    Args:
        spectra: Array of spectra
        
    Returns:
        Dictionary of optimal number of components for each variance threshold
    """
    n_components_dict = {}
    
    logger.info(f"PCA analysis for spectra shape: {spectra.shape}")
    
    pca = PCA().fit(spectra)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    for threshold_key, threshold_value in EXPLAINED_VARIANCE.items():
        n_components = np.where(cumulative_variance >= threshold_value)[0][0] + 1
        n_components_dict[threshold_key] = n_components
        logger.info(f"Components needed for {threshold_value*100}% variance: {n_components}")

            
    return n_components_dict

def apply_pca_transformation(train_spectra: np.ndarray, test_spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PCA transformation to the data.
    
    Args:
        spectra: Original spectra array
        
    Returns:
        PCA-transformed data
    """
    # while True:
    #     try:
    #         n_components = input("\nEnter desired variance threshold between 0 and 1(e.g., 0.95) or the number of principal components: ")
    #         n_components = float(n_components)

    #         if 0 < n_components and n_components < 1:
    #             logger.info(f"\nApplying PCA transformation for {n_components}% variance...")
    #             break
    #         elif 1 <= n_components and n_components <= train_spectra.shape[1]:
    #             n_components = int(n_components)
    #             logger.info(f"Applying PCA with {n_components} principal components")
    #             break
    #         else:
    #             logger.error(f"Invalid variance threshold.")
    #     except ValueError:
    #         logger.error("Please enter a valid number.")
    
    pca = PCA(n_components=NCOMP, svd_solver='full')
    X_train_pca = pca.fit_transform(train_spectra)
    X_test_pca = pca.transform(test_spectra)
    logger.info(f"Transformed shape (train, test): {X_train_pca.shape}, {X_test_pca.shape}")
    
    return X_train_pca, X_test_pca

