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
import joblib
import os

from data_explorer.visualization import plot_pca_variance

# # Add the project root to Python path
# project_root = str(Path(__file__).parent.parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # Import config using absolute import
# from classifier_SVM.config.config import EXPLAINED_VARIANCE, NCOMP

EXPLAINED_VARIANCE = {
    '95': 0.95,
    '90': 0.90,
    '85': 0.85,
    '80': 0.80,
    '75': 0.75,
    '70': 0.70,
    '65': 0.65,
    '60': 0.60
}


logger = logging.getLogger(__name__)


def perform_pca_analysis(spectra: np.ndarray, show_plot:True) -> Dict[str, int]:
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
        n_components_dict[f"{threshold_value*100}%"] = int(n_components)
        logger.info(f"Components needed for {threshold_value*100}% variance: {n_components}")

    if show_plot:
        plot_pca_variance(pca)

            
    return pca, n_components_dict

def apply_pca_transformation(train_spectra: np.ndarray, test_spectra: np.ndarray, ncomp:int, pca_name:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PCA transformation to the data.

    """
    pca = PCA(n_components=ncomp, svd_solver='full')
    X_train_pca = pca.fit_transform(train_spectra)
    X_test_pca = pca.transform(test_spectra)
    logger.info(f"Transformed shape (train, test): {X_train_pca.shape}, {X_test_pca.shape}")

    save_dir='saved_models/pca'
    save_path = os.path.join(save_dir, pca_name)
    joblib.dump(pca, save_path)
    logger.info(f"PCA model saved to {save_path}")

    return X_train_pca, X_test_pca

