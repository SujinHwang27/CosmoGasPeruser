"""
Script to plot PCA variance analysis.
"""

import logging
import numpy as np
from sklearn.decomposition import PCA

from pathlib import Path
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from classifier_SVM.src.data_loader import load_data
from data_explorer.visualization import plot_pca_variance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load data
    redshift = 2.4  # You can change this value
    logger.info(f"Loading data for redshift {redshift}...")
    data = load_data(redshift)
    
    # Combine all spectra
    all_spectra = []
    for physics, spectra in data.items():
        all_spectra.extend(spectra)
    all_spectra = np.array(all_spectra)
    
    # Perform PCA analysis
    logger.info("Performing PCA analysis...")
    pca = PCA().fit(spectra)    
    # Plot PCA variance
    logger.info("Plotting PCA variance...")
    plot_pca_variance(pca)

if __name__ == "__main__":
    main() 