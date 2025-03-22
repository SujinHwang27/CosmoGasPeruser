"""
Main script for running the SVM classification pipeline.
"""

import logging
import os
import argparse
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, prepare_dataset
from src.feature_extraction import perform_pca_analysis, apply_pca_transformation
from src.model import train_and_evaluate_svm
# from src.evaluation import (analyze_performance_vs_overfit, create_factors_heatmap,
#                           evaluate_model_performance)
# from src.visualization import plot_results

# # Add the project root to Python path
# project_root = str(Path(__file__).parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # 1. Load and prepare data
    data = load_data()
    spectra, labels = prepare_dataset(data)
    
    # 2. Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, 
        labels, 
        test_size=0.2,  
        random_state=42,  
        stratify=labels  # maintain class distribution
    )
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # 3. Perform PCA analysis on training data only
    logger.info("Performing PCA analysis on training data...")
    n_components_dict = perform_pca_analysis(X_train)
  
    # 4. Apply PCA transformation to train set
    X_train_pca, X_test_pca = apply_pca_transformation(X_train, X_test)
    
    # 5. Train and evaluate SVM model
    logger.info(f"\nTraining and evaluating SVM model")
    results = train_and_evaluate_svm(
        X_train_pca,
        y_train,
        X_test_pca,
        y_test
    )
    
    # 6. Plot results
    # plot_results(results, n_components_dict)

if __name__ == "__main__":
    main()
