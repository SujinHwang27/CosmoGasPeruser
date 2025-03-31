"""
Main script for running the SVM classification pipeline.
"""

import logging
import os
import argparse
import numpy as np
import psutil
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, prepare_dataset
from src.feature_extraction import perform_pca_analysis, apply_pca_transformation
from src.model import train_and_evaluate_svm


# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from classifier_SVM.config.config import PARAM_GRID_DICT, NCOMP


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
    
    # 3. Perform PCA analysis on training data only if NCOMP is not 194
    if NCOMP != 194:
        logger.info("Performing PCA analysis on training data...")
        n_components_dict = perform_pca_analysis(X_train)
        
        # 4. Apply PCA transformation to train set
        X_train, X_test = apply_pca_transformation(X_train, X_test)
    else:
        logger.info("Skipping PCA analysis as NCOMP equals 194 (original feature dimension)")
    
    # Measure memory before training
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB


    # 5. Train and evaluate SVM model
    logger.info(f"\nTraining and evaluating SVM model")
    results = train_and_evaluate_svm(
        X_train,
        y_train,
        X_test,
        y_test,
        4
    )
    
    # Measure memory after training
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    print(f"Memory Usage Before Training: {mem_before:.2f} MB")
    print(f"Memory Usage After Training: {mem_after:.2f} MB")
    print(f"Memory Increase: {mem_after - mem_before:.2f} MB")

    # 6. Plot results
    # plot_results(results, n_components_dict)

if __name__ == "__main__":
    main()
