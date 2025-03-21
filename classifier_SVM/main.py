"""
Main script for running the SVM classification pipeline.
"""

import logging
import os
import argparse
from src.data_loader import load_data, prepare_dataset
from src.feature_extraction import (perform_pca_analysis, apply_pca_transformation,
                                  extract_local_minima_features)
# from src.model import (prepare_training_data, train_svm_with_cv,
#                       save_grid_search_results, train_final_model)
# from src.evaluation import (analyze_performance_vs_overfit, create_factors_heatmap,
#                           evaluate_model_performance)
# from src.visualization import (plot_spectrum, plot_spectrum_with_minima,
#                              plot_grayscale_spectra, plot_pca_variance)
# from config.config import PARAM_GRID_1, PARAM_GRID_2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_redshift_input():
    """
    Get redshift value from user input with validation.
    
    Returns:
        float: Validated redshift value
    """
    while True:
        try:
            redshift = float(input("Which redshift of data are you processing?(e.g., 0.1, 0.3, 2.2, 2.4): "))
            if redshift < 0:
                print("Redshift must be positive. Please try again.")
                continue
            return redshift
        except ValueError:
            print("Please enter a valid number. Try again.")

def main():
    # Get redshift from user input
    redshift = get_redshift_input()
    logger.info(f"Processing data for redshift: {redshift}")
    
    # # Create output directories
    # os.makedirs(f"{redshift}/output/models", exist_ok=True)
    # os.makedirs(f"{redshift}/output/results", exist_ok=True)
    # os.makedirs(f"{redshift}/output/plots", exist_ok=True)
    
    # 1. Load and prepare data
    logger.info(f"Loading data for redshift {redshift}...")
    data_by_physics = load_data(redshift)
    spectra, labels = prepare_dataset(data_by_physics)
    logger.info(f"spectra {spectra.shape}, labels {labels.shape}")
    
    # 2. Perform PCA analysis
    logger.info("Performing PCA analysis...")
    n_components_dict = perform_pca_analysis(spectra)
    
    # 3. Apply PCA transformation for different variance thresholds
    logger.info("\nAvailable variance thresholds for PCA:")
    for variance_key in n_components_dict.keys():
        logger.info(f"- {variance_key}% (will use {n_components_dict[variance_key]} components)")
    

    pca_transformed_spectra = apply_pca_transformation(spectra)

    logger.info(f"Spectra shape after pca transformation: {pca_transformed_spectra.shape}")
    
    # # 4. Extract local minima features
    # logger.info("Extracting local minima features...")
    # minima_features = extract_local_minima_features(spectra)
    
    # # 5. Train and evaluate models for each feature set
    # for variance_key, transformed_data in pca_transformed_data.items():
    #     logger.info(f"Training models for {variance_key}% variance PCA features...")
        
    #     # Prepare training data
    #     X, y = prepare_training_data(transformed_data, labels, size_per_class=2000)
        
    #     # Train with different parameter grids
    #     for grid_id, param_grid in enumerate([PARAM_GRID_1, PARAM_GRID_2], 1):
    #         logger.info(f"Training with parameter grid {grid_id}...")
    #         grid_search, best_params = train_svm_with_cv(X, y, param_grid)
            
    #         # Save results
    #         results_file = f"output/results/grid_search_{variance_key}_grid{grid_id}.xlsx"
    #         save_grid_search_results(grid_search, results_file)
            
    #         # Train final model with best parameters
    #         model_file = f"output/models/svm_{variance_key}_grid{grid_id}.pkl"
    #         final_model = train_final_model(X, y, best_params, model_file)
            
    #         # Evaluate model performance
    #         evaluate_model_performance(final_model, X, y, str(redshift))
    
    # # 6. Analyze results
    # logger.info("Analyzing results...")
    # for results_file in os.listdir("output/results"):
    #     if results_file.endswith(".xlsx"):
    #         file_path = os.path.join("output/results", results_file)
    #         analyze_performance_vs_overfit(file_path)
    #         create_factors_heatmap(file_path)

if __name__ == "__main__":
    main()
