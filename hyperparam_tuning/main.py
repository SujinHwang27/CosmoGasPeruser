# Hyperparameter Tuning before training on the full training data
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import logging
import pandas as pd
from pathlib import Path
import sys

# # Add the project root to Python path
# project_root = str(Path(__file__).parent.parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# Import config using absolute import
from classifier_SVM.config.config import (TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE, 
                                       CV_FOLDS, CLASS_SIZE, DATA_SIZE,NCOMP,
                                       PARAM_GRID_DICT)

logger = logging.getLogger(__name__)



def hyperparam_tuning_with_cv(X: np.ndarray, 
                      y: np.ndarray, 
                      param_grid_no: int, model) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Train with cross-validation and hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Labels
        param_grid: Grid of parameters to search
        model: Model to train
        
    Returns:
        Tuple of (GridSearchCV object, best parameters)
    """
    # Create output directory for grid search results
    output_dir = Path("hyperparam_tuning/grid_search_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_search = GridSearchCV(
        model(),
        PARAM_GRID_DICT.get(int(param_grid_no)),
        cv=KFold(n_splits=CV_FOLDS, shuffle=True),
        scoring='accuracy',
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Performing grid search...")
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Save grid search results with proper file extension
    output_file = output_dir / f"grid_search_results_{DATA_SIZE}_{NCOMP}_{param_grid_no}.xlsx"
    save_grid_search_results(grid_search=grid_search, output_file=str(output_file))

    return grid_search, best_params