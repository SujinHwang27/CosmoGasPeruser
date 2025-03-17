"""
SVM model training and evaluation functionality.
"""

import numpy as np
from typing import Dict, Tuple, Any
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import logging
import pandas as pd
from ..config.config import (TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE, 
                           CV_FOLDS, PARAM_GRID_1, PARAM_GRID_2)

logger = logging.getLogger(__name__)

def prepare_training_data(X: np.ndarray, 
                         y: np.ndarray,
                         size_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data by sampling from each class.
    
    Args:
        X: Feature matrix
        y: Labels
        size_per_class: Number of samples to use from each class
        
    Returns:
        Tuple of (X, y) containing features and labels
    """
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain at least 2 classes for classification")
    
    X_balanced = []
    y_balanced = []
    
    unique_classes = np.unique(y)
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) < size_per_class:
            logger.warning(f"Class {class_label} has fewer samples ({len(class_indices)}) "
                         f"than requested size_per_class ({size_per_class}). "
                         "Using all available samples.")
            selected_indices = class_indices
        else:
            selected_indices = np.random.choice(class_indices, 
                                             size=size_per_class, 
                                             replace=False)
        X_balanced.append(X[selected_indices])
        y_balanced.extend([class_label] * len(selected_indices))

    return np.vstack(X_balanced), np.array(y_balanced).astype(int)

def train_svm_with_cv(X: np.ndarray, 
                      y: np.ndarray, 
                      param_grid: Dict[str, Any]) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Train SVM model with cross-validation and hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Labels
        param_grid: Grid of parameters to search
        
    Returns:
        Tuple of (GridSearchCV object, best parameters)
    """
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain at least 2 classes for classification")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TRAIN_TEST_SPLIT_RATIO, 
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    logger.info(f"Testing data shapes - X: {X_test.shape}, y: {y_test.shape}")
    
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=KFold(n_splits=min(CV_FOLDS, len(np.unique(y_train))), shuffle=True),
        scoring='accuracy',
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search, best_params

def save_grid_search_results(grid_search: GridSearchCV, 
                           output_file: str) -> None:
    """
    Save grid search results to Excel file.
    
    Args:
        grid_search: GridSearchCV results
        output_file: Path to output Excel file
    """
    df = pd.DataFrame.from_dict(grid_search.cv_results_)
    df['overfit'] = df['mean_train_score'] - df['mean_test_score']
    
    condition_1 = df['mean_train_score'] == 1
    condition_2 = abs(df['mean_test_score'] - df['mean_train_score']) == \
                 abs(df['mean_test_score'] - df['mean_train_score']).min()
    
    df['highlight'] = ''
    df.loc[condition_1, 'highlight'] = 'red'
    df.loc[condition_2, 'highlight'] = 'green'
    
    df.to_excel(output_file, index=False)
    logger.info(f"Grid search results saved to {output_file}")

def train_final_model(X: np.ndarray, 
                     y: np.ndarray, 
                     best_params: Dict[str, Any],
                     model_path: str) -> SVC:
    """
    Train final model with best parameters and save it.
    
    Args:
        X: Feature matrix
        y: Labels
        best_params: Best parameters from grid search
        model_path: Path to save the model
        
    Returns:
        Trained SVM model
    """
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain at least 2 classes for classification")
    
    model = SVC(**best_params)
    model.fit(X, y)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return model
