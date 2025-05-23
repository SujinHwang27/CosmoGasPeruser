"""
SVM model training and evaluation functionality.
"""

import numpy as np
from typing import Dict, Tuple, Any
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import logging
import pandas as pd
from pathlib import Path
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config using absolute import
from classifier_SVM.config.config import (TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE, 
                                       CV_FOLDS, CLASS_SIZE, DATA_SIZE,NCOMP,
                                       PARAM_GRID_DICT)

logger = logging.getLogger(__name__)



def train_svm_with_cv(X: np.ndarray, 
                      y: np.ndarray, 
                      param_grid_no: int) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Train SVM model with cross-validation and hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Labels
        param_grid: Grid of parameters to search
        
    Returns:
        Tuple of (GridSearchCV object, best parameters)
    """
    # Create output directory for grid search results
    output_dir = Path("output/grid_search_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_search = GridSearchCV(
        SVC(),
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

def save_grid_search_results(grid_search: GridSearchCV, 
                           output_file: str) -> None:
    """
    Save grid search results to Excel file.
    
    Args:
        grid_search: GridSearchCV results
        output_file: Path to output Excel file
    """
    # Convert grid search results to DataFrame
    df = pd.DataFrame.from_dict(grid_search.cv_results_)
    
    # Calculate overfit score
    df['overfit'] = df['mean_train_score'] - df['mean_test_score']
    
    # Add highlighting conditions
    condition_1 = df['mean_train_score'] == 1
    condition_2 = abs(df['mean_test_score'] - df['mean_train_score']) == \
                 abs(df['mean_test_score'] - df['mean_train_score']).min()
    
    df['highlight'] = ''
    df.loc[condition_1, 'highlight'] = 'red'
    df.loc[condition_2, 'highlight'] = 'green'
    
    # Save to Excel with proper engine
    df.to_excel(output_file, index=False, engine='openpyxl')
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
    
    model = SVC(**best_params)
    model.fit(X, y)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return model

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, param_grid_no):
    """
    Train and evaluate SVM model with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_splits: Number of folds for cross-validation
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Create output directory for models if it doesn't exist
    output_dir = Path("output/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select parameter grid based on param_grid_no
    
    param_grid = PARAM_GRID_DICT.get(int(param_grid_no), 3)  

    grid_search, best_params = train_svm_with_cv(X_train, y_train, param_grid_no)
    
    model = train_final_model(X_train, y_train, best_params, output_dir / f"final_svm_model_{DATA_SIZE}_{NCOMP}_{param_grid_no}.pkl")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Generate classification reports
    train_report = classification_report(y_train, y_train_pred)
    test_report = classification_report(y_test, y_test_pred)
    
    # Generate confusion matrices
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Log results
    logger.info("\nCross-validation results:")
    logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    logger.info("\nTraining set results:")
    logger.info(f"Accuracy: {train_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(train_report)
    
    logger.info("\nTest set results:")
    logger.info(f"Accuracy: {test_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(test_report)
    
    # Return results dictionary
    results = {
        'grid_search': grid_search,
        'best_params': best_params,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_report': train_report,
        'test_report': test_report,
        'train_conf_matrix': train_conf_matrix,
        'test_conf_matrix': test_conf_matrix,
        'model': model
    }
    
    return results
