"""
Model evaluation and analysis functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)

def analyze_performance_vs_overfit(results_file: str) -> None:
    """
    Analyze and plot performance vs overfit from grid search results.
    
    Args:
        results_file: Path to Excel file containing grid search results
    """
    sheets = pd.ExcelFile(results_file).sheet_names
    
    for sheet in sheets:
        df = pd.read_excel(results_file, sheet_name=sheet)
        
        if 'overfit' not in df.columns or 'mean_test_score' not in df.columns:
            logger.warning(f"Skipping sheet {sheet}: required columns not found.")
            continue
        
        plt.figure(figsize=(8, 5))
        plt.scatter(df['mean_test_score'], 
                   df['overfit'], 
                   alpha=0.7, 
                   c='blue', 
                   label='Overfit vs Test Score')
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='No Overfit Line')
        
        plt.title(f"Overfit vs Mean Test Score for {sheet}")
        plt.xlabel("Mean Test Score")
        plt.ylabel("Overfit (Train - Test Score)")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

def create_factors_heatmap(results_file: str) -> None:
    """
    Create and plot correlation heatmaps for different factors.
    
    Args:
        results_file: Path to Excel file containing grid search results
    """
    sheet_names = ['redshift_0.1', 'redshift_0.3', 'redshift_2.2', 'redshift_2.4']
    columns_to_include = [
        'param_C', 'param_gamma', 'param_kernel', 
        'mean_test_score', 'mean_train_score', 
        'overfit', 'n_comp', 'data_size'
    ]
    
    for sheet in sheet_names:
        try:
            data = pd.read_excel(results_file, sheet_name=sheet)
            selected_data = data[columns_to_include]
            selected_data = pd.get_dummies(selected_data, drop_first=False)
            correlation_matrix = selected_data.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       fmt=".2f", 
                       cbar=True)
            plt.title(f"Factors correlation Heatmap for {sheet}")
            plt.show()
        except Exception as e:
            logger.error(f"Error processing sheet {sheet}: {e}")

def evaluate_model_performance(model: Any, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray,
                             redshift: str) -> None:
    """
    Evaluate model performance and display metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        redshift: Current redshift value
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model Performance for redshift {redshift}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'],
                yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
    plt.title(f'Confusion Matrix - Redshift {redshift}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def analyze_feature_importance(grid_search: GridSearchCV, 
                             feature_names: list,
                             redshift: str) -> None:
    """
    Analyze and plot feature importance from the best model.
    
    Args:
        grid_search: Fitted GridSearchCV object
        feature_names: List of feature names
        redshift: Current redshift value
    """
    if hasattr(grid_search.best_estimator_, 'coef_'):
        importance = np.abs(grid_search.best_estimator_.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f'Feature Importance - Redshift {redshift}')
        plt.show()
