"""
Test cases for model.py
"""

import unittest
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from src.model import (prepare_training_data, train_svm_with_cv,
                      save_grid_search_results, train_final_model)
import os
import tempfile
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create random test data
        self.n_samples = 100
        self.n_features = 10
        self.n_classes = 3
        
        # Create balanced dataset
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.repeat(np.arange(self.n_classes), self.n_samples // self.n_classes)
        
        # Create unbalanced dataset
        self.X_unbalanced = np.random.randn(self.n_samples, self.n_features)
        self.y_unbalanced = np.concatenate([
            np.repeat(0, 60),
            np.repeat(1, 30),
            np.repeat(2, 10)
        ])

    def test_prepare_training_data_balanced(self):
        """Test training data preparation with balanced dataset."""
        size_per_class = 20
        X_balanced, y_balanced = prepare_training_data(self.X, self.y, size_per_class)
        
        # Check shapes
        self.assertEqual(X_balanced.shape[0], size_per_class * self.n_classes)
        self.assertEqual(X_balanced.shape[1], self.n_features)
        self.assertEqual(len(y_balanced), size_per_class * self.n_classes)
        
        # Check class balance
        for i in range(self.n_classes):
            self.assertEqual(np.sum(y_balanced == i), size_per_class)

    def test_prepare_training_data_unbalanced(self):
        """Test training data preparation with unbalanced dataset."""
        size_per_class = 20
        X_balanced, y_balanced = prepare_training_data(
            self.X_unbalanced, self.y_unbalanced, size_per_class
        )
        
        # Check if warning is logged for class with insufficient samples
        # Note: This would require mocking the logger to properly test
        
        # Check shapes
        self.assertEqual(X_balanced.shape[0], 100)  # All available samples
        self.assertEqual(X_balanced.shape[1], self.n_features)
        self.assertEqual(len(y_balanced), 100)

    def test_prepare_training_data_insufficient_classes(self):
        """Test error when insufficient classes are present."""
        with self.assertRaises(ValueError) as context:
            prepare_training_data(self.X, np.ones(self.n_samples), size_per_class=20)
        self.assertIn("Dataset must contain at least 2 classes", str(context.exception))

    def test_train_svm_with_cv(self):
        """Test SVM training with cross-validation."""
        param_grid = {
            'C': [1, 10],
            'kernel': ['linear']
        }
        
        grid_search, best_params = train_svm_with_cv(self.X, self.y, param_grid)
        
        # Check if GridSearchCV object is returned
        self.assertIsInstance(grid_search, GridSearchCV)
        # Check if best parameters are returned
        self.assertIsInstance(best_params, dict)
        # Check if best parameters are in param_grid
        for param, value in best_params.items():
            self.assertIn(value, param_grid[param])

    def test_train_svm_with_cv_insufficient_classes(self):
        """Test error when insufficient classes are present."""
        param_grid = {'C': [1], 'kernel': ['linear']}
        with self.assertRaises(ValueError) as context:
            train_svm_with_cv(self.X, np.ones(self.n_samples), param_grid)
        self.assertIn("Dataset must contain at least 2 classes", str(context.exception))

    def test_save_grid_search_results(self):
        """Test saving grid search results."""
        # Create dummy grid search results
        param_grid = {'C': [1, 10], 'kernel': ['linear']}
        grid_search = GridSearchCV(SVC(), param_grid, cv=2)
        grid_search.fit(self.X, self.y)
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            output_file = tmp.name
        
        try:
            save_grid_search_results(grid_search, output_file)
            
            # Check if file exists
            self.assertTrue(os.path.exists(output_file))
            
            # Check if file contains expected columns
            df = pd.read_excel(output_file)
            expected_columns = ['mean_test_score', 'mean_train_score', 'overfit', 'highlight']
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # Check if highlight column contains expected values
            self.assertTrue(all(df['highlight'].isin(['', 'red', 'green'])))
            
        finally:
            # Clean up
            os.unlink(output_file)

    def test_train_final_model(self):
        """Test final model training."""
        best_params = {'C': 1, 'kernel': 'linear'}
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            model = train_final_model(self.X, self.y, best_params, model_path)
            
            # Check if model is trained
            self.assertIsInstance(model, SVC)
            self.assertTrue(hasattr(model, 'support_'))
            
            # Check if model file exists
            self.assertTrue(os.path.exists(model_path))
            
        finally:
            # Clean up
            os.unlink(model_path)

    def test_train_final_model_insufficient_classes(self):
        """Test error when insufficient classes are present."""
        best_params = {'C': 1, 'kernel': 'linear'}
        with self.assertRaises(ValueError) as context:
            train_final_model(self.X, np.ones(self.n_samples), best_params, 'dummy.pkl')
        self.assertIn("Dataset must contain at least 2 classes", str(context.exception))

if __name__ == '__main__':
    unittest.main() 