"""
Test cases for feature_extraction.py
"""

import unittest
import numpy as np
from src.feature_extraction import (find_local_minima, calculate_minimum_width,
                                  perform_pca_analysis, apply_pca_transformation,
                                  extract_local_minima_features)
from config.config import EXPLAINED_VARIANCE

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create a simple spectrum with known local minima
        self.spectrum = np.array([1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.2, 0.4, 1.0])
        self.expected_minima_indices = np.array([2, 6])
        self.expected_minima_values = np.array([0.3, 0.2])
        
        # Create random spectra for PCA and feature extraction tests
        self.n_samples = 100
        self.n_features = 50
        self.spectra = np.random.randn(self.n_samples, self.n_features)

    def test_find_local_minima(self):
        """Test local minima detection."""
        indices, values = find_local_minima(self.spectrum)
        
        # Check if correct minima are found
        np.testing.assert_array_equal(indices, self.expected_minima_indices)
        np.testing.assert_array_equal(values, self.expected_minima_values)
        
        # Test with flat spectrum
        flat_spectrum = np.ones(10)
        indices, values = find_local_minima(flat_spectrum)
        self.assertEqual(len(indices), 0)
        self.assertEqual(len(values), 0)

    def test_calculate_minimum_width(self):
        """Test minimum width calculation."""
        # Test with known minimum
        width = calculate_minimum_width(self.spectrum, 2)
        self.assertEqual(width, 3)  # Width from index 1 to 3
        
        # Test with edge minimum
        width = calculate_minimum_width(self.spectrum, 6)
        self.assertEqual(width, 3)  # Width from index 5 to 7
        
        # Test with invalid index
        with self.assertRaises(IndexError):
            calculate_minimum_width(self.spectrum, 10)

    def test_perform_pca_analysis(self):
        """Test PCA analysis."""
        n_components_dict = perform_pca_analysis(self.spectra)
        
        # Check if dictionary contains all variance thresholds
        self.assertEqual(set(n_components_dict.keys()), set(EXPLAINED_VARIANCE.keys()))
        
        # Check if components are positive integers
        for n_components in n_components_dict.values():
            self.assertIsInstance(n_components, int)
            self.assertGreater(n_components, 0)
            self.assertLessEqual(n_components, self.n_features)
        
        # Check if components are in descending order
        components = list(n_components_dict.values())
        self.assertEqual(components, sorted(components, reverse=True))

    def test_apply_pca_transformation(self):
        """Test PCA transformation."""
        # First perform PCA analysis
        n_components_dict = perform_pca_analysis(self.spectra)
        
        # Test transformation for each variance threshold
        for variance_key, n_components in n_components_dict.items():
            transformed_data = apply_pca_transformation(self.spectra, n_components)
            
            # Check shapes
            self.assertEqual(transformed_data.shape[0], self.n_samples)
            self.assertEqual(transformed_data.shape[1], n_components)
            
            # Check if data is centered
            np.testing.assert_array_almost_equal(
                np.mean(transformed_data, axis=0),
                np.zeros(n_components)
            )

    def test_extract_local_minima_features(self):
        """Test local minima feature extraction."""
        num_samples = 10
        sample_size = 5
        
        features = extract_local_minima_features(
            self.spectra,
            num_samples=num_samples,
            sample_size=sample_size
        )
        
        # Check shape
        self.assertEqual(features.shape[0], num_samples)
        self.assertEqual(features.shape[1], 6)  # 6 statistical features
        
        # Check if features are within expected ranges
        self.assertTrue(np.all(features[:, 0] >= 0))  # mean
        self.assertTrue(np.all(features[:, 1] >= 0))  # min
        self.assertTrue(np.all(features[:, 2] >= 0))  # max
        self.assertTrue(np.all(features[:, 3] >= 0))  # std
        self.assertTrue(np.all(features[:, 4] >= 0))  # median
        self.assertTrue(np.all(features[:, 5] >= 0))  # count

    def test_extract_local_minima_features_empty(self):
        """Test feature extraction with empty spectra."""
        empty_spectra = np.array([]).reshape(0, 10)
        with self.assertRaises(ValueError):
            extract_local_minima_features(empty_spectra)

if __name__ == '__main__':
    unittest.main() 