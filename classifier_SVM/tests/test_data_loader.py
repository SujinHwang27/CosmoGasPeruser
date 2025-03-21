"""
Test cases for data_loader.py
"""

import unittest
import numpy as np
import os
import shutil
import tempfile
from src.data_loader import get_physics_values, load_data, prepare_dataset
from config.config import DATA_DIR

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with temporary test data."""
        # Create temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test data structure
        cls.test_redshift = 0.1
        cls.physics_values = ['1', '2', '3']
        
        for physics in cls.physics_values:
            folder_name = f"{physics}_{cls.test_redshift}"
            folder_path = os.path.join(cls.test_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create dummy flux data
            flux_data = np.random.rand(1000, 194)  # 1000 spectra, 194 wavelengths
            np.save(os.path.join(folder_path, "flux.npy"), flux_data)
        
        # Override DATA_DIR for testing
        cls.original_data_dir = DATA_DIR
        import src.data_loader
        src.data_loader.DATA_DIR = cls.test_dir

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Restore original DATA_DIR
        import src.data_loader
        src.data_loader.DATA_DIR = cls.original_data_dir
        
        # Remove temporary directory
        shutil.rmtree(cls.test_dir)

    def test_get_physics_values(self):
        """Test physics values discovery."""
        values = get_physics_values(self.test_redshift)
        
        # Check if all expected physics values are found
        self.assertEqual(set(values), set(self.physics_values))
        # Check if values are sorted
        self.assertEqual(values, sorted(values))
        # Check if values are strings
        self.assertTrue(all(isinstance(v, str) for v in values))

    def test_get_physics_values_insufficient(self):
        """Test error when insufficient physics values are found."""
        # Remove one physics value folder
        folder_to_remove = os.path.join(self.test_dir, f"{self.physics_values[0]}_{self.test_redshift}")
        shutil.rmtree(folder_to_remove)
        
        # Test if ValueError is raised
        with self.assertRaises(ValueError) as context:
            get_physics_values(self.test_redshift)
        self.assertIn("Dataset must contain at least 2 physics values", str(context.exception))

    def test_get_physics_values_invalid_redshift(self):
        """Test error when invalid redshift is provided."""
        with self.assertRaises(ValueError) as context:
            get_physics_values(-1.0)
        self.assertIn("Dataset must contain at least 2 physics values", str(context.exception))

    def test_load_data(self):
        """Test data loading functionality."""
        data = load_data(self.test_redshift)
        
        # Check if all physics values are present
        self.assertEqual(set(data.keys()), set(self.physics_values))
        
        # Check data structure for each physics value
        for physics, flux_data in data.items():
            self.assertIsInstance(flux_data, np.ndarray)
            self.assertEqual(flux_data.shape[0], 10)  # 10 spectra
            self.assertEqual(flux_data.shape[1], 100)  # 100 wavelengths

    def test_load_data_missing_file(self):
        """Test handling of missing flux files."""
        # Remove flux.npy from one folder
        folder_to_modify = os.path.join(self.test_dir, f"{self.physics_values[0]}_{self.test_redshift}")
        os.remove(os.path.join(folder_to_modify, "flux.npy"))
        
        data = load_data(self.test_redshift)
        # Check if the physics value with missing file is not in the data
        self.assertNotIn(self.physics_values[0], data)

    def test_load_data_invalid_redshift(self):
        """Test error when invalid redshift is provided."""
        with self.assertRaises(ValueError) as context:
            load_data(-1.0)
        self.assertIn("Dataset must contain at least 2 physics values", str(context.exception))

    def test_prepare_dataset(self):
        """Test dataset preparation."""
        # Create test data
        test_data = {
            '1': np.random.rand(5, 100),
            '2': np.random.rand(3, 100),
            '3': np.random.rand(4, 100)
        }
        
        spectra, labels = prepare_dataset(test_data)
        
        # Check shapes
        self.assertEqual(spectra.shape[0], 12)  # 5 + 3 + 4 spectra
        self.assertEqual(spectra.shape[1], 100)  # 100 wavelengths
        self.assertEqual(len(labels), 12)  # Same as number of spectra
        
        # Check label types
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        
        # Check if labels match the data
        label_counts = {str(label): np.sum(labels == label) for label in np.unique(labels)}
        for physics, count in label_counts.items():
            self.assertEqual(count, test_data[physics].shape[0])

    def test_prepare_dataset_empty(self):
        """Test handling of empty dataset."""
        empty_data = {}
        with self.assertRaises(ValueError):
            prepare_dataset(empty_data)

if __name__ == '__main__':
    unittest.main() 