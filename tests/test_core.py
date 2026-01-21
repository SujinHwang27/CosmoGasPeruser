import pytest
import numpy as np
import os
import shutil
from src.core.data import DataIngestor
from src.core.transforms import PCATransform, DCTTransform, FisherTransform

@pytest.fixture
def mock_dataset(tmp_path):
    """Creates a mock Sherwood-style dataset."""
    dataset_dir = tmp_path / "mock_dataset"
    for i in range(1, 5):
        class_dir = dataset_dir / str(i)
        class_dir.mkdir(parents=True)
        # Create 10 samples with 50 features
        data = np.random.rand(10, 50)
        np.save(class_dir / "flux.npy", data)
    return str(dataset_dir)

def test_data_ingestor(mock_dataset):
    ingestor = DataIngestor(mock_dataset, filename="flux.npy", num_classes=4)
    X, y = ingestor.load()
    
    assert X.shape == (40, 50)
    assert len(np.unique(y)) == 4
    assert np.all(np.unique(y) == [1, 2, 3, 4])

def test_pca_transform():
    X = np.random.rand(100, 50)
    pca = PCATransform(n_components=5)
    X_reduced = pca.fit_transform(X)
    
    assert X_reduced.shape == (100, 5)

def test_dct_transform():
    X = np.random.rand(10, 100)
    # Test High Freq mode
    dct_t = DCTTransform(n_coefficients=10, mode="high_freq")
    X_reduced = dct_t.fit_transform(X)
    assert X_reduced.shape == (10, 10)
    
    # Test Full mode
    dct_full = DCTTransform(mode="full")
    X_full = dct_full.fit_transform(X)
    assert X_full.shape == (10, 100)

def test_fisher_transform():
    # 2 classes, 10 samples each, 5 features
    X = np.random.rand(20, 5)
    y = np.array([1]*10 + [2]*10)
    
    # Test scoring mode
    fisher = FisherTransform()
    scores = fisher.fit_transform(X, y)
    assert len(scores) == 5
    
    # Test reduction mode
    fisher_red = FisherTransform(top_k=2)
    X_reduced = fisher_red.fit_transform(X, y)
    assert X_reduced.shape == (20, 2)


