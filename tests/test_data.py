"""Tests for src/core/data.py — data ingestion and normalization."""

import numpy as np
import os
import pytest
import tempfile
from src.core.data import DataIngestor, SignalClusteringData, DataSummary


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock class data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(42)
        for i in range(1, 5):
            class_dir = os.path.join(tmpdir, str(i))
            os.makedirs(class_dir)
            data = rng.standard_normal((20, 2048)).astype(np.float64)
            np.save(os.path.join(class_dir, "flux.npy"), data)
            np.save(os.path.join(class_dir, "data.npy"), data)
        yield tmpdir


class TestDataIngestor:
    def test_load_shapes(self, mock_data_dir):
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        X, y = ingestor.load()
        assert X.shape == (80, 2048)  # 4 classes x 20 samples
        assert y.shape == (80,)

    def test_class_labels(self, mock_data_dir):
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        _, y = ingestor.load()
        unique = np.unique(y)
        np.testing.assert_array_equal(unique, [1, 2, 3, 4])

    def test_load_per_class(self, mock_data_dir):
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        X_list, y = ingestor.load_per_class()
        assert len(X_list) == 4
        assert all(x.shape == (20, 2048) for x in X_list)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ingestor = DataIngestor("/nonexistent/path", filename="flux.npy")
            ingestor.load()

    def test_fallback_to_data_npy(self, mock_data_dir):
        """Test fallback when flux.npy doesn't exist but data.npy does."""
        # Remove flux.npy, keep data.npy
        for i in range(1, 5):
            os.remove(os.path.join(mock_data_dir, str(i), "flux.npy"))
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        X, y = ingestor.load()
        assert X.shape == (80, 2048)


class TestSignalClusteringData:
    def test_normalize_wavelet_per_level_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2048))
        result = SignalClusteringData.normalize_wavelet_per_level(X)
        assert result.shape == X.shape

    def test_normalize_wavelet_per_level_zero_mean(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2048))
        result = SignalClusteringData.normalize_wavelet_per_level(X)
        # Each wavelet level should have approximately zero mean
        level_boundaries = {
            'D1': (0, 1024), 'D2': (1024, 1536), 'D3': (1536, 1792),
            'D4': (1792, 1920), 'D5': (1920, 1984), 'D6': (1984, 2016),
            'A6': (2016, 2048),
        }
        for name, (start, end) in level_boundaries.items():
            level_mean = np.mean(result[:, start:end])
            assert abs(level_mean) < 1e-10, f"Level {name} mean={level_mean}"

    def test_normalize_preserves_no_nan(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2048))
        result = SignalClusteringData.normalize_wavelet_per_level(X)
        assert not np.any(np.isnan(result))


class TestDataSummary:
    def test_summary_fields(self):
        summary = DataSummary(
            shape=(100, 24), dtype="float64",
            min=-3.0, max=3.0, mean=0.0, std=1.0,
            has_nan=False, has_inf=False
        )
        assert summary.shape == (100, 24)
        assert not summary.has_nan
