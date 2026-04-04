"""Tests for src/core/cluster.py — K-Means clustering module."""

import numpy as np
import pandas as pd
import pytest
from src.core.cluster import k_sweep, fit_kmeans, compute_cluster_stats


@pytest.fixture
def synthetic_vectors():
    """Create synthetic separability vectors: 100 samples x 24 features."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 24))


class TestKSweep:
    def test_returns_dataframe(self, synthetic_vectors):
        result = k_sweep(synthetic_vectors, k_range=range(2, 5), seed=42)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, synthetic_vectors):
        result = k_sweep(synthetic_vectors, k_range=range(2, 5), seed=42)
        assert set(result.columns) == {'k', 'inertia', 'silhouette'}

    def test_correct_row_count(self, synthetic_vectors):
        result = k_sweep(synthetic_vectors, k_range=range(2, 6), seed=42)
        assert len(result) == 4  # K=2,3,4,5

    def test_inertia_decreases(self, synthetic_vectors):
        result = k_sweep(synthetic_vectors, k_range=range(2, 8), seed=42)
        inertias = result['inertia'].values
        # Inertia should generally decrease as K increases
        assert inertias[0] > inertias[-1]


class TestFitKmeans:
    def test_output_shapes(self, synthetic_vectors):
        labels, centroids = fit_kmeans(synthetic_vectors, k=5, seed=42)
        assert labels.shape == (100,)
        assert centroids.shape == (5, 24)

    def test_label_range(self, synthetic_vectors):
        labels, _ = fit_kmeans(synthetic_vectors, k=5, seed=42)
        assert set(np.unique(labels)).issubset(set(range(5)))

    def test_all_samples_assigned(self, synthetic_vectors):
        labels, _ = fit_kmeans(synthetic_vectors, k=5, seed=42)
        assert len(labels) == 100

    def test_deterministic(self, synthetic_vectors):
        l1, c1 = fit_kmeans(synthetic_vectors, k=5, seed=42)
        l2, c2 = fit_kmeans(synthetic_vectors, k=5, seed=42)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(c1, c2)


class TestComputeClusterStats:
    def test_returns_dataframe(self, synthetic_vectors):
        labels, centroids = fit_kmeans(synthetic_vectors, k=3, seed=42)
        stats = compute_cluster_stats(synthetic_vectors, labels, centroids)
        assert isinstance(stats, pd.DataFrame)

    def test_correct_columns(self, synthetic_vectors):
        labels, centroids = fit_kmeans(synthetic_vectors, k=3, seed=42)
        stats = compute_cluster_stats(synthetic_vectors, labels, centroids)
        expected = {'cluster', 'size', 'pct_total', 'mean_l2', 'max_l2', 'centroid_magnitude'}
        assert set(stats.columns) == expected

    def test_sizes_sum_to_total(self, synthetic_vectors):
        labels, centroids = fit_kmeans(synthetic_vectors, k=3, seed=42)
        stats = compute_cluster_stats(synthetic_vectors, labels, centroids)
        assert stats['size'].sum() == 100

    def test_pct_sums_to_100(self, synthetic_vectors):
        labels, centroids = fit_kmeans(synthetic_vectors, k=3, seed=42)
        stats = compute_cluster_stats(synthetic_vectors, labels, centroids)
        assert abs(stats['pct_total'].sum() - 100.0) < 1e-6
