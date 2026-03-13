"""
Clustering module for signal clustering analysis.

Performs K-Means clustering on separability fingerprints with
K-sweep for elbow detection and silhouette analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List, Dict


def k_sweep(fingerprints: np.ndarray, k_range: range, seed: int = 42, n_init: int = 20) -> pd.DataFrame:
    """
    Run K-Means sweep over a range of K values.

    Args:
        fingerprints: Input data, shape (n_samples, n_features)
        k_range: Range of K values to sweep
        seed: Random seed
        n_init: Number of initializations

    Returns:
        DataFrame with columns [k, inertia, silhouette]
    """
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = kmeans.fit_predict(fingerprints)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(fingerprints, labels)

        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette': silhouette
        })

        print(f"  K={k}: inertia={inertia:.2f}, silhouette={silhouette:.4f}")

    return pd.DataFrame(results)


def fit_kmeans(fingerprints: np.ndarray, k: int, seed: int = 42, n_init: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit KMeans with specified K.

    Args:
        fingerprints: Input data
        k: Number of clusters
        seed: Random seed
        n_init: Number of initializations

    Returns:
        (labels, centroids) - cluster labels and cluster centroids
    """
    scaler = StandardScaler()
    fingerprints_scaled = scaler.fit_transform(fingerprints)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = kmeans.fit_predict(fingerprints_scaled)

    return labels, kmeans.cluster_centers_


def compute_cluster_stats(fingerprints: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> pd.DataFrame:
    """
    Compute statistics for each cluster.

    Args:
        fingerprints: Original fingerprints (unscaled)
        labels: Cluster labels
        centroids: Cluster centroids (in scaled space)

    Returns:
        DataFrame with per-cluster statistics
    """
    stats = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_points = fingerprints[mask]

        # Distance to centroid in original space
        # Note: centroids are in scaled space, compute distances in original for consistency
        distances = np.linalg.norm(cluster_points - centroids[label], axis=1)

        stats.append({
            'cluster': label,
            'size': int(np.sum(mask)),
            'pct_total': float(100 * np.sum(mask) / len(labels)),
            'mean_l2': float(np.mean(distances)),
            'max_l2': float(np.max(distances)),
            'centroid_magnitude': float(np.linalg.norm(centroids[label]))
        })

    return pd.DataFrame(stats)


def contingency_matrix(labels_a: np.ndarray, labels_b: np.ndarray, k_a: int, k_b: int) -> np.ndarray:
    """
    Compute contingency matrix between two label assignments.

    Args:
        labels_a: First set of labels
        labels_b: Second set of labels
        k_a: Number of clusters in first set
        k_b: Number of clusters in second set

    Returns:
        Contingency matrix of shape (k_a, k_b)
    """
    contingency = np.zeros((k_a, k_b), dtype=int)
    for i in range(len(labels_a)):
        contingency[labels_a[i], labels_b[i]] += 1
    return contingency
