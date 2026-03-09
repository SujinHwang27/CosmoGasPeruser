import numpy as np
import os
from collections import Counter
import argparse

def survey_clusters(labels_path, params_path, name):
    print(f"\n========================================")
    print(f" Surveying: {name}")
    print(f"========================================")
    if not os.path.exists(labels_path) or not os.path.exists(params_path):
        print(f"[!] Missing files for {name}. Ensure both labels and params exist.")
        return

    labels = np.load(labels_path)
    params = np.load(params_path, mmap_mode='r')
    
    n_points = len(labels)
    n_features = params.shape[1]
    n_clusters = len(np.unique(labels))
    
    print(f"Total points: {n_points}")
    print(f"Feature dimension: {n_features}")
    print(f"Number of clusters (k): {n_clusters}")
    print("-" * 40)
    
    counts = Counter(labels)
    for k in sorted(counts.keys()):
        cluster_mask = (labels == k)
        cluster_points = params[cluster_mask]
        
        pct = (counts[k] / n_points) * 100
        
        centroid = np.mean(cluster_points, axis=0)
        std_dev = np.std(cluster_points, axis=0)
        
        # Calculate distance of points in this cluster to its centroid
        distances_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
        avg_dist = np.mean(distances_to_centroid)
        max_dist = np.max(distances_to_centroid)
        
        print(f"Cluster {k} | Size: {counts[k]:>5d} ({pct:>5.1f}%) | "
              f"Mean L2 to Centroid: {avg_dist:.2f} | Max L2: {max_dist:.2f}")
        
        # Summarize centroid structure (mean of absolute components)
        print(f"  -> Centroid magnitude (L2): {np.linalg.norm(centroid):.2f}")
        print(f"  -> Top 3 prominent features (indices): {np.argsort(np.abs(centroid))[-3:][::-1]}")
    print("\n")

if __name__ == "__main__":
    # Wavelet K=8
    survey_clusters(
        'data/feature_discovery/experiments/wavelet_k8_primary/cluster_labels.npy',
        'data/feature_discovery/base_data/params_wavelet.npy',
        'Wavelet K=8'
    )
    # Wavelet K=5
    survey_clusters(
        'data/feature_discovery/experiments/wavelet_k5_test/cluster_labels.npy',
        'data/feature_discovery/base_data/params_wavelet.npy',
        'Wavelet K=5'
    )
    # DCT K=8
    survey_clusters(
        'data/feature_discovery/experiments/dct_k8_test/cluster_labels.npy',
        'data/feature_discovery/base_data/params_dct.npy',
        'DCT K=8'
    )
