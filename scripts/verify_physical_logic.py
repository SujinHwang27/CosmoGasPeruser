import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def verify_physical_interpretation():
    # 1. Load data
    print("Loading data for verification...")
    
    # Updated path
    labels_path = "data/feature_discovery/experiments/wavelet_k8_primary/cluster_labels.npy"
    if not os.path.exists(labels_path):
        print(f"Error: Could not find labels at {labels_path}")
        return
        
    labels = np.load(labels_path)
    
    features = []
    for c in range(1, 5):
        path = f"data/processed/wavelet_db8_l6_d12/{c}/data.npy"
        if os.path.exists(path):
            features.append(np.load(path, mmap_mode='r'))
        else:
            print(f"Warning: Could not find features for class {c} at {path}")
    
    if len(features) < 4:
        print("Error: Missing class features. Cannot verify physical logic.")
        return
        
    n_clusters = len(np.unique(labels))
    cluster_stats = []
    
    print("Calculating cluster physical signatures...")
    for k in range(n_clusters):
        indices = np.where(labels == k)[0]
        
        # Calculate mean "feature magnitude" across the classes for these indices
        mags = []
        for c in range(4):
            # Take the mean absolute value of the 512 wavelet coefficients for these indices
            mags.append(np.mean(np.abs(features[c][indices])))
            
        cluster_stats.append({
            'Cluster': k,
            'Size': len(indices),
            'Mean_Wavelet_Mag': np.mean(mags),
            'Class_Variance': np.var(mags)
        })
        
    stats_df = pd.DataFrame(cluster_stats)
    print("\nCluster Physical Stats:")
    print(stats_df.sort_values('Mean_Wavelet_Mag'))
    
    # Save the stats to comparisons
    output_csv = "data/feature_discovery/comparisons/clustering_physical_evidence.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    stats_df.to_csv(output_csv, index=False)
    print(f"Saved physical evidence to {output_csv}")

if __name__ == "__main__":
    verify_physical_interpretation()
