import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def verify_physical_interpretation():
    # 1. Load data
    # We need the absorption field A used for processing
    # and the cluster labels
    print("Loading data for verification...")
    
    # Let's load the labels (K8 is our primary)
    labels = np.load("data/feature_discovery/clustering_results/cluster_labels.npy")
    
    # We need to see the "typical" absorption value for indices in each cluster.
    # The 'micro_classifier_params.npy' contains the behavioral vectors,
    # but the 'process_signals.py' script generated the actual features.
    # However, a simpler way is to look at the 'A' field for the 4 representative 
    # samples used to train the SVMs.
    
    # Let's load Class 1-4 data from the processed wavelet directory
    # (Actually, let's just use the raw flux from one sample per class to keep it simple)
    # The process_signals script saved the transformed data. Let's use that.
    
    features = []
    for c in range(1, 5):
        features.append(np.load(f"data/processed/wavelet_db8_l6_d12/{c}/data.npy", mmap_mode='r'))
    
    # features[c] is (16384, 512)
    # The micro-classifier i was trained on features[0][i], features[1][i], ...
    
    n_clusters = len(np.unique(labels))
    cluster_stats = []
    
    print("Calculating cluster physical signatures...")
    for k in range(n_clusters):
        indices = np.where(labels == k)[0]
        
        # Calculate mean "feature magnitude" across the classes for these indices
        # Since we use Wavelets, a high absolute value indicates a strong signal/fluctuation 
        # at that specific scale.
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
    
    # Save the stats
    stats_df.to_csv("data/feature_discovery/clustering_physical_evidence.csv", index=False)

if __name__ == "__main__":
    verify_physical_interpretation()
