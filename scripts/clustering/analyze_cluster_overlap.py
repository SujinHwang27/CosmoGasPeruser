import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

def analyze_overlap():
    # Use reorganized paths
    path_k8 = "data/feature_discovery/experiments/wavelet_k8_primary/cluster_labels.npy"
    path_k5 = "data/feature_discovery/experiments/wavelet_k5_test/cluster_labels.npy"
    output_csv = "data/feature_discovery/comparisons/k_stability/cluster_membership_comparison.csv"
    
    if not os.path.exists(path_k8) or not os.path.exists(path_k5):
        print(f"Error: Could not find labels at {path_k8} or {path_k5}")
        return

    labels_k8 = np.load(path_k8)
    labels_k5 = np.load(path_k5)
    
    # Create a DataFrame for easy comparison
    df = pd.DataFrame({
        'Index': range(len(labels_k8)),
        'Cluster_K8': labels_k8,
        'Cluster_K5': labels_k5
    })
    
    # Save the complete membership table
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved complete membership table to {output_csv}")
    
    # Compute contingency table
    k5_unique = np.unique(labels_k5)
    k8_unique = np.unique(labels_k8)
    
    cm = np.zeros((len(k5_unique), len(k8_unique)), dtype=int)
    for i, u5 in enumerate(k5_unique):
        for j, u8 in enumerate(k8_unique):
            cm[i, j] = np.sum((labels_k5 == u5) & (labels_k8 == u8))
    
    # Format and print the contingency table
    cm_df = pd.DataFrame(cm, 
                         index=[f"K5_G{i}" for i in k5_unique], 
                         columns=[f"K8_G{j}" for j in k8_unique])
    
    print("\nContingency Table (Overlap Counts):")
    print(cm_df)
    
    # Percentages for the report
    cm_perc = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
    print("\nOverlap Percentages (Row-wise: % of K5 group coming from K8 groups):")
    print(cm_perc.round(1))
    
    cm_perc_k8 = cm_df.div(cm_df.sum(axis=0), axis=1) * 100
    print("\nAbsorption Percentages (Column-wise: % of K8 group moving into K5 groups):")
    print(cm_perc_k8.round(1))

if __name__ == "__main__":
    analyze_overlap()
