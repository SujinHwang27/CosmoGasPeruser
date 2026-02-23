import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

def analyze_overlap():
    labels_k8 = np.load("data/feature_discovery/clustering_results/cluster_labels.npy")
    labels_k5 = np.load("data/feature_discovery/clustering_results_k5/cluster_labels.npy")
    
    # Create a DataFrame for easy comparison
    df = pd.DataFrame({
        'Index': range(len(labels_k8)),
        'Cluster_K8': labels_k8,
        'Cluster_K5': labels_k5
    })
    
    # Save the complete membership table
    df.to_csv("data/feature_discovery/cluster_membership_comparison.csv", index=False)
    print("Saved complete membership table to data/feature_discovery/cluster_membership_comparison.csv")
    
    # Compute contingency table
    # We want a table where rows are K5 and columns are K8
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
    
    # Percentages for the report (How much of K5 Group X came from K8 Group Y)
    cm_perc = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
    print("\nOverlap Percentages (Row-wise: % of K5 group coming from K8 groups):")
    print(cm_perc.round(1))
    
    # How much of K8 Group Y was absorbed into K5 Group X
    cm_perc_k8 = cm_df.div(cm_df.sum(axis=0), axis=1) * 100
    print("\nAbsorption Percentages (Column-wise: % of K8 group moving into K5 groups):")
    print(cm_perc_k8.round(1))

if __name__ == "__main__":
    analyze_overlap()
