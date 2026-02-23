import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

def compare_wavelet_vs_dct():
    # Load labels
    labels_wavelet = np.load("data/feature_discovery/clustering_results/cluster_labels.npy")
    labels_dct = np.load("data/feature_discovery/clustering_results_dct/cluster_labels.npy")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Index': range(len(labels_wavelet)),
        'Cluster_Wavelet': labels_wavelet,
        'Cluster_DCT': labels_dct
    })
    
    df.to_csv("data/feature_discovery/wavelet_vs_dct_membership.csv", index=False)
    print("Saved membership comparison to data/feature_discovery/wavelet_vs_dct_membership.csv")
    
    # Compute contingency table
    w_unique = np.unique(labels_wavelet)
    d_unique = np.unique(labels_dct)
    
    cm = np.zeros((len(d_unique), len(w_unique)), dtype=int)
    for i, ud in enumerate(d_unique):
        for j, uw in enumerate(w_unique):
            cm[i, j] = np.sum((labels_dct == ud) & (labels_wavelet == uw))
            
    cm_df = pd.DataFrame(cm, 
                         index=[f"DCT_G{i}" for i in d_unique], 
                         columns=[f"Wave_G{j}" for j in w_unique])
    
    print("\nContingency Table (Rows=DCT, Cols=Wavelet):")
    print(cm_df)
    
    # Overlap Percentages
    cm_perc = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
    print("\nOverlap Percentages (Row-wise: % of DCT group coming from Wavelet groups):")
    print(cm_perc.round(1))

if __name__ == "__main__":
    compare_wavelet_vs_dct()
