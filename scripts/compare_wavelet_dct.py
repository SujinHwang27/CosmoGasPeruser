import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

def compare_wavelet_vs_dct():
    # Use reorganized paths
    path_wavelet = "data/feature_discovery/experiments/wavelet_k8_primary/cluster_labels.npy"
    path_dct = "data/feature_discovery/experiments/dct_k8_test/cluster_labels.npy"
    output_csv = "data/feature_discovery/comparisons/transform_invariance/wavelet_vs_dct_membership.csv"
    
    if not os.path.exists(path_wavelet) or not os.path.exists(path_dct):
        print(f"Error: Could not find labels at {path_wavelet} or {path_dct}")
        return

    labels_wavelet = np.load(path_wavelet)
    labels_dct = np.load(path_dct)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Index': range(len(labels_wavelet)),
        'Cluster_Wavelet': labels_wavelet,
        'Cluster_DCT': labels_dct
    })
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved membership comparison to {output_csv}")
    
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
