import numpy as np
import os
import argparse
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import pandas as pd

def visualize_3d(params_file, labels_file, output_html):
    print(f"Loading data from {params_file}...")
    params = np.load(params_file)
    labels = np.load(labels_file)
    
    # Standardize
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params)
    
    # Run UMAP 3D
    print("Running 3D UMAP (n_components=3)...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(params_scaled)
    
    # Prepare DataFrame
    df = pd.DataFrame({
        'UMAP 1': embedding[:, 0],
        'UMAP 2': embedding[:, 1],
        'UMAP 3': embedding[:, 2],
        'Cluster': [f"Group {l}" for l in labels]
    })
    
    # Create interactive plot
    print("Creating Plotly 3D scatter...")
    fig = px.scatter_3d(
        df, x='UMAP 1', y='UMAP 2', z='UMAP 3',
        color='Cluster',
        title=f"3D UMAP Visualization of Classifier Personalities",
        opacity=0.6,
        size_max=5
    )
    
    # Improve layout
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"Saved interactive 3D plot to {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="data/feature_discovery/base_data/params_wavelet.npy")
    parser.add_argument("--labels", type=str, default="data/feature_discovery/experiments/wavelet_k8_primary/cluster_labels.npy")
    parser.add_argument("--output", type=str, default="data/feature_discovery/experiments/wavelet_k8_primary/umap_3d_k8.html")
    
    args = parser.parse_args()
    
    visualize_3d(args.params, args.labels, args.output)
