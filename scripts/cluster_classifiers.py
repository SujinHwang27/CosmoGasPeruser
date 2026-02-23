import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def cluster_probes(input_file, output_dir, k=10):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading classifier parameters from {input_file}...")
    params = np.load(input_file)
    n_probes, d = params.shape
    
    # Step 1: Standardize
    print("Standardizing data...")
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params)
    
    # Step 2: Clustering
    print(f"Clustering into k={k} groups using KMeans...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(params_scaled)
    
    # Step 3: Dimensionality Reduction for Visualization
    print("Running UMAP for visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(params_scaled)
    
    # Step 4: Visualization
    print("Generating plots...")
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame({
        'UMAP 1': embedding[:, 0],
        'UMAP 2': embedding[:, 1],
        'Cluster': [f"Group {c}" for c in cluster_labels]
    })
    
    sns.scatterplot(data=df, x='UMAP 1', y='UMAP 2', hue='Cluster', palette='viridis', s=10, alpha=0.5)
    plt.title(f"Clustering of {n_probes} Micro-Classifiers (k={k})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    plot_file = os.path.join(output_dir, "classifier_clusters_umap.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    print(f"Saved cluster plot to {plot_file}")
    
    # Step 5: Save results
    labels_file = os.path.join(output_dir, "cluster_labels.npy")
    np.save(labels_file, cluster_labels)
    
    # Report cluster sizes
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("\nCluster Sizes:")
    print(counts)
    
    with open(os.path.join(output_dir, "cluster_summary.txt"), "w") as f:
        f.write(f"K-Means Clustering with k={k}\n")
        f.write(str(counts))
        
    print(f"Saved clustering results to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/feature_discovery/micro_classifier_params.npy", help="Path to params file")
    parser.add_argument("--output_dir", type=str, default="data/feature_discovery/clustering_results", help="Output directory")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    
    args = parser.parse_args()
    
    cluster_probes(args.input, args.output_dir, k=args.k)
