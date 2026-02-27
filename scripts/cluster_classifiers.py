import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import mlflow.sklearn

def cluster_probes(input_file, output_dir, k=10, random_seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    # Start MLflow run
    mlflow.set_experiment("Sightline Clustering")
    with mlflow.start_run(run_name=os.path.basename(output_dir) or "clustering_run"):
        # Log parameters
        mlflow.log_param("k", k)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("input_file", input_file)
        
        print(f"Loading classifier parameters from {input_file}...")
        params = np.load(input_file)
        n_probes, d = params.shape
        mlflow.log_param("n_samples", n_probes)
        mlflow.log_param("n_features", d)
        
        # Step 1: Standardize
        print("Standardizing data...")
        scaler = StandardScaler()
        params_scaled = scaler.fit_transform(params)
        
        # Step 2: Clustering
        print(f"Clustering into k={k} groups using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(params_scaled)
        
        # Log Metrics
        inertia = kmeans.inertia_
        mlflow.log_metric("inertia", inertia)
        
        # Calculate silhouette score on a sample if too large
        if n_probes > 20000:
            sample_indices = np.random.choice(n_probes, 10000, replace=False)
            sil = silhouette_score(params_scaled[sample_indices], cluster_labels[sample_indices])
        else:
            sil = silhouette_score(params_scaled, cluster_labels)
        mlflow.log_metric("silhouette_score", sil)
        
        # Step 3: Dimensionality Reduction for Visualization
        print("Running UMAP for visualization...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_seed)
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
        plt.close() # Free memory
        print(f"Saved cluster plot to {plot_file}")
        
        # Log Plot to MLflow
        mlflow.log_artifact(plot_file)
        
        # Step 5: Save results
        labels_file = os.path.join(output_dir, "cluster_labels.npy")
        np.save(labels_file, cluster_labels)
        mlflow.log_artifact(labels_file)
        
        # Report cluster sizes
        counts = pd.Series(cluster_labels).value_counts().sort_index()
        print("\nCluster Sizes:")
        print(counts)
        
        summary_file = os.path.join(output_dir, "cluster_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"K-Means Clustering with k={k}\n")
            f.write(str(counts))
        
        mlflow.log_artifact(summary_file)
        
        # Log the model itself
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
            
        print(f"Saved clustering results and logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/feature_discovery/base_data/params_l1_mechanism.npy", help="Path to params file")
    parser.add_argument("--output_dir", type=str, default="data/feature_discovery/experiments/new_run", help="Output directory")
    parser.add_argument("--k", type=int, default=8, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    cluster_probes(args.input, args.output_dir, k=args.k, random_seed=args.seed)
