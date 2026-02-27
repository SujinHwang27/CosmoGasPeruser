import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow

def verify_k(input_file, output_dir, max_k=20):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading classifier parameters from {input_file}...")
    params = np.load(input_file)
    
    # Standardize
    print("Standardizing data...")
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    mlflow.set_experiment("Sightline Clustering Optimization")
    
    print(f"Evaluating K-Means for k in {list(k_range)}...")
    for k in tqdm(k_range):
        with mlflow.start_run(run_name=f"k={k}", nested=True):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(params_scaled)
            inertias.append(kmeans.inertia_)
            
            # Silhouette score can be slow, using a subsample of 5000 for speed
            score = silhouette_score(params_scaled, kmeans.labels_, sample_size=5000, random_state=42)
            silhouette_scores.append(score)
            
            mlflow.log_param("k", k)
            mlflow.log_metric("inertia", kmeans.inertia_)
            mlflow.log_metric("silhouette_score", score)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Sum of Squares)', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, marker='s', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a vertical line at k=8
    plt.axvline(x=8, color='green', linestyle=':', label='Current k=8')

    plt.title('K-Means Optimization: Elbow Method & Silhouette Score')
    fig.tight_layout()
    
    plot_path = os.path.join(output_dir, "k_optimization_elbow_silhouette.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved optimization plot to {plot_path}")
    
    # Log final plot in a parent run
    with mlflow.start_run(run_name="Optimization Summary"):
        mlflow.log_artifact(plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/feature_discovery/base_data/params_wavelet.npy", help="Path to params file")
    parser.add_argument("--output_dir", type=str, default="data/feature_discovery/comparisons/k_stability", help="Output directory")
    parser.add_argument("--max_k", type=int, default=20, help="Maximum k to test")
    
    args = parser.parse_args()
    
    verify_k(args.input, args.output_dir, max_k=args.max_k)
