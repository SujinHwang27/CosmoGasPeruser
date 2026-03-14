"""
Stage 3: Clustering for Signal Clustering Analysis

Runs K-Means clustering on separability vectors with K-sweep,
elbow detection, and silhouette analysis.

Outputs:
    - data/feature_discovery/labels_wavelet_k*.npy: Cluster labels
    - data/feature_discovery/centroids_wavelet_k*.npy: Cluster centroids
    - results/cluster_stats_*.csv: Per-cluster statistics
    - figs/fig_elbow_*.png: Elbow plots
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.cluster import k_sweep, fit_kmeans, compute_cluster_stats, contingency_matrix


def plot_elbow(sweep_df, run_name, k_chosen, save_path):
    """
    Plot inertia and silhouette vs K.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Inertia on left axis
    color = 'tab:blue'
    ax1.set_xlabel('K', fontsize=12)
    ax1.set_ylabel('Inertia', color=color, fontsize=12)
    ax1.plot(sweep_df['k'], sweep_df['inertia'], 'o-', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)

    # Silhouette on right axis
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
    ax2.plot(sweep_df['k'], sweep_df['silhouette'], 's-', color=color, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    # Mark chosen K
    ax1.axvline(k_chosen, color='red', linestyle='--', alpha=0.7, label=f'Chosen K={k_chosen}')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Elbow Analysis: {run_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Clustering')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which separability vector set to cluster')
    parser.add_argument('--k', type=int, default=5,
                       help='Target K for final clustering')
    parser.add_argument('--k_sweep_range', type=str, default='2,21',
                       help='K range for sweep (start,end exclusive)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Output directory for labels')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Output directory for stats')
    parser.add_argument('--figs_dir', type=str, default='figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Parse k range
    k_start, k_end = map(int, args.k_sweep_range.split(','))
    k_range = range(k_start, k_end)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 3: Clustering")
    print("=" * 60)

    separability_vectors_wavelet = None
    separability_vectors_raw = None

    # Load separability vectors if needed
    if args.run in ['wavelet', 'both']:
        sv_path = os.path.join(args.output_dir, 'separability_vectors_wavelet.npy')
        separability_vectors_wavelet = np.load(sv_path)
        print(f"Loaded wavelet separability vectors: {separability_vectors_wavelet.shape}")

    if args.run in ['raw', 'both']:
        sv_path = os.path.join(args.output_dir, 'separability_vectors_raw.npy')
        separability_vectors_raw = np.load(sv_path)
        print(f"Loaded raw separability vectors: {separability_vectors_raw.shape}")

    def process_separability_vectors(separability_vectors, run_name):
        """Process a single separability vector set."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        separability_vectors_scaled = scaler.fit_transform(separability_vectors)

        print(f"\n{'='*40}")
        print(f"Running: {run_name}")
        print(f"{'='*40}")

        # K-sweep
        print(f"\nK-sweep over range {list(k_range)}...")
        sweep_df = k_sweep(separability_vectors_scaled, k_range, seed=args.seed)
        sweep_df.to_csv(os.path.join(args.results_dir, f'sweep_{run_name}.csv'), index=False)

        # Plot elbow
        plot_elbow(sweep_df, run_name, args.k,
                  os.path.join(args.figs_dir, f'fig_elbow_{run_name}.png'))

        # Fit final K
        print(f"\nFitting final K={args.k}...")
        labels, centroids = fit_kmeans(separability_vectors, args.k, seed=args.seed)

        # Save labels and centroids
        labels_path = os.path.join(args.output_dir, f'labels_{run_name}_k{args.k}.npy')
        centroids_path = os.path.join(args.output_dir, f'centroids_{run_name}_k{args.k}.npy')
        np.save(labels_path, labels.astype(np.int32))
        np.save(centroids_path, centroids)
        print(f"Saved: {labels_path}")
        print(f"Saved: {centroids_path}")

        # Compute stats
        stats_df = compute_cluster_stats(separability_vectors, labels, centroids)
        stats_path = os.path.join(args.results_dir, f'cluster_stats_{run_name}_k{args.k}.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved: {stats_path}")

        print(f"\nCluster sizes ({run_name}):")
        for _, row in stats_df.iterrows():
            print(f"  Cluster {row['cluster']}: {row['size']} ({row['pct_total']:.1f}%)")

        # K=8 stability check (run if primary is not K=5)
        if args.k != 5:
            print(f"\nRunning K=5 stability check...")
            labels_k5, centroids_k5 = fit_kmeans(separability_vectors, 5, seed=args.seed)

            # Save K=5
            labels_k5_path = os.path.join(args.output_dir, f'labels_{run_name}_k5.npy')
            np.save(labels_k5_path, labels_k5.astype(np.int32))
            print(f"Saved: {labels_k5_path}")

            # Contingency
            cont = contingency_matrix(labels_k5, labels, 5, args.k)
            cont_df = pd.DataFrame(cont, index=[f'K5_{i}' for i in range(5)],
                                    columns=[f'K{args.k}_{i}' for i in range(args.k)])
            cont_path = os.path.join(args.results_dir, f'contingency_k5_vs_k{args.k}_{run_name}.csv')
            cont_df.to_csv(cont_path)
            print(f"Saved: {cont_path}")

        return labels, centroids

    # Process
    results = {}
    if args.run in ['wavelet', 'both']:
        results['wavelet'] = process_separability_vectors(separability_vectors_wavelet, 'wavelet')

    if args.run in ['raw', 'both']:
        results['raw'] = process_separability_vectors(separability_vectors_raw, 'raw')

    print("\n" + "=" * 60)
    print("Stage 3 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
