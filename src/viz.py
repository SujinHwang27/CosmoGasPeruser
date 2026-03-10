"""
Visualization module for signal clustering analysis.

Includes:
- UMAP 2D and 3D plots
- Cluster mean absorption profiles
- Spatial index maps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import umap


def plot_umap_2d(fingerprints: np.ndarray, labels: np.ndarray, run_name: str, save_path: str):
    """
    Plot 2D UMAP embedding colored by cluster labels.
    """
    # Standardize
    scaler = StandardScaler()
    fingerprints_scaled = scaler.fit_transform(fingerprints)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(fingerprints_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'UMAP 2D: {run_name}')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return embedding


def plot_umap_3d_html(fingerprints: np.ndarray, labels: np.ndarray, run_name: str, save_path: str):
    """
    Create interactive 3D UMAP plot with Plotly.
    """
    # Standardize
    scaler = StandardScaler()
    fingerprints_scaled = scaler.fit_transform(fingerprints)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
    embedding = reducer.fit_transform(fingerprints_scaled)

    # Plotly
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'UMAP3': embedding[:, 2],
        'Cluster': labels.astype(str)
    })

    fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3', color='Cluster',
                        title=f'UMAP 3D: {run_name}')
    fig.update_layout(scene=dict(xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3'))
    fig.write_html(save_path)
    print(f"Saved: {save_path}")


def plot_cluster_profiles(X_abs_per_class: List[np.ndarray], labels: np.ndarray,
                         class_labels: np.ndarray, run_name: str, save_path: str):
    """
    Plot mean absorption profiles per cluster, with 4 classes overlaid.
    """
    n_clusters = len(np.unique(labels))
    n_classes = len(X_abs_per_class)

    # Determine layout
    n_cols = 4
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    class_names = ['NoFeedback', 'StellarWind', 'WindAGN', 'WindStrongAGN']

    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_size = len(cluster_indices)

        for c in range(n_classes):
            # Get absorption for this class at the sightlines belonging to this cluster
            X_class = X_abs_per_class[c]
            if len(cluster_indices) > 0:
                X_cluster_class = X_class[cluster_indices]
                profile = np.mean(X_cluster_class, axis=0)
                std = np.std(X_cluster_class, axis=0)

                ax.plot(profile, color=colors[c], label=class_names[c], alpha=0.8)
                ax.fill_between(np.arange(len(profile)), profile - std, profile + std,
                              color=colors[c], alpha=0.2)

        ax.set_title(f'Cluster {cluster_id} (n={cluster_size})')
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('A(λ)')
        if cluster_id == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Mean Absorption Profiles: {run_name}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_spatial_map(labels: np.ndarray, X_abs: np.ndarray, run_name: str, save_path: str):
    """
    Plot spatial index map: colored strip of cluster assignments + mean absorption.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 4), sharex=True)

    # Top: cluster assignments as colored strip
    cmap = plt.cm.tab10
    colors = cmap(labels / 10.0)
    axes[0].imshow([colors], aspect='auto', interpolation='nearest')
    axes[0].set_ylabel('Cluster')
    axes[0].set_yticks([])
    axes[0].set_xlim(0, 2048)

    # Bottom: mean absorption profile
    mean_abs = np.mean(X_abs, axis=0)
    axes[1].plot(mean_abs, color='black', linewidth=0.5)
    axes[1].fill_between(np.arange(len(mean_abs)), 0, mean_abs, alpha=0.3)
    axes[1].set_ylabel('A(λ)')
    axes[1].set_xlabel('Pixel')
    axes[1].set_xlim(0, 2048)
    axes[1].set_ylim(0, np.max(mean_abs) * 1.1)

    plt.suptitle(f'Spatial Index Map: {run_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_umap_comparison(output_dir: str, k: int, figs_dir: str):
    """
    Generate side-by-side UMAP comparison for wavelet and raw runs.
    """
    import os
    from sklearn.preprocessing import StandardScaler

    # Load data
    fingerprints_wavelet = np.load(os.path.join(output_dir, 'fingerprints_wavelet.npy'))
    fingerprints_raw = np.load(os.path.join(output_dir, 'fingerprints_raw.npy'))
    labels_wavelet = np.load(os.path.join(output_dir, f'labels_wavelet_k{k}.npy'))
    labels_raw = np.load(os.path.join(output_dir, f'labels_raw_k{k}.npy'))

    # Standardize
    scaler_w = StandardScaler()
    fp_w = scaler_w.fit_transform(fingerprints_wavelet)
    scaler_r = StandardScaler()
    fp_r = scaler_r.fit_transform(fingerprints_raw)

    # UMAP
    reducer_w = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reducer_r = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    emb_w = reducer_w.fit_transform(fp_w)
    emb_r = reducer_r.fit_transform(fp_r)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Row 1: K=8
    axes[0, 0].scatter(emb_w[:, 0], emb_w[:, 1], c=labels_wavelet, cmap='tab10', s=8, alpha=0.6)
    axes[0, 0].set_title(f'Wavelet K={k}')
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')

    axes[0, 1].scatter(emb_r[:, 0], emb_r[:, 1], c=labels_raw, cmap='tab10', s=8, alpha=0.6)
    axes[0, 1].set_title(f'Raw K={k}')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')

    # Also compute K=5 if available
    try:
        labels_w5 = np.load(os.path.join(output_dir, 'labels_wavelet_k5.npy'))
        labels_r5 = np.load(os.path.join(output_dir, 'labels_raw_k5.npy'))

        emb_w5 = reducer_w.fit_transform(fp_w)
        emb_r5 = reducer_r.fit_transform(fp_r)

        axes[1, 0].scatter(emb_w5[:, 0], emb_w5[:, 1], c=labels_w5, cmap='tab10', s=8, alpha=0.6)
        axes[1, 0].set_title('Wavelet K=5')
        axes[1, 0].set_xlabel('UMAP 1')
        axes[1, 0].set_ylabel('UMAP 2')

        axes[1, 1].scatter(emb_r5[:, 0], emb_r5[:, 1], c=labels_r5, cmap='tab10', s=8, alpha=0.6)
        axes[1, 1].set_title('Raw K=5')
        axes[1, 1].set_xlabel('UMAP 1')
        axes[1, 1].set_ylabel('UMAP 2')
    except:
        # Remove bottom row if K=5 not available
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    plt.suptitle('UMAP Comparison: Wavelet vs Raw', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(figs_dir, 'fig_umap2d_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_spatial_map_comparison(output_dir: str, k: int, figs_dir: str):
    """
    Generate stacked spatial map comparison for wavelet and raw runs.
    """
    import os

    # Load labels
    labels_wavelet = np.load(os.path.join(output_dir, f'labels_wavelet_k{k}.npy'))
    labels_raw = np.load(os.path.join(output_dir, f'labels_raw_k{k}.npy'))

    # Load absorption
    from src.core.data import SignalClusteringData
    data = SignalClusteringData()
    X_abs_per_class, _ = data.load_flux_per_class()
    X_abs = np.vstack([np.clip(1.0 - flux, 0.0, 1.0) for flux in X_abs_per_class])

    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    # Top two: colored strips
    cmap = plt.cm.tab10

    colors_w = cmap(labels_wavelet / 10.0)
    axes[0].imshow([colors_w], aspect='auto', interpolation='nearest')
    axes[0].set_ylabel('Wavelet\nCluster')
    axes[0].set_yticks([])

    colors_r = cmap(labels_raw / 10.0)
    axes[1].imshow([colors_r], aspect='auto', interpolation='nearest')
    axes[1].set_ylabel('Raw\nCluster')
    axes[1].set_yticks([])

    # Compute mean absorption
    mean_abs = np.mean(X_abs, axis=0)

    # Bottom two: mean absorption profiles
    for ax, labels, name in [(axes[2], labels_wavelet, 'Wavelet'), (axes[3], labels_raw, 'Raw')]:
        ax.plot(mean_abs, color='black', linewidth=0.5)
        ax.fill_between(np.arange(len(mean_abs)), 0, mean_abs, alpha=0.3)
        ax.set_ylabel('A(λ)')
        ax.set_xlim(0, 2048)

    axes[3].set_xlabel('Pixel')

    plt.suptitle(f'Spatial Index Map Comparison: K={k}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(figs_dir, 'fig_spatial_map_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def find_cluster_regions(indices):
    """
    Find continuous regions from a list of indices.
    Returns list of (start_idx, end_idx) tuples.
    """
    if len(indices) == 0:
        return []
    indices = np.sort(np.asarray(indices))
    diff = np.diff(indices)
    breaks = np.where(diff > 1)[0]

    # Build regions
    regions = []
    prev = 0
    for b in breaks:
        regions.append((indices[prev], indices[b]))
        prev = b + 1
    # Last region
    regions.append((indices[prev], indices[-1]))

    return regions
