"""
Auditing module for signal clustering analysis.

Provides functions for:
- Cross-run cluster overlap analysis
- Post-hoc feature attribution
- Inter-class variance audit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def compute_contingency(labels1: np.ndarray, labels2: np.ndarray, n_clusters1: int = 8, n_clusters2: int = 8) -> np.ndarray:
    """
    Compute contingency matrix between two label arrays.

    Args:
        labels1: Cluster labels from first run
        labels2: Cluster labels from second run
        n_clusters1: Number of clusters in first run
        n_clusters2: Number of clusters in second run

    Returns:
        contingency matrix of shape (n_clusters1, n_clusters2)
    """
    contingency = np.zeros((n_clusters1, n_clusters2), dtype=int)
    for i in range(n_clusters1):
        for j in range(n_clusters2):
            contingency[i, j] = np.sum((labels1 == i) & (labels2 == j))
    return contingency


def cross_run_overlap(labels_wavelet: np.ndarray, labels_raw: np.ndarray, k: int = 8) -> pd.DataFrame:
    """
    Compute cross-run overlap analysis.

    Args:
        labels_wavelet: K=8 cluster labels from wavelet run
        labels_raw: K=8 cluster labels from raw run
        k: Number of clusters

    Returns:
        DataFrame with overlap summary per cluster
    """
    contingency = compute_contingency(labels_wavelet, labels_raw, k, k)

    # Find best matches using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-contingency)

    results = []
    for i in range(k):
        w_size = np.sum(labels_wavelet == i)
        best_raw = col_ind[i]
        overlap = contingency[i, best_raw]
        overlap_pct = 100 * overlap / w_size if w_size > 0 else 0

        # Determine regime based on cluster size
        total = len(labels_wavelet)
        pct = 100 * w_size / total
        if pct > 15:
            regime = 'bulk'
        elif pct > 5:
            regime = 'transition'
        else:
            regime = 'signal_island'

        results.append({
            'wavelet_cluster': i,
            'wavelet_size': w_size,
            'wavelet_pct': pct,
            'best_raw_match': best_raw,
            'overlap_count': overlap,
            'overlap_pct': overlap_pct,
            'regime': regime
        })

    return pd.DataFrame(results), contingency


def plot_cross_run_overlap(overlap_df: pd.DataFrame, contingency: np.ndarray,
                           save_path: str, k: int = 8):
    """
    Plot cross-run overlap as grouped bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(k)
    width = 0.35

    # Get wavelet size and overlap
    wavelet_sizes = overlap_df['wavelet_size'].values
    overlaps = overlap_df['overlap_count'].values

    # Normalize overlaps to percentages
    overlap_pcts = 100 * overlaps / wavelet_sizes

    bars1 = ax.bar(x - width/2, wavelet_sizes, width, label='Wavelet Cluster Size', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, overlap_pcts * wavelet_sizes / 100, width,
                   label='Overlap with Best Raw', color='coral', alpha=0.7)

    # Color signal islands differently
    regimes = overlap_df['regime'].values
    for i, regime in enumerate(regimes):
        if regime == 'signal_island':
            bars1[i].set_color('darkblue')
            bars2[i].set_color('darkred')

    ax.set_xlabel('Wavelet Cluster')
    ax.set_ylabel('Count / Percentage')
    ax.set_title('Cross-Run Cluster Overlap: Wavelet K=8 vs Raw K=8')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(k)])
    ax.legend()

    # Add percentage labels on overlap bars
    ax2 = ax.twinx()
    ax2.plot(x, overlap_pcts, 'go-', linewidth=2, markersize=6, label='Overlap %')
    ax2.set_ylabel('Overlap Percentage (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def wavelet_attribution(output_dir: str, k: int = 8) -> pd.DataFrame:
    """
    Compute wavelet level attribution per cluster.

    Loads wavelet features and computes mean absolute activation per level.
    Note: D1 and D2 are excluded due to sparsity issues.
    """
    # Load wavelet data (need to know level boundaries)
    # The wavelet data is D1-D6 + A6 concatenated
    # Assuming 2048 features: D1(1) + D2(2) + D3(4) + D4(8) + D5(16) + D6(32) + A6(64) = 127

    # Load labels
    labels = np.load(f'{output_dir}/labels_wavelet_k{k}.npy')

    # Load pre-normalized wavelet per class
    wavelet_per_class = []
    for c in range(1, 5):
        wavelet_per_class.append(np.load(f'{output_dir}/../feature_discovery/wavelet_class{c}.npy'))

    # Use levels D3-D6+A6 (skip D1, D2 which are sparse)
    # Approximate level boundaries for db8 with 6 decomposition levels:
    # Level boundaries (approximate): D1=1, D2=2, D3=4, D4=8, D5=16, D6=32, A6=64
    level_sizes = [1, 2, 4, 8, 16, 32, 64]
    level_names = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'A6']

    # Skip D1, D2 - use D3-D6, A6 (indices 2-6)
    level_indices = [2, 3, 4, 5, 6]  # D3, D4, D5, D6, A6
    selected_levels = [level_names[i] for i in level_indices]
    selected_sizes = [level_sizes[i] for i in level_indices]

    # Compute boundaries
    boundaries = np.cumsum([0] + level_sizes)

    results = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_size = np.sum(mask)

        # Get all wavelet data for this cluster (any class)
        # Average across classes
        for c in range(4):
            X_w = wavelet_per_class[c][mask]
            if c == 0:
                X_cluster = X_w
            else:
                X_cluster = np.vstack([X_cluster, X_w])

        # Compute mean absolute activation per level
        level_means = []
        for li in level_indices:
            start, end = boundaries[li], boundaries[li+1]
            level_data = X_cluster[:, start:end]
            level_means.append(np.mean(np.abs(level_data)))

        results.append({
            'cluster': cluster_id,
            'size': cluster_size,
            **{l: m for l, m in zip(selected_levels, level_means)}
        })

    return pd.DataFrame(results)


def plot_wavelet_attribution(attribution_df: pd.DataFrame, save_path: str):
    """
    Plot wavelet attribution heatmap.
    """
    level_cols = [c for c in attribution_df.columns if c.startswith('D') or c == 'A6']

    fig, ax = plt.subplots(figsize=(10, 6))

    data = attribution_df[level_cols].values
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(level_cols)))
    ax.set_xticklabels(level_cols)
    ax.set_yticks(np.arange(len(attribution_df)))
    ax.set_yticklabels([f'C{i}' for i in attribution_df['cluster']])

    # Add values
    for i in range(len(attribution_df)):
        for j in range(len(level_cols)):
            text = ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=8)

    ax.set_title('Wavelet Attribution: Cluster × Level')
    ax.set_xlabel('Wavelet Level')
    ax.set_ylabel('Cluster')

    plt.colorbar(im, ax=ax, label='Mean |Activation|')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def raw_attribution(output_dir: str, k: int = 8, n_bins: int = 64) -> pd.DataFrame:
    """
    Compute raw spectral attribution per cluster.

    Groups 2048 pixels into bins and computes mean absolute deviation from grand mean.
    """
    from src.core.data import SignalClusteringData

    # Load labels
    labels = np.load(f'{output_dir}/labels_raw_k{k}.npy')

    # Load absorption data (per-class)
    data = SignalClusteringData()
    X_abs_per_class, _ = data.load_flux_per_class()
    X_abs_per_class = [np.clip(1.0 - flux, 0.0, 1.0) for flux in X_abs_per_class]

    # Compute grand mean across all classes
    X_abs = np.vstack(X_abs_per_class)
    grand_mean = np.mean(X_abs, axis=0)

    # Compute bin size
    bin_size = 2048 // n_bins

    results = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_size = np.sum(mask)

        # Get cluster absorption from all classes (averaging across classes)
        X_cluster_list = []
        for c in range(4):
            X_class = X_abs_per_class[c][mask]
            X_cluster_list.append(X_class)
        X_cluster = np.vstack(X_cluster_list)

        # Mean absolute deviation from grand mean per bin
        deviations = np.abs(X_cluster - grand_mean)
        bin_means = []
        for b in range(n_bins):
            start, end = b * bin_size, (b + 1) * bin_size
            bin_means.append(np.mean(deviations[:, start:end]))

        results.append({
            'cluster': cluster_id,
            'size': cluster_size,
            **{f'bin_{b}': m for b, m in enumerate(bin_means)}
        })

    return pd.DataFrame(results)


def plot_raw_attribution(attribution_df: pd.DataFrame, save_path: str, n_bins: int = 64):
    """
    Plot raw attribution heatmap.
    """
    bin_cols = [c for c in attribution_df.columns if c.startswith('bin_')]

    fig, ax = plt.subplots(figsize=(14, 5))

    data = attribution_df[bin_cols].values
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(0, n_bins, 4))
    ax.set_xticklabels(np.arange(0, n_bins, 4) * (2048 // n_bins))
    ax.set_yticks(np.arange(len(attribution_df)))
    ax.set_yticklabels([f'C{i}' for i in attribution_df['cluster']])

    ax.set_title('Raw Spectral Attribution: Cluster × Pixel Bin')
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Cluster')

    plt.colorbar(im, ax=ax, label='Mean |Deviation|')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def variance_audit(labels_wavelet: np.ndarray, labels_raw: np.ndarray, k: int = 8) -> pd.DataFrame:
    """
    Compute inter-class variance audit.

    For each cluster, computes variance of 4 class mean absorption profiles across pixels.
    Normalizes by largest bulk cluster.
    """
    from src.core.data import SignalClusteringData

    # Load absorption data
    data = SignalClusteringData()
    X_abs_per_class, _ = data.load_flux_per_class()
    X_abs_per_class = [np.clip(1.0 - flux, 0.0, 1.0) for flux in X_abs_per_class]

    results = []

    for run_name, labels in [('wavelet', labels_wavelet), ('raw', labels_raw)]:
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_size = np.sum(mask)

            # Compute mean profile per class
            class_means = []
            for c in range(4):
                X_class_c = X_abs_per_class[c][mask]
                class_means.append(np.mean(X_class_c, axis=0))

            # Stack and compute variance across classes per pixel
            class_means = np.array(class_means)  # (4, 2048)
            inter_class_var = np.var(class_means, axis=0)  # (2048,)
            mean_var = np.mean(inter_class_var)

            # Determine regime
            total = len(labels)
            pct = 100 * cluster_size / total
            if pct > 15:
                regime = 'bulk'
            elif pct > 5:
                regime = 'transition'
            else:
                regime = 'signal_island'

            results.append({
                'run': run_name,
                'cluster': cluster_id,
                'size': cluster_size,
                'pct': pct,
                'inter_class_variance': mean_var,
                'regime': regime
            })

    df = pd.DataFrame(results)

    # Normalize by largest bulk cluster
    bulk_variances = df[df['regime'] == 'bulk']['inter_class_variance']
    baseline = bulk_variances.max() if len(bulk_variances) > 0 else 1.0

    if baseline > 0:
        df['normalized_variance'] = df['inter_class_variance'] / baseline
    else:
        df['normalized_variance'] = 1.0

    return df


def plot_variance_ratios(variance_df: pd.DataFrame, save_path: str):
    """
    Plot variance ratios as grouped bar chart.
    """
    runs = variance_df['run'].unique()
    k = len(variance_df) // len(runs)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(k)
    width = 0.35

    for i, run in enumerate(runs):
        run_data = variance_df[variance_df['run'] == run]
        offsets = width * (i - 0.5 * (len(runs) - 1))
        values = run_data['normalized_variance'].values
        colors = ['steelblue' if r == 'bulk' else ('coral' if r == 'transition' else 'darkred')
                  for r in run_data['regime']]
        ax.bar(x + offsets, values, width, label=run.capitalize(), color=colors, alpha=0.7)

    ax.axhline(y=1.0, color='gray', linestyle='--', label='Bulk baseline')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Normalized Inter-Class Variance')
    ax.set_title('Inter-Class Variance Audit')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(k)])
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
