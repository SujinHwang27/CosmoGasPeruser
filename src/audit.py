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


def compute_contingency(labels1: np.ndarray, labels2: np.ndarray, n_clusters1: int = 5, n_clusters2: int = 5) -> np.ndarray:
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


def cross_run_overlap(labels_wavelet: np.ndarray, labels_raw: np.ndarray, k: int = 5) -> pd.DataFrame:
    """
    Compute cross-run overlap analysis.

    Args:
        labels_wavelet: K=5 cluster labels from wavelet run
        labels_raw: K=5 cluster labels from raw run
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


def plot_contingency_heatmap(contingency: np.ndarray, save_path: str, k: int = 5):
    """
    Plot the 5×5 contingency matrix as an annotated heatmap.
    Rows = Wavelet clusters, Columns = Raw clusters.
    Each cell shows absolute count and row-percentage.
    """
    import seaborn as sns

    # Compute row-wise percentages
    row_sums = contingency.sum(axis=1, keepdims=True)
    pct_matrix = 100.0 * contingency / row_sums

    # Build annotation strings: count\n(pct%)
    annot = np.empty_like(contingency, dtype=object)
    for i in range(k):
        for j in range(k):
            annot[i, j] = f"{contingency[i, j]:,}\n({pct_matrix[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(contingency, annot=annot, fmt='', cmap='YlOrRd',
                xticklabels=[f'Raw {j}' for j in range(k)],
                yticklabels=[f'Wavelet {i}' for i in range(k)],
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title('Cross-Run Contingency Matrix (K=5)', fontsize=14, pad=15)
    ax.set_xlabel('Raw Cluster', fontsize=12)
    ax.set_ylabel('Wavelet Cluster', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shared_umap_overlay_3d(fp_wavelet: np.ndarray, fp_raw: np.ndarray,
                                 labels_wavelet: np.ndarray, labels_raw: np.ndarray,
                                 overlap_df: pd.DataFrame,
                                 save_path: str, k: int = 5):
    """
    Create a 3D interactive plot where wavelet and raw fingerprints are
    projected into the same 3D UMAP space via a single joint UMAP fit.

    Both fingerprint arrays (each 16384 × 24) are concatenated into a
    (32768 × 24) matrix, standardized together, and projected through one
    3D UMAP. The resulting embedding is split back: wavelet half and raw half
    live in identical coordinates. Matched cluster pairs share the same color.
    Both layers use opacity=0.5 so overlapping sightlines appear denser.
    """
    import umap
    import plotly.graph_objects as go

    n = fp_wavelet.shape[0]  # 16384

    # ── Step 1: Concatenate and fit a single 3D UMAP ──
    combined = np.vstack([fp_wavelet, fp_raw])  # (32768, 24)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
    embedding = reducer.fit_transform(combined_scaled)

    emb_w = embedding[:n]    # wavelet half
    emb_r = embedding[n:]    # raw half

    # ── Step 2: Build matched cluster pair mapping ──
    match_map = {}
    for _, row in overlap_df.iterrows():
        match_map[int(row['wavelet_cluster'])] = int(row['best_raw_match'])

    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Map raw cluster labels → wavelet cluster color index
    raw_to_wavelet_color = {}
    for w_clust, r_clust in match_map.items():
        raw_to_wavelet_color[r_clust] = w_clust

    # ── Step 3: Plot in 3D with Plotly ──
    fig = go.Figure()

    # Wavelet points
    for w_clust in range(k):
        mask = (labels_wavelet == w_clust)
        fig.add_trace(go.Scatter3d(
            x=emb_w[mask, 0], y=emb_w[mask, 1], z=emb_w[mask, 2],
            mode='markers',
            marker=dict(size=2, color=cluster_colors[w_clust], opacity=0.5, symbol='circle'),
            name=f'Wavelet C{w_clust}',
            legendgroup=f'cluster_{w_clust}',
        ))

    # Raw points
    for r_clust in np.unique(labels_raw):
        mask = (labels_raw == r_clust)
        color_idx = raw_to_wavelet_color.get(int(r_clust), int(r_clust))
        fig.add_trace(go.Scatter3d(
            x=emb_r[mask, 0], y=emb_r[mask, 1], z=emb_r[mask, 2],
            mode='markers',
            marker=dict(size=2, color=cluster_colors[color_idx], opacity=0.5, symbol='diamond'),
            name=f'Raw C{r_clust} (→W{color_idx})',
            legendgroup=f'cluster_{color_idx}',
        ))

    fig.update_layout(
        title='Shared 3D UMAP: Wavelet vs Raw Fingerprints (K=5)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
        ),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(save_path)
    print(f"Saved: {save_path}")


def plot_separability_greyscale(fp: np.ndarray, labels: np.ndarray,
                                run_name: str, save_path: str, k: int = 5):
    """
    Generate a greyscale image of separability vectors (fingerprints) grouped
    by cluster assignment.

    Each row is a sightline (sorted by cluster), each column is one of the 24
    fingerprint dimensions. Pixel intensity = normalised separability value.
    Cluster boundaries are drawn as horizontal red lines.
    """
    # Sort sightlines by cluster
    sort_idx = np.argsort(labels)
    sorted_labels = labels[sort_idx]
    sorted_fp = fp[sort_idx]

    # Normalize to [0, 1] for greyscale display
    vmin, vmax = sorted_fp.min(), sorted_fp.max()
    if vmax - vmin > 0:
        normed = (sorted_fp - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(sorted_fp)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(normed, aspect='auto', cmap='gray', interpolation='nearest',
              vmin=0, vmax=1)

    # Draw cluster boundaries
    boundaries = []
    for c in range(k):
        cluster_end = np.searchsorted(sorted_labels, c, side='right')
        if cluster_end < len(sorted_labels):
            boundaries.append(cluster_end)
            ax.axhline(y=cluster_end - 0.5, color='red', linewidth=1.0, alpha=0.8)

    # Label clusters on the y-axis
    cluster_centers = []
    prev = 0
    for b in boundaries:
        cluster_centers.append((prev + b) / 2)
        prev = b
    cluster_centers.append((prev + len(sorted_labels)) / 2)
    ax.set_yticks(cluster_centers[:k])
    ax.set_yticklabels([f'Cluster {i}' for i in range(k)])

    ax.set_xlabel('Fingerprint Dimension (0–23)', fontsize=12)
    ax.set_ylabel('Sightlines (sorted by cluster)', fontsize=12)
    ax.set_title(f'Separability Vectors — {run_name} (K={k})', fontsize=14, pad=15)

    cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Normalized Intensity')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def wavelet_attribution(output_dir: str, k: int = 5) -> pd.DataFrame:
    """
    Compute wavelet level attribution per cluster.

    Loads wavelet features and computes mean absolute activation per level.
    Uses all wavelet levels: D1-D6 + A6.
    """
    # Load wavelet data (need to know level boundaries)
    # The wavelet data is D1-D6 + A6 concatenated, 2048 features total

    # Load labels
    labels = np.load(f'{output_dir}/labels_wavelet_k{k}.npy')

    # Load pre-normalized wavelet per class
    wavelet_per_class = []
    for c in range(1, 5):
        wavelet_per_class.append(np.load(f'{output_dir}/../feature_discovery/wavelet_class{c}.npy'))

    # Level boundaries for 2048 features:
    # D1: 0-1024, D2: 1024-1536, D3: 1536-1792, D4: 1792-1920, D5: 1920-1984, D6: 1984-2016, A6: 2016-2048
    level_sizes = [1024, 512, 256, 128, 64, 32, 32]
    level_names = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'A6']

    # Use all levels (D1-D6 + A6)
    level_indices = [0, 1, 2, 3, 4, 5, 6]
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


def raw_attribution(output_dir: str, k: int = 5, n_bins: int = 64) -> pd.DataFrame:
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


def variance_audit(labels_wavelet: np.ndarray, labels_raw: np.ndarray, k: int = 5) -> pd.DataFrame:
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

    # Use distinct colors for Wavelet vs Raw
    run_colors = {'wavelet': 'steelblue', 'raw': 'coral'}

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(k)
    width = 0.35

    for i, run in enumerate(runs):
        run_data = variance_df[variance_df['run'] == run]
        offsets = width * (i - 0.5 * (len(runs) - 1))
        values = run_data['normalized_variance'].values
        # Use same color for all bars within each run
        color = run_colors.get(run, 'gray')
        ax.bar(x + offsets, values, width, label=run.capitalize(), color=color, alpha=0.7)

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
