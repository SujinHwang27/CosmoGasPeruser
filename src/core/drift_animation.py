"""
Cluster membership drift animation.

Visualizes how sightlines move between raw-run and wavelet-run
cluster assignments using an animated 2D UMAP transition.

Phase 1: All 16384 sightlines at raw UMAP positions, colored by raw cluster.
Phase 2: Points animate to wavelet UMAP positions; drifting sightlines
         change color to wavelet cluster color, stable sightlines stay fixed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
from sklearn.preprocessing import StandardScaler
import umap


# ── Fixed 5-color palette (matched clusters share the same color) ──
CLUSTER_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
]


def compute_shared_umap_2d(fp_wavelet: np.ndarray, fp_raw: np.ndarray,
                           n_neighbors: int = 15, min_dist: float = 0.1,
                           random_state: int = 42):
    """
    Fit a single 2D UMAP on the concatenation of wavelet and raw fingerprints
    so both representations share the same coordinate space.

    Returns:
        emb_raw: (N, 2)  — raw half positions
        emb_wav: (N, 2)  — wavelet half positions
    """
    n = fp_wavelet.shape[0]
    combined = np.vstack([fp_raw, fp_wavelet])  # raw first, wavelet second
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=random_state
    )
    embedding = reducer.fit_transform(combined_scaled)

    emb_raw = embedding[:n]
    emb_wav = embedding[n:]
    return emb_raw, emb_wav


def build_color_arrays(labels_wavelet: np.ndarray, labels_raw: np.ndarray,
                       overlap_df: pd.DataFrame, k: int = 5):
    """
    Build per-sightline start/end color arrays using fixed 5-color palette.

    Matched clusters share the same color index (= wavelet cluster index).
    - Stable sightlines: start color == end color (no change)
    - Drifting sightlines: start color (raw cluster's matched color) ≠ end color

    Returns:
        colors_start: (N, 4) RGBA — initial colors (raw cluster assignment)
        colors_end:   (N, 4) RGBA — final colors (wavelet cluster assignment)
        is_stable:    (N,) bool  — True if sightline has no membership drift
    """
    n = len(labels_wavelet)

    # Build mapping: raw_cluster → wavelet_cluster (i.e., which color to use)
    raw_to_color_idx = {}
    for _, row in overlap_df.iterrows():
        w_clust = int(row['wavelet_cluster'])
        r_clust = int(row['best_raw_match'])
        raw_to_color_idx[r_clust] = w_clust  # raw cluster r_clust gets color of wavelet cluster w_clust

    # Per-sightline colors
    colors_start = np.zeros((n, 4))  # RGBA
    colors_end = np.zeros((n, 4))
    is_stable = np.zeros(n, dtype=bool)

    for i in range(n):
        w_clust = labels_wavelet[i]
        r_clust = labels_raw[i]

        # End color = wavelet cluster color (direct mapping)
        colors_end[i] = to_rgba(CLUSTER_COLORS[w_clust])

        # Start color = raw cluster's matched color
        color_idx = raw_to_color_idx.get(int(r_clust), int(r_clust))
        colors_start[i] = to_rgba(CLUSTER_COLORS[color_idx])

        # Stable = raw cluster is the best_raw_match for this sightline's wavelet cluster
        expected_raw = int(overlap_df.loc[
            overlap_df['wavelet_cluster'] == w_clust, 'best_raw_match'
        ].values[0])
        is_stable[i] = (r_clust == expected_raw)

    return colors_start, colors_end, is_stable


def render_drift_animation(emb_raw: np.ndarray, emb_wav: np.ndarray,
                           colors_start: np.ndarray, colors_end: np.ndarray,
                           is_stable: np.ndarray,
                           save_path: str,
                           n_frames: int = 90, fps: int = 30,
                           hold_start: int = 15, hold_end: int = 15,
                           marker_size: float = 0.5, dpi: int = 150):
    """
    Render the drift animation as a GIF.

    Timeline:
        [0, hold_start)             — static at raw positions
        [hold_start, hold_start+n_frames)  — animate raw → wavelet
        [hold_start+n_frames, ...)  — static at wavelet positions

    Stable sightlines: fixed color throughout, only position changes.
    Drifting sightlines: color transitions from raw cluster color → wavelet cluster color.
    """
    total_frames = hold_start + n_frames + hold_end

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Compute axis limits from both embeddings
    all_pts = np.vstack([emb_raw, emb_wav])
    margin = 0.05
    x_range = all_pts[:, 0].max() - all_pts[:, 0].min()
    y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
    ax.set_xlim(all_pts[:, 0].min() - margin * x_range,
                all_pts[:, 0].max() + margin * x_range)
    ax.set_ylim(all_pts[:, 1].min() - margin * y_range,
                all_pts[:, 1].max() + margin * y_range)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw stable points first (background), then drifters (foreground)
    stable_mask = is_stable
    drift_mask = ~is_stable

    # Scatter objects
    scat_stable = ax.scatter(
        emb_raw[stable_mask, 0], emb_raw[stable_mask, 1],
        c=colors_start[stable_mask], s=marker_size, alpha=0.6,
        edgecolors='none', rasterized=True
    )
    scat_drift = ax.scatter(
        emb_raw[drift_mask, 0], emb_raw[drift_mask, 1],
        c=colors_start[drift_mask], s=marker_size * 1.5, alpha=0.8,
        edgecolors='none', rasterized=True
    )

    n_stable = stable_mask.sum()
    n_drift = drift_mask.sum()
    title = ax.set_title(
        f'Cluster Membership Drift: Raw → Wavelet\n'
        f'Stable: {n_stable:,} ({100*n_stable/len(is_stable):.1f}%)  '
        f'Drifted: {n_drift:,} ({100*n_drift/len(is_stable):.1f}%)',
        color='white', fontsize=12, pad=10
    )

    phase_text = ax.text(
        0.02, 0.02, '', transform=ax.transAxes,
        color='white', fontsize=10, alpha=0.8,
        verticalalignment='bottom'
    )

    def update(frame):
        if frame < hold_start:
            t = 0.0
            phase_text.set_text('Raw cluster positions')
        elif frame < hold_start + n_frames:
            t = (frame - hold_start) / (n_frames - 1)
            pct = int(t * 100)
            phase_text.set_text(f'Transitioning... {pct}%')
        else:
            t = 1.0
            phase_text.set_text('Wavelet cluster positions')

        # Interpolate positions
        pos_stable = (1 - t) * emb_raw[stable_mask] + t * emb_wav[stable_mask]
        pos_drift = (1 - t) * emb_raw[drift_mask] + t * emb_wav[drift_mask]

        scat_stable.set_offsets(pos_stable)
        scat_drift.set_offsets(pos_drift)

        # Stable: color never changes
        # Drifters: interpolate color
        if t > 0:
            interp_colors = (1 - t) * colors_start[drift_mask] + t * colors_end[drift_mask]
            scat_drift.set_facecolors(interp_colors)

        return scat_stable, scat_drift, phase_text

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 // fps, blit=False
    )

    # Save as MP4 using ffmpeg writer
    mp4_path = save_path.replace('.gif', '.mp4') if save_path.endswith('.gif') else save_path
    print(f"Rendering {total_frames} frames at {fps} fps → {mp4_path}")
    writer_mp4 = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(mp4_path, writer=writer_mp4, dpi=dpi)
    print(f"Saved: {mp4_path}")

    # Also save as GIF using Pillow writer
    gif_path = mp4_path.replace('.mp4', '.gif')
    print(f"Rendering GIF → {gif_path}")
    writer_gif = animation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer_gif, dpi=dpi)
    print(f"Saved: {gif_path}")
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_drift_animation(data_dir: str, results_dir: str, figs_dir: str,
                           k: int = 5):
    """
    End-to-end: load data, compute UMAP, build colors, render animation.
    """
    import os

    # Load fingerprints and labels
    fp_wavelet = np.load(os.path.join(data_dir, 'fingerprints_wavelet.npy'))
    fp_raw = np.load(os.path.join(data_dir, 'fingerprints_raw.npy'))
    labels_wavelet = np.load(os.path.join(data_dir, f'labels_wavelet_k{k}.npy'))
    labels_raw = np.load(os.path.join(data_dir, f'labels_raw_k{k}.npy'))

    # Load overlap summary
    overlap_csv = os.path.join(results_dir, 'cross_run_overlap_summary.csv')
    overlap_df = pd.read_csv(overlap_csv)

    print("Computing shared 2D UMAP...")
    emb_raw, emb_wav = compute_shared_umap_2d(fp_wavelet, fp_raw)

    print("Building color arrays...")
    colors_start, colors_end, is_stable = build_color_arrays(
        labels_wavelet, labels_raw, overlap_df, k
    )

    save_path = os.path.join(figs_dir, 'fig_drift_animation.mp4')
    os.makedirs(figs_dir, exist_ok=True)
    render_drift_animation(emb_raw, emb_wav, colors_start, colors_end,
                           is_stable, save_path)
    return save_path
