"""
Stage: Prepare Inputs for Signal Clustering Analysis

Loads wavelet features and raw flux, computes absorption field,
runs sanity checks, and produces validation plots.

Outputs:
    - results/stage1_input_summary.csv: Summary statistics
    - figs/fig_absorption_vs_flux.png: A(λ) vs F(λ) for representative sightlines
    - figs/fig_wavelet_scalogram.png: Wavelet coefficient visualization
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.data import SignalClusteringData


def plot_absorption_vs_flux(X_flux, X_abs, y, save_path):
    """
    Plot absorption vs flux for 4 representative sightlines (one per class).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    class_names = ['NoFeedback', 'StellarWind', 'WindAGN', 'WindStrongAGN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Take one representative sightline from each class
    # Class 1: index 0, Class 2: index 16384, Class 3: index 32768, Class 4: index 49152
    class_indices = [0, 16384, 32768, 49152]

    for idx, (ax, class_idx, class_name, color) in enumerate(zip(axes, class_indices, class_names, colors)):
        # Plot flux
        ax.plot(X_flux[class_idx], color=color, alpha=0.7, label='F (flux)')
        # Plot absorption on secondary axis
        ax2 = ax.twinx()
        ax2.plot(X_abs[class_idx], color=color, linestyle='--', alpha=0.7, label='A = 1-F')

        ax.set_xlabel('Pixel')
        ax.set_ylabel('Flux', color=color)
        ax2.set_ylabel('Absorption', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax.set_title(f'{class_name} (index {class_idx})')
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_wavelet_scalogram(X_wavelet, y, save_path):
    """
    Plot D1-D6 + A6 wavelet coefficients for one sightline per class.
    Shows the multi-scale decomposition structure.
    """
    # Wavelet level boundaries for db8 with 6 levels:
    # D1: 0-1024, D2: 1024-1536, D3: 1536-1792, D4: 1792-1920, D5: 1920-1984, D6: 1984-2016, A6: 2016-2048
    level_boundaries = {
        'D1': (0, 1024),
        'D2': (1024, 1536),
        'D3': (1536, 1792),
        'D4': (1792, 1920),
        'D5': (1920, 1984),
        'D6': (1984, 2016),
        'A6': (2016, 2048)
    }

    class_indices = [0, 16384, 32768, 49152]
    class_names = ['NoFeedback', 'StellarWind', 'WindAGN', 'WindStrongAGN']

    # Compute global y-axis limits across all classes being plotted
    y_min = min(X_wavelet[idx].min() for idx in class_indices)
    y_max = max(X_wavelet[idx].max() for idx in class_indices)
    # Add 5% padding
    y_margin = (y_max - y_min) * 0.05
    y_lim = (y_min - y_margin, y_max + y_margin)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    for row, (class_idx, class_name) in enumerate(zip(class_indices, class_names)):
        ax = axes[row]

        # Plot each wavelet level
        colors = plt.cm.viridis(np.linspace(0, 1, 7))
        for (level_name, (start, end)), color in zip(level_boundaries.items(), colors):
            coeffs = X_wavelet[class_idx, start:end]
            x_positions = np.arange(start, end)
            ax.plot(x_positions, coeffs, color=color, alpha=0.8, label=level_name, linewidth=0.5)

        ax.set_ylabel(class_name)
        ax.set_xlim(0, 2048)
        ax.set_ylim(y_lim)
        ax.legend(loc='upper right', ncol=7, fontsize=8)

    axes[-1].set_xlabel('Pixel Position')
    plt.suptitle('Wavelet Decomposition (D1-D6 + A6) per Class', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_summary_csv(X_wavelet, X_abs, y, save_path):
    """
    Save summary statistics to CSV.
    """
    # Overall statistics
    summary_data = {
        'wavelet_shape': [X_wavelet.shape],
        'wavelet_dtype': [str(X_wavelet.dtype)],
        'wavelet_min': [float(np.min(X_wavelet))],
        'wavelet_max': [float(np.max(X_wavelet))],
        'wavelet_mean': [float(np.mean(X_wavelet))],
        'wavelet_std': [float(np.std(X_wavelet))],
        'absorption_shape': [X_abs.shape],
        'absorption_dtype': [str(X_abs.dtype)],
        'absorption_min': [float(np.min(X_abs))],
        'absorption_max': [float(np.max(X_abs))],
        'absorption_mean': [float(np.mean(X_abs))],
        'absorption_std': [float(np.std(X_abs))],
    }

    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare Inputs for Signal Clustering')
    parser.add_argument('--output_dir', type=str, default='results/signal_clustering_v2',
                        help='Output directory for results')
    parser.add_argument('--figs_dir', type=str, default='results/signal_clustering_v2/figs',
                        help='Output directory for figures')
    parser.add_argument('--flux_path', type=str,
                        default='data/preprocessed/Sherwood_z0.3_inf',
                        help='Path to flux data')
    parser.add_argument('--wavelet_path', type=str,
                        default='data/processed/wavelet_db8_l6_d12',
                        help='Path to wavelet data')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Preparing Inputs")
    print("=" * 60)

    # Initialize data loader
    data = SignalClusteringData(flux_path=args.flux_path, wavelet_path=args.wavelet_path)

    # Load data (wavelet with per-level z-score normalization across all data)
    print("\nLoading wavelet features (normalized per level across all classes)...")
    X_wavelet_per_class, y = data.load_wavelet_global_normalized()
    # Stack for overall statistics
    X_wavelet = np.vstack(X_wavelet_per_class)
    print(f"  Wavelet shape: {X_wavelet.shape}")

    # Save normalized wavelet per-class for downstream stages
    wavelet_output_dir = 'data/feature_discovery'
    os.makedirs(wavelet_output_dir, exist_ok=True)
    for c, X in enumerate(X_wavelet_per_class, start=1):
        np.save(os.path.join(wavelet_output_dir, f'wavelet_class{c}.npy'), X)
    print(f"  Saved normalized wavelets to: {wavelet_output_dir}/wavelet_class[1-4].npy")

    print("\nLoading flux and computing absorption...")
    X_flux, _ = data.load_flux()
    X_abs = np.clip(1.0 - X_flux, 0.0, 1.0)
    print(f"  Absorption shape: {X_abs.shape}")

    # Run validation
    print("\nRunning validation checks...")
    validation = data.validate(X_wavelet, X_abs, y)

    print(f"  Wavelet valid: {validation['wavelet']['valid']}")
    print(f"  Absorption valid: {validation['absorption']['valid']}")
    print(f"  Class distribution valid: {validation['class_valid']}")
    print(f"  Overall valid: {validation['all_valid']}")

    if not validation['all_valid']:
        print("\nWARNING: Validation failed! Check results carefully.")

    # Print class distribution
    print("\nClass distribution:")
    for cls, count in validation['class_distribution'].items():
        print(f"  Class {cls}: {count}")

    # Print absorption statistics
    print(f"\nAbsorption statistics:")
    print(f"  Mean: {validation['absorption']['mean']:.4f}")
    print(f"  Range: [{validation['absorption']['shape'][0]} samples x {validation['absorption']['shape'][1]} pixels]")

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, 'stage1_input_summary.csv')
    save_summary_csv(X_wavelet, X_abs, y, summary_path)

    # Generate plots
    print("\nGenerating validation plots...")

    # Plot 1: Absorption vs Flux (need to reload flux for this)
    print("  Reloading flux for absorption vs flux plot...")
    X_flux_for_plot, _ = data.load_flux()
    X_abs_for_plot = np.clip(1.0 - X_flux_for_plot, 0.0, 1.0)
    plot_path1 = os.path.join(args.figs_dir, 'fig_absorption_vs_flux.png')
    plot_absorption_vs_flux(X_flux_for_plot, X_abs_for_plot, y, plot_path1)
    del X_flux_for_plot, X_abs_for_plot

    # Plot 2: Wavelet Scalogram
    plot_path2 = os.path.join(args.figs_dir, 'fig_wavelet_scalogram.png')
    plot_wavelet_scalogram(X_wavelet, y, plot_path2)

    print("\n" + "=" * 60)
    print("Stage 1 Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - {summary_path}")
    print(f"  - {plot_path1}")
    print(f"  - {plot_path2}")


if __name__ == '__main__':
    main()
