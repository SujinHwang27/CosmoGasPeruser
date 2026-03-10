"""
Stage 2: Micro-Probing for Signal Clustering Analysis

Computes 24-dimensional behavioral fingerprints for each spectral index
using RBF SVM one-vs-one micro-probing.

Outputs:
    - data/feature_discovery/fingerprints_wavelet.npy: Fingerprints from wavelet input
    - data/feature_discovery/fingerprints_raw.npy: Fingerprints from raw absorption input
    - results/stage2_fingerprint_summary.csv: Summary statistics
    - figs/fig_fingerprint_norms.png: L2 norm histograms
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.data import SignalClusteringData
from src.probe import run_probe


def plot_fingerprint_norms(fingerprints_wavelet, fingerprints_raw, save_path):
    """
    Plot side-by-side histograms of L2 norms for both runs.
    Uses the same x and y axis limits for both panels for fair comparison.
    """
    norms_wavelet = np.linalg.norm(fingerprints_wavelet, axis=1)
    norms_raw = np.linalg.norm(fingerprints_raw, axis=1)

    # Compute common axis limits for both panels
    x_max = max(np.max(norms_wavelet), np.max(norms_raw))
    y_max = max(np.max(np.histogram(norms_wavelet, bins=50)[0]),
                 np.max(np.histogram(norms_raw, bins=50)[0]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Wavelet
    axes[0].hist(norms_wavelet, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(norms_wavelet), color='red', linestyle='--', label=f'Mean: {np.mean(norms_wavelet):.3f}')
    axes[0].set_xlim(0, x_max * 1.05)
    axes[0].set_ylim(0, y_max * 1.1)
    axes[0].set_xlabel('L2 Norm')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Wavelet Fingerprint Norms')
    axes[0].legend()

    # Raw
    axes[1].hist(norms_raw, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(norms_raw), color='red', linestyle='--', label=f'Mean: {np.mean(norms_raw):.3f}')
    axes[1].set_xlim(0, x_max * 1.05)
    axes[1].set_ylim(0, y_max * 1.1)
    axes[1].set_xlabel('L2 Norm')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Raw Absorption Fingerprint Norms')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_fingerprint_summary(fingerprints_wavelet, fingerprints_raw, save_path):
    """
    Save summary statistics per dimension.
    """
    stats = []

    for dim in range(24):
        stats.append({
            'dimension': dim,
            'wavelet_mean': float(np.mean(fingerprints_wavelet[:, dim])),
            'wavelet_std': float(np.std(fingerprints_wavelet[:, dim])),
            'wavelet_min': float(np.min(fingerprints_wavelet[:, dim])),
            'wavelet_max': float(np.max(fingerprints_wavelet[:, dim])),
            'raw_mean': float(np.mean(fingerprints_raw[:, dim])),
            'raw_std': float(np.std(fingerprints_raw[:, dim])),
            'raw_min': float(np.min(fingerprints_raw[:, dim])),
            'raw_max': float(np.max(fingerprints_raw[:, dim])),
        })

    df = pd.DataFrame(stats)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Micro-Probing')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which feature set to probe')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Output directory for fingerprints')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Output directory for summaries')
    parser.add_argument('--figs_dir', type=str, default='figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 2: Micro-Probing")
    print("=" * 60)

    # Initialize data loader
    data = SignalClusteringData()

    fingerprints_wavelet = None
    fingerprints_raw = None

    if args.run in ['wavelet', 'both']:
        print("\n" + "=" * 40)
        print("Running Wavelet Probe")
        print("=" * 40)

        # Load pre-normalized wavelet per-class from Stage 1 output
        print("Loading pre-normalized wavelet features per class...")
        wavelet_dir = 'data/feature_discovery'
        X_wavelet_per_class = []
        for c in range(1, 5):
            X_wavelet_per_class.append(np.load(os.path.join(wavelet_dir, f'wavelet_class{c}.npy')))
        y = np.concatenate([np.full(16384, c) for c in range(1, 5)])
        print(f"  Loaded {len(X_wavelet_per_class)} classes, shape: {X_wavelet_per_class[0].shape}")

        # Run probe
        print("\nRunning micro-probing...")
        fingerprints_wavelet = run_probe(X_wavelet_per_class, n_jobs=args.n_jobs)

        # Save
        output_path = os.path.join(args.output_dir, 'fingerprints_wavelet.npy')
        np.save(output_path, fingerprints_wavelet)
        print(f"Saved: {output_path}")

    if args.run in ['raw', 'both']:
        print("\n" + "=" * 40)
        print("Running Raw Absorption Probe")
        print("=" * 40)

        # Load absorption per-class
        print("Loading flux and computing absorption per class...")
        X_flux_per_class, y = data.load_flux_per_class()
        X_abs_per_class = [np.clip(1.0 - flux, 0.0, 1.0) for flux in X_flux_per_class]
        print(f"  Loaded {len(X_abs_per_class)} classes, shape: {X_abs_per_class[0].shape}")

        # Run probe
        print("\nRunning micro-probing...")
        fingerprints_raw = run_probe(X_abs_per_class, n_jobs=args.n_jobs)

        # Save
        output_path = os.path.join(args.output_dir, 'fingerprints_raw.npy')
        np.save(output_path, fingerprints_raw)
        print(f"Saved: {output_path}")

    # Generate summary and plots if both runs completed
    if args.run == 'both' and fingerprints_wavelet is not None and fingerprints_raw is not None:
        print("\n" + "=" * 40)
        print("Generating Summaries and Plots")
        print("=" * 40)

        # Summary CSV
        summary_path = os.path.join(args.results_dir, 'stage2_fingerprint_summary.csv')
        save_fingerprint_summary(fingerprints_wavelet, fingerprints_raw, summary_path)

        # Plot fingerprint norms
        norms_path = os.path.join(args.figs_dir, 'fig_fingerprint_norms.png')
        plot_fingerprint_norms(fingerprints_wavelet, fingerprints_raw, norms_path)

        # Print statistics
        print("\nFingerprint Statistics:")
        print(f"  Wavelet: shape={fingerprints_wavelet.shape}, mean_norm={np.mean(np.linalg.norm(fingerprints_wavelet, axis=1)):.4f}")
        print(f"  Raw:     shape={fingerprints_raw.shape}, mean_norm={np.mean(np.linalg.norm(fingerprints_raw, axis=1)):.4f}")

    print("\n" + "=" * 60)
    print("Stage 2 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
