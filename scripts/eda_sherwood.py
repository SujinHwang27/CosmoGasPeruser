import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from tqdm import tqdm
from src.core.data import DataIngestor

# Constants
ABS_THRESH = 0.05
EPS = 1e-12

def detect_local_minima(flux):
    """Detects absorption dip centers."""
    minima_idx, _ = find_peaks(-flux)
    return minima_idx

def total_equivalent_width(lambda_arr, flux):
    """Numerical EW across entire spectrum."""
    return np.trapz(1.0 - flux, lambda_arr)

def local_equivalent_widths(lambda_arr, flux, minima_idx):
    """Computes pseudo-EW around each local minimum."""
    local_ews = []
    for i in minima_idx:
        left = i
        right = i
        # Expand left
        while left > 0 and flux[left] < 1.0:
            left -= 1
        # Expand right
        while right < len(flux) - 1 and flux[right] < 1.0:
            right += 1
        ew = np.trapz(1.0 - flux[left:right], lambda_arr[left:right])
        local_ews.append(ew)
    return np.array(local_ews)

def absorption_depths(flux, minima_idx):
    """Extracts depth (1.0 - flux) at minima."""
    return 1.0 - flux[minima_idx]

def line_density(minima_idx, lambda_arr):
    """Number of lines per wavelength unit."""
    return len(minima_idx) / (lambda_arr[-1] - lambda_arr[0]) if len(lambda_arr) > 1 else 0

def gap_statistics(lambda_arr, minima_idx):
    """Calculates spacing between absorbers."""
    if len(minima_idx) < 2:
        return {"gap_mean": 0.0, "gap_std": 0.0, "gap_min": 0.0, "gap_max": 0.0, "gaps": np.array([])}
    positions = np.sort(lambda_arr[minima_idx])
    gaps = np.diff(positions)
    return {
        "gap_mean": np.mean(gaps),
        "gap_std": np.std(gaps),
        "gap_min": np.min(gaps),
        "gap_max": np.max(gaps),
        "gaps": gaps
    }

def absorption_activity_profile(lambda_arr, flux, binedges):
    """EW per wavelength bin."""
    activity = np.zeros(len(binedges) - 1)
    # Also return density per bin for Plot 8
    counts = np.zeros(len(binedges) - 1)
    minima_idx = detect_local_minima(flux)
    for i in range(len(binedges) - 1):
        mask = (lambda_arr >= binedges[i]) & (lambda_arr < binedges[i+1])
        if np.any(mask):
            activity[i] = np.trapz(1.0 - flux[mask], lambda_arr[mask])
        bin_minima = minima_idx[(lambda_arr[minima_idx] >= binedges[i]) & (lambda_arr[minima_idx] < binedges[i+1])]
        counts[i] = len(bin_minima)
    
    bin_widths = np.diff(binedges)
    densities = counts / bin_widths
    return activity, densities

def extract_tier1_features(lambda_arr, flux, wavelength_bins):
    """Master Tier 1 Feature Extractor."""
    minima_idx = detect_local_minima(flux)
    local_ews = local_equivalent_widths(lambda_arr, flux, minima_idx)
    depths = absorption_depths(flux, minima_idx)
    gaps_info = gap_statistics(lambda_arr, minima_idx)
    activity, bin_densities = absorption_activity_profile(lambda_arr, flux, wavelength_bins)
    
    features = {
        "total_ew": total_equivalent_width(lambda_arr, flux),
        "ew_mean": np.mean(local_ews) if len(local_ews) > 0 else 0,
        "ew_std": np.std(local_ews) if len(local_ews) > 0 else 0,
        "line_density": line_density(minima_idx, lambda_arr),
        "depth_mean": np.mean(depths) if len(depths) > 0 else 0,
        "depth_std": np.std(depths) if len(depths) > 0 else 0,
        "depth_max": np.max(depths) if len(depths) > 0 else 0,
        "gap_mean": gaps_info["gap_mean"],
        "gap_std": gaps_info["gap_std"],
        "activity_profile": activity,
        "bin_densities": bin_densities, # For plot 8
        "raw_counts": len(minima_idx),
        # Extra lists for visualization
        "local_ews": local_ews,
        "depths": depths,
        "gaps": gaps_info["gaps"]
    }
    return features

def perform_refined_eda(base_path, output_dir="eda_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {base_path}...")
    ingestor = DataIngestor(base_path, filename="flux.npy", num_classes=4)
    X, y = ingestor.load()
    
    # Load wavelength data (standardized across classes)
    wave_path = os.path.join(base_path, "1", "wave.npy")
    if os.path.exists(wave_path):
        lambda_arr = np.load(wave_path)
        print(f"Loaded wavelength array from {wave_path} (Range: {lambda_arr.min():.2f} - {lambda_arr.max():.2f})")
    else:
        print("wave.npy not found, falling back to pixel indices.")
        lambda_arr = np.arange(X.shape[1])
        
    n_samples, n_features = X.shape
    classes = np.unique(y)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    wavelength_bins = np.linspace(lambda_arr.min(), lambda_arr.max(), 51) # 50 bins
    
    # 1. Greyscale Stacked Visualization (Per-Class Files)
    print("Generating Greyscale Stacks (Per Class)...")
    for i, c in enumerate(classes):
        mask = (y == c)
        stack = X[mask][:100]
        n_stack = stack.shape[0]
        
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.imshow(stack, cmap='gray', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        
        # Add thin white lines between rows
        for row in range(n_stack + 1):
            ax.axhline(row - 0.5, color='white', linewidth=0.5, alpha=0.8)
            
        ax.set_title(f"Class {c} Greyscale Stack (Indices 1-100)")
        ax.set_ylabel("Spectrum Index")
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        
        # Update X-ticks to show wavelength
        x_ticks = np.linspace(0, n_features-1, 6)
        x_tick_labels = [f"{lambda_arr[int(i)]:.1f}" for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        # We'll label every index but only if readable, or a subset. Let's try every 5th first.
        indices_to_label = np.arange(0, n_stack, 5)
        ax.set_yticks(indices_to_label)
        ax.set_yticklabels(indices_to_label + 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"greyscale_class_{c}.png"))
        plt.close()

    # 2. Tier 1 Feature Extraction
    print("Extracting Tier 1 Features...")
    all_features = {c: [] for c in classes}
    for i in tqdm(range(n_samples), desc="Extracting features"):
        feat = extract_tier1_features(lambda_arr, X[i], wavelength_bins)
        all_features[y[i]].append(feat)
    
    # 3. Tier 1 Visualization Playbook (2x2 Panels)
    print("Generating Tier 1 Visualization Playbook...")
    
    # helper for 2x2
    def get_axes_2x2():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        return fig, axes.flatten()

    # Plot 1: Total EW vs Line Density (1 file, 2x2)
    fig, axes = get_axes_2x2()
    for i, c in enumerate(classes):
        ews = [f["total_ew"] for f in all_features[c]]
        densities = [f["line_density"] for f in all_features[c]]
        axes[i].scatter(densities, ews, color=colors[i], alpha=0.5, s=15)
        axes[i].set_title(f"Class {c}: EW vs Density")
        axes[i].set_xlabel(r"Line Density (lines/$\mathrm{\AA}$)")
        axes[i].set_ylabel(r"Total EW ($\mathrm{\AA}$)")
        axes[i].grid(True, alpha=0.2)
    fig.suptitle("Total Absorption vs Line Density")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_ew_vs_density_2x2.png"))
    plt.close()

    # Plot 2: Local EW Distribution (2 files)
    # File 2a: Hist + KDE 2x2
    fig, axes = get_axes_2x2()
    all_kde_data = {}
    for i, c in enumerate(classes):
        data = np.concatenate([f["local_ews"] for f in all_features[c] if len(f["local_ews"]) > 0])
        axes[i].hist(data, bins=40, density=True, alpha=0.3, color=colors[i])
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(0, np.max(data), 200)
            axes[i].plot(x_range, kde(x_range), color=colors[i], lw=2)
            all_kde_data[c] = (x_range, kde(x_range))
        axes[i].set_title(f"Class {c}: Local EW Dist")
        axes[i].set_xlabel("Local EW")
    fig.suptitle("Local EW Distribution (Hist + KDE)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_local_ew_dist_2x2.png"))
    plt.close()

    # File 2b: Overlapping KDE
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(classes):
        if c in all_kde_data:
            x, y_val = all_kde_data[c]
            plt.plot(x, y_val, color=colors[i], label=f"Class {c}", lw=2)
    plt.title("Comparative Local EW KDE")
    plt.xlabel("Local EW")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "tier1_local_ew_kde_overlap.png"))
    plt.close()

    # Plot 3: Depth vs Local EW (1 file, 2x2)
    fig, axes = get_axes_2x2()
    for i, c in enumerate(classes):
        all_depths = np.concatenate([f["depths"] for f in all_features[c] if len(f["depths"]) > 0])
        all_ews = np.concatenate([f["local_ews"] for f in all_features[c] if len(f["local_ews"]) > 0])
        axes[i].scatter(all_depths, all_ews, color=colors[i], alpha=0.2, s=5)
        axes[i].set_title(f"Class {c}: Depth vs Local EW")
        axes[i].set_xlabel("Depth")
        axes[i].set_ylabel("Local EW")
    fig.suptitle("Depth vs Local Equivalent Width")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_depth_vs_ew_2x2.png"))
    plt.close()

    # Plot 4: Gap Distribution (1 file, 2x2 panel: Hist + CDF overlay)
    fig, axes = get_axes_2x2()
    for i, c in enumerate(classes):
        all_gaps = np.concatenate([f["gaps"] for f in all_features[c] if len(f["gaps"]) > 0])
        if len(all_gaps) > 0:
            axes[i].hist(all_gaps, bins=40, density=True, alpha=0.3, color=colors[i], label="Hist")
            sorted_gaps = np.sort(all_gaps)
            cdf = np.arange(len(sorted_gaps)) / float(len(sorted_gaps))
            # Overlay CDF on twin axis
            ax2 = axes[i].twinx()
            ax2.plot(sorted_gaps, cdf, color='black', lw=1.5, label="CDF")
            ax2.set_ylim(0, 1.05)
            if i == 0: axes[i].legend(loc='upper right')
        axes[i].set_title(f"Class {c}: Gap Dist (Hist + CDF)")
        axes[i].set_xlabel(r"Gap ($\mathrm{\AA}$)")
    fig.suptitle("Gap Distribution with CDF Overlays")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_gap_dist_2x2.png"))
    plt.close()

    # Plot 5: Mean Gap vs Line Density (1 file, 2x2)
    fig, axes = get_axes_2x2()
    for i, c in enumerate(classes):
        gaps = [f["gap_mean"] for f in all_features[c]]
        densities = [f["line_density"] for f in all_features[c]]
        axes[i].scatter(densities, gaps, color=colors[i], alpha=0.5, s=15)
        axes[i].set_title(f"Class {c}: Mean Gap vs Density")
        axes[i].set_xlabel("Line Density")
        axes[i].set_ylabel("Mean Gap")
    fig.suptitle("Mean Gap vs Line Density")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_mean_gap_vs_density_2x2.png"))
    plt.close()

    # Plot 6: Depth Dispersion vs Line Density (1 file, 2x2)
    fig, axes = get_axes_2x2()
    for i, c in enumerate(classes):
        depth_stds = [f["depth_std"] for f in all_features[c]]
        densities = [f["line_density"] for f in all_features[c]]
        axes[i].scatter(densities, depth_stds, color=colors[i], alpha=0.5, s=15)
        axes[i].set_title(f"Class {c}: Depth Std vs Density")
        axes[i].set_xlabel("Line Density")
        axes[i].set_ylabel("Depth Std")
    fig.suptitle("Depth Dispersion vs Line Density")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_depth_disp_vs_density_2x2.png"))
    plt.close()

    # Plot 7: Absorption Activity Profile (2 files)
    # File 7a: 2x2 panel
    fig, axes = get_axes_2x2()
    bin_centers = (wavelength_bins[:-1] + wavelength_bins[1:]) / 2
    for i, c in enumerate(classes):
        profiles = np.array([f["activity_profile"] for f in all_features[c]])
        mean_prof = np.mean(profiles, axis=0)
        std_prof = np.std(profiles, axis=0)
        axes[i].plot(bin_centers, mean_prof, color=colors[i], lw=2)
        axes[i].fill_between(bin_centers, mean_prof - std_prof, mean_prof + std_prof, color=colors[i], alpha=0.2)
        axes[i].set_title(f"Class {c}: Activity Profile")
        axes[i].set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        axes[i].set_ylabel("Activity (EW per bin)")
    fig.suptitle("Spatial Absorption Activity Profile")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_activity_profile_2x2.png"))
    plt.close()

    # File 7b: Overlapping
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(classes):
        profiles = np.array([f["activity_profile"] for f in all_features[c]])
        plt.plot(bin_centers, np.mean(profiles, axis=0), color=colors[i], label=f"Class {c}", lw=2)
    plt.title("Comparative Absorption Activity Overlap")
    plt.xlabel(r"Wavelength ($\mathrm{\AA}$)")
    plt.ylabel("Mean EW per bin")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "tier1_activity_overlap.png"))
    plt.close()

    # Plot 8: Activity vs Density (Per Bin) (1 file, 2x2)
    fig, axes = get_axes_2x2()
    for i, c in enumerate(classes):
        bin_dens = np.array([f["bin_densities"] for f in all_features[c]]) # (N_samples, N_bins)
        bin_acts = np.array([f["activity_profile"] for f in all_features[c]]) # (N_samples, N_bins)
        
        # Plot all bin points colored by bin index
        for b_idx in range(len(wavelength_bins) - 1):
            axes[i].scatter(bin_dens[:, b_idx], bin_acts[:, b_idx], alpha=0.1, s=2, c=[plt.cm.viridis(b_idx/50)], label=None)
        
        axes[i].set_title(f"Class {c}: Activity vs Density (Per Bin)")
        axes[i].set_xlabel("Bin Density")
        axes[i].set_ylabel("Bin Activity")
    fig.suptitle("Activity vs Density Correlation (Binned)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tier1_activity_vs_density_binned_2x2.png"))
    plt.close()

    # 4. Save KPIs
    print("Saving KPIs...")
    stats_lines = ["Class,Mean_Total_EW,Mean_Line_Density,Mean_Depth,Mean_Gap,Mean_Raw_Count"]
    for c in classes:
        m_ew = np.mean([f["total_ew"] for f in all_features[c]])
        m_dens = np.mean([f["line_density"] for f in all_features[c]])
        m_depth = np.mean([f["depth_mean"] for f in all_features[c]])
        m_gap = np.mean([f["gap_mean"] for f in all_features[c]])
        m_count = np.mean([f["raw_counts"] for f in all_features[c]])
        stats_lines.append(f"{c},{m_ew:.4f},{m_dens:.4f},{m_depth:.4f},{m_gap:.4f},{m_count:.2f}")
    
    with open(os.path.join(output_dir, "tier1_summary_stats.txt"), "w") as f:
        f.write("\n".join(stats_lines))
    
    print(f"Tier 1 EDA completed. Results in {output_dir}")

if __name__ == "__main__":
    base_data_path = "data/preprocessed/Sherwood_z0.3_inf/"
    perform_refined_eda(base_data_path)
