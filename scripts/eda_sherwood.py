import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from src.core.data import DataIngestor

def calculate_local_minima(spectrum, threshold=0.95):
    """
    Identifies local minima with weighting and saturation handling.
    Pre-processing: values between threshold and 1.0 become threshold.
    Detection: f[i] <= f[i-1] and f[i] <= f[i+1].
    Saturation: Sequential zeros are treated as one feature at the midpoint.
    Weighting: w = 1 - f_min.
    """
    proc_spec = np.copy(spectrum)
    proc_spec[(proc_spec >= threshold) & (proc_spec <= 1.0)] = threshold
    
    indices = []
    weights = []
    
    n = len(proc_spec)
    i = 1
    while i < n - 1:
        # Check for sequential zeros (saturation)
        if proc_spec[i] == 0:
            start = i
            while i < n - 1 and proc_spec[i] == 0:
                i += 1
            mid = (start + i - 1) // 2
            indices.append(mid)
            weights.append(1.0) # weight for flux 0 is 1.0
            continue
            
        # Standard local minima
        if proc_spec[i] <= proc_spec[i-1] and proc_spec[i] <= proc_spec[i+1]:
            # Only count if it's below threshold (meaning it's an actual feature)
            if proc_spec[i] < threshold:
                indices.append(i)
                weights.append(1.0 - proc_spec[i])
        i += 1
        
    return indices, weights

def perform_refined_eda(base_path, output_dir="eda_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {base_path}...")
    ingestor = DataIngestor(base_path, filename="flux.npy", num_classes=4)
    X, y = ingestor.load()
    
    n_samples, n_features = X.shape
    classes = np.unique(y)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. & 2. Data Inventory and Global Stats (Tables)
    stats_lines = ["Class,Count,Mean,Median,Std,Saturated_Pct"]
    balance_lines = ["Class,Count"]
    
    for c in classes:
        mask = (y == c)
        class_data = X[mask]
        count = len(class_data)
        mean_val = np.mean(class_data)
        med_val = np.median(class_data)
        std_val = np.std(class_data)
        sat_pct = np.mean(class_data < 0.01) * 100 # Near 0
        
        stats_lines.append(f"{c},{count},{mean_val:.4f},{med_val:.4f},{std_val:.4f},{sat_pct:.2f}%")
        balance_lines.append(f"{c},{count}")
        
    with open(os.path.join(output_dir, "global_stats.txt"), "w") as f:
        f.write("\n".join(stats_lines))
    with open(os.path.join(output_dir, "class_balance.txt"), "w") as f:
        f.write("\n".join(balance_lines))
        
    # 3. Spectral Analysis (Power Spectrum)
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(classes):
        class_data = X[y == c]
        fft_vals = np.abs(fft(class_data, axis=1))
        mean_ps = np.mean(fft_vals[:, :n_features//2], axis=0)
        plt.loglog(mean_ps, label=f"Class {c}", color=colors[i])
    plt.title("Mean 1D Power Spectrum")
    plt.xlabel("k")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(output_dir, "power_spectra.png"))
    plt.close()
    
    # 4. Multi-Sample Comparative Inspection (Indices 1-5)
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    for idx_in_class in range(5):
        ax = axes[idx_in_class]
        for i, c in enumerate(classes):
            # We assume indices align across folders because they are simulation seeds
            # DataIngestor stacks them, so find class start
            class_offset = np.where(y == c)[0][0]
            ax.plot(X[class_offset + idx_in_class], label=f"Class {c}", color=colors[i], alpha=0.7)
        ax.set_title(f"Comparison of Data Index {idx_in_class + 1}")
        ax.set_ylabel("Flux")
        if idx_in_class == 0:
            ax.legend(loc='upper right', ncol=4)
    plt.xlabel("Pixel Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_sample_comparison.png"))
    plt.close()
    
    # 5. Greyscale Stacked Visualization (Indices 1-100)
    fig, axes = plt.subplots(4, 1, figsize=(15, 8))
    for i, c in enumerate(classes):
        mask = (y == c)
        stack = X[mask][:100]
        axes[i].imshow(stack, cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[i].set_title(f"Class {c} (Indices 1-100)")
        axes[i].set_ylabel("Sample Index")
    plt.tight_label = False
    plt.xlabel("Pixel Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "greyscale_stacks.png"))
    plt.close()
    
    # 6. Local Minima Statistical Analysis
    print("Running Local Minima Analysis...")
    all_weighted_counts = {c: [] for c in classes}
    spatial_weights = {c: np.zeros(n_features) for c in classes}
    
    for i in range(n_samples):
        c = y[i]
        indices, weights = calculate_local_minima(X[i])
        weighted_count = sum(weights)
        all_weighted_counts[c].append(weighted_count)
        for idx, w in zip(indices, weights):
            spatial_weights[c][idx] += w
            
    # Plots and Tables for Local Minima
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(classes):
        plt.hist(all_weighted_counts[c], bins=30, alpha=0.5, label=f"Class {c}", color=colors[i])
    plt.title("Distribution of Weighted Local Minima Count")
    plt.xlabel("Weighted Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "local_minima_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    for i, c in enumerate(classes):
        plt.plot(spatial_weights[c], label=f"Class {c}", color=colors[i], alpha=0.6)
    plt.title("Spatial Frequency of Weighted Local Minima")
    plt.xlabel("Pixel Index")
    plt.ylabel("Summed Weights")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "local_minima_spatial.png"))
    plt.close()
    
    minima_stats = ["Class,Mean_Weighted_Count,Std_Weighted_Count,Top_3_Indices"]
    for c in classes:
        counts = all_weighted_counts[c]
        mean_c = np.mean(counts)
        std_c = np.std(counts)
        top_indices = np.argsort(spatial_weights[c])[-3:][::-1]
        top_str = ";".join(map(str, top_indices))
        minima_stats.append(f"{c},{mean_c:.2f},{std_c:.2f},{top_str}")
        
    with open(os.path.join(output_dir, "local_minima_stats.txt"), "w") as f:
        f.write("\n".join(minima_stats))
    
    print(f"Refined EDA completed. Files saved to {output_dir}")

if __name__ == "__main__":
    base_data_path = "data/preprocessed/Sherwood_z0.3_inf/"
    perform_refined_eda(base_data_path)
