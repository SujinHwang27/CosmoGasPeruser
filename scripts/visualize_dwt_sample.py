import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#121212'

def main():
    # 1. Load sample data
    # Using Sample 100 from Class 4
    data_dir = "data/preprocessed/Sherwood_z0.3_inf/4"
    flux_path = os.path.join(data_dir, "flux.npy")
    flux_data = np.load(flux_path, mmap_mode='r')
    sample_idx = 100
    flux = flux_data[sample_idx]
    A = 1.0 - flux  # Absorption field
    
    # 2. Wavelet Decomposition
    wavelet_name = 'db8'
    coeffs = pywt.wavedec(A, wavelet_name, mode='periodization', level=6)
    # pywt.wavedec returns: [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
    
    # 3. Get Mother Wavelet for plotting
    wavelet = pywt.Wavelet(wavelet_name)
    phi, psi, x_wave = wavelet.wavefun(level=10)
    
    # 4. Create plotting grid
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(6, 2)
    
    # Row 0: Original Flux
    ax_flux = fig.add_subplot(gs[0, :])
    ax_flux.plot(flux, color='#00ff9f', linewidth=1.5)
    ax_flux.set_title(f'Original Flux Data | Sample {sample_idx}, Class 4', color='white', fontsize=14)
    ax_flux.set_ylabel('Normalized Flux')
    ax_flux.grid(alpha=0.1)

    # Row 1: Absorption Field (1-Flux)
    ax_abs = fig.add_subplot(gs[1, :])
    ax_abs.plot(A, color='#00d4ff', linewidth=1.5)
    ax_abs.set_title(f'Signal Input: Absorption (1-Flux)', color='white', fontsize=14)
    ax_abs.set_ylabel('Amplitude')
    ax_abs.grid(alpha=0.1)
    
    # Row 2: Mother Wavelet and A6
    ax_mw = fig.add_subplot(gs[2, 0])
    ax_mw.plot(x_wave, psi, color='#ff007c', linewidth=2)
    ax_mw.set_title('db8 Mother Wavelet (psi)', color='white')
    ax_mw.grid(alpha=0.1)

    ax_a6 = fig.add_subplot(gs[2, 1])
    ax_a6.plot(coeffs[0], color='#ffffff', linewidth=1.5)
    ax_a6.set_title('Approximation: A6', color='white')
    ax_a6.grid(alpha=0.1)
    
    # Rows 3-5: Detail Levels D6 down to D1
    levels = [6, 5, 4, 3, 2, 1]
    colors = ['#f9d423', '#fbc2eb', '#e14eca', '#a18cd1', '#4facfe', '#43e97b']
    
    for i, (lvl, color) in enumerate(zip(levels, colors)):
        row = 3 + (i // 2)
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        # coeffs indexing: [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
        d_coeffs = coeffs[7 - lvl]
        ax.plot(d_coeffs, color=color, linewidth=1)
        ax.set_title(f'Detail Level: D{lvl}', color='white')
        ax.grid(alpha=0.1)

    plt.tight_layout()
    output_dir = "data/feature_discovery/experiments/wavelet_per_level"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dwt_decomposition_sample_{sample_idx}.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
