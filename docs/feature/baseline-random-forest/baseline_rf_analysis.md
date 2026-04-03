# Baseline Random Forest Analysis: feature/baseline-random-forest

## 1. Context and Objective
The `feature/baseline-random-forest` branch implements a robust baseline for classifying synthetic quasar absorption spectra across four physical feedback models: **NoFeedback**, **StellarWind**, **WindAGN**, and **WindStrongAGN**. This baseline serves as a benchmark for future architectural developments (e.g., 1D-CNNs) and validates the multi-scale sensitivity of wavelet-transformed spectral features.

## 2. Implementation Overview

### 2.1 Classification Pipeline
The classification pipeline is modularized into data ingestion, transformation, and model units, synchronized via the `src.core` package.

```mermaid
graph LR
    D[DataIngestor] -->|Raw/Wavelet| T[WaveletTransform]
    T -->|Features| M[BaselineRFClassifier]
    M -->|Predictions| V[MLflow Tracking]
    V -->|Artifacts| R[Results/CMs]
```

### 2.2 Core Modules & Scripts
- **[src/core/models.py](file:///\\wsl.localhost\Ubuntu\home\sujin\CosmoGasPeruser\src\core\models.py)**: Contains the `BaselineRFClassifier` logic (RandomForest + StandardScaling).
- **[src/core/data.py](file:///\\wsl.localhost\Ubuntu\home\sujin\CosmoGasPeruser\src\core\data.py)**: Contains `DataIngestor` for standardizing 4-class folder loading.
- **[src/core/transforms.py](file:///\\wsl.localhost\Ubuntu\home\sujin\CosmoGasPeruser\src\core\transforms.py)**: Implements the 6-level DWT (`db8`) and DCT logic.
- **[train_rf_baseline.py](file:///\\wsl.localhost\Ubuntu\home\sujin\CosmoGasPeruser\scripts\baseline\train_rf_baseline.py)**: The primary entry point for training and CV.
- **[eda_sherwood.py](file:///\\wsl.localhost\Ubuntu\home\sujin\CosmoGasPeruser\scripts\eda_sherwood.py)**: Exploratory Data Analysis script.

### 2.3 MLflow Tracking Strategy
Experiments are tracked in the **"Baseline Random Forest"** MLflow experiment:
- **Parent-Child Hierarchy**: A "Global Session" run groups individual experiments (e.g., "RF - Raw Spectra", "RF - Wavelet D1") as nested runs.
- **Metrics & Scoping**: Each run logs specific metrics like `mean_accuracy`, `std_accuracy`, and the best hyperparameters.
- **Artifact Store**: Confusion matrices (`.png`) and summary plots are stored in the local `mlartifacts/` directory.

## 3. Theoretical Basis & Mathematics

### 3.1 On Using 1 − F (Absorption Field)

Raw flux $F = e^{-\tau}$ ranges from 0 (complete absorption) to 1 (no absorption). The signal of interest—the absorbing gas—lives in the *deviations from 1*. Transforming to the absorption field **$A = 1 - F$** is essential for classifier performance.

**Impact on Wavelet Coefficients:**
| Component | Detail Coefficients ($D_n$) | Approximation Coefficients ($A_n$) |
|:---:|---|---|
| **Effect of 1-F** | Magnitudes are identical; signs flip. | Meaningfully different. |
| **Why?** | High-pass filters only see edges. | Low-pass filters average the signal level. |

**Rationale for SVM/ML Models:**
The dot product $\mathbf{x}^\top \mathbf{x}'$ in kernels is swamped by the $0.8$ baseline in raw flux $F$. With $A$, the dot product is near zero if absorption features do not overlap, allowing the model to focus on **physically meaningful similarity**.

**Mathematical Example (Haar Wavelet):**
Consider signal: $F = [1.0, 1.0, 1.0, 0.3, 0.3, 1.0, 1.0, 1.0]$.
Corresponding Absorption: $A = [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0]$.
Applying Haar decomposition ($a_k = \frac{x_{2k} + x_{2k+1}}{\sqrt{2}}$, $d_k = \frac{x_{2k} - x_{2k+1}}{\sqrt{2}}$):
- **Flux $F$ Approximation:** `[1.41, 0.92, 0.92, 1.41]` (Non-zero baseline)
- **Absorption $A$ Approximation:** `[0.00, 0.49, 0.49, 0.00]` (Zero baseline)

### 3.2 Flux Thresholding (0.95 - 1.0 Range)
Pixels with $F \in [0.95, 1.0]$ correspond to $A \in [0, 0.05]$ (weak absorption).
- **Hard Thresholding (Ignoring them)**: Not recommended. It introduces discontinuities that create high-frequency artifacts (Gibbs ringing) in transforms.
- **Current Strategy**: Keep all values of $A = 1 - F$ to preserve cosmological information from low-density regions (void statistics).

### 3.3 Wavelet Decomposition Details
We use a **6-level Discrete Wavelet Transform (DWT)** with the **Daubechies 8 (`db8`)** family.

| Level | Scale (pixels) | Length | Physical Interpretation |
|---|---|---|---|
| Detail 1 | 1–2 | 1024 | Sub-resolution noise |
| Detail 2 | 2–4 | 512 | Narrowest absorption lines |
| Detail 3 | 4–8 | 256 | Typical Lyman-alpha line widths |
| Detail 4 | 8–16 | 128 | Broad lines, blended systems |
| Detail 5 | 16–32 | 64 | Large-scale clustering, DLA wings |
| Detail 6 | 32–64 | 32 | Large-scale structure |
| Approximation 6 | 64+ | 32 | Continuum shape, mean flux |

**Boundary Handling**: `mode='periodization'` is used to ensure the DWT is a perfect orthogonal decomposition totaling 2048 coefficients.

### 3.4 Stage 1 Empirical Results — 16,384 Sightlines (db8, 6 levels)

| Level | Coeff Len | Total Dim | Mean | Std | Min | Max | Sparsity (\|x\| < 1e-4) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 1024 | 4096 | ≈ 0.000 | 0.000102 | -0.0368 | 0.0448 | **99.5%** |
| D2 | 512 | 2048 | ≈ 0.000 | 0.001112 | -0.2000 | 0.1875 | 78.8% |
| D3 | 256 | 1024 | ≈ 0.000 | 0.010451 | -0.5212 | 0.4914 | 45.3% |
| D4 | 128 | 512 | ≈ 0.000 | 0.057702 | -2.2631 | 1.2349 | 15.7% |
| D5 | 64 | 256 | ≈ 0.000 | 0.168914 | -3.0656 | 2.6027 | 2.3% |
| D6 | 32 | 128 | ≈ 0.000 | 0.319480 | -4.3922 | 4.5365 | 0.3% |
| A6 | 32 | 128 | 0.1770 | 0.543933 | -2.2452 | 9.5971 | 0.2% |

**Energy Per Level (Standardized via `src.core.utils.calculate_energy`):**

| Level | NoFeedback | StellarWind | WindAGN | WindStrongAGN |
|:---:|---:|---:|---:|---:|
| D1 | 5.28e-7 | 1.50e-5 | 1.34e-5 | 1.36e-5 |
| D2 | 2.86e-4 | 8.29e-4 | 7.13e-4 | 7.03e-4 |
| D3 | 0.02669 | 0.03362 | 0.03095 | 0.02059 |
| D4 | 0.4453 | 0.5035 | 0.4696 | 0.2864 |
| D5 | 1.8619 | 2.1213 | 1.9869 | 1.3340 |
| D6 | 3.1412 | 3.7842 | 3.5179 | 2.6213 |
| A6 | 8.3881 | 13.734 | 11.928 | 7.830 |

## 4. Performance Metrics

The following 10-fold Cross-Validation results were obtained on the full 65,536-spectrum ensemble:

| Experiment | Accuracy | Std Dev | n_estimators | max_depth | min_split | min_leaf | max_features |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF - Raw Spectra** | **0.4514** | 0.0023 | 100 | 25 | 100 | 50 | 0.1 |
| RF - Concatenated Wavelet | 0.3139 | 0.0030 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet D1 | 0.3232 | 0.0031 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet D2 | 0.2946 | 0.0031 | 100 | 25 | 50 | 50 | 0.1 |
| RF - Wavelet D3 | 0.2745 | 0.0043 | 100 | 25 | 50 | 50 | 0.1 |
| RF - Wavelet D4 | 0.2334 | 0.0028 | 100 | 25 | 50 | 50 | 0.1 |
| RF - Wavelet D5 | 0.2154 | 0.0021 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet D6 | 0.2094 | 0.0031 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet A6 | 0.2177 | 0.0015 | 100 | 25 | 100 | 50 | sqrt |

### Discovery: The Resolution Gap
Performance peaks at **D1 (~0.32)** and drops monotonically to **D6 (~0.21)**. This confirms that the classification signal resides in high-frequency morphology. Raw spectra remain superior (0.45), suggesting that fixed-basis wavelets like `db8` may lose localized information compared to learnable convolutional filters.

## 5. Directory Structure
- `data/raw/`: Original FITS/TXT spectral data.
- `data/processed/wavelet_db8_l6_d12/`: Extracted DWT coefficients.
- `results/baseline_rf/`: Confusion matrices and performance summaries.
- `docs/feature/baseline-random-forest/`: This documentation.
