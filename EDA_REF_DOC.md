# EDA Refactoring Documentation (Tier 1)

## Overview
This document combines the implementation plan and task status for the refactoring of the Exploratory Data Analysis (EDA) code on the `feature/eda-sherwood` branch.

## Implementation Plan

### 1. Greyscale Stacked Visualization
- **Per-Class Files**: `greyscale_class_{c}.png` generated for each of the 4 classes.
- **Improved Visuals**: Added 0.5px white divider rows between spectra for a natural "stack" look.
- **Indexing**: Included spectrum index numbers (1-100) on the left Y-axis for clear data point identification.
- **Real Wavelength Axis**: Integrated `wave.npy` to show real wavelength values on the X-axis.

### 2. Tier 1 Metrics (Master Feature Extractor)
Implemented a robust, model-free feature extraction pipeline including:
- **Wavelength Integration**: All metrics are now calculated using actual wavelength values from `wave.npy` rather than pixel indices.
- **Total Equivalent Width (EW)**: Overall absorption strength.
- **Local EW Distribution**: Strength regime of individual features.
- **Line Density**: Feature count per wavelength unit.
- **Depth Dispersion**: Heterogeneity/mixing of absorbers.
- **Gap Statistics**: Clustering vs randomness (Poisson).

### 3. Visualization Playbook (2x2 Panels)
Generated high-density plots in 2x2 layouts (one panel per class) to provide executive-grade insights:
- **Total EW vs Line Density**: Scatter plot to distinguish strength vs count.
- **Local EW Distribution**: 2x2 Hist+KDE panels + 1 overlapping KDE comparison.
- **Depth vs Local EW**: Scatter plot checking for saturation/width trends.
- **Gap Distribution**: Hist + CDF overlay for clustering detection.
- **Mean Gap vs Line Density**: Packing effect vs physics-driven structure.
- **Depth Dispersion vs Line Density**: Crowding-driven heterogeneity.
- **Absorption Activity Profile**: 2x2 spatial frequency + 1 overlapping mean activity.
- **Activity vs Density (Per Bin)**: Binned correlation analysis.

## Task Checklist Status

- [x] Research and Understand Current EDA Implementation
- [x] Refactor Greyscale Stacked Visualization
    - [x] Generate per-class files
    - [x] Add dividers and indexing
- [x] Refactor Local Minima Statistical Analysis (Tier 1 Integration)
    - [x] Implement Tier 1 Metrics
    - [x] Implement 2x2 Panel Visualizations
- [/] Verification
    - [/] Cleanup and Git synchronization
