# EDA Refactoring Documentation (Tier 1)

## Overview
This document combines the implementation plan, task status, and final walkthrough for the refactoring of the Exploratory Data Analysis (EDA) code on the `feature/eda-sherwood` branch.

## Implementation Details

### 1. Advanced Feature Extraction
The script now implements the **Master Tier 1 Feature Extractor**, which calculates:
- **Real Wavelength Integration**: All metrics are now calculated using actual wavelength values from `wave.npy` rather than pixel indices.
- **Strength**: Total EW, Mean/Std Local EW, Max Depth.
- **Structure**: Line Density, Gap (spacing) statistics.
- **Heterogeneity**: Depth Dispersion (Std of absorption depths).
- **Spatial Distribution**: Absorption Activity Profile (EW per wavelength bin).

### 2. Enhanced Greyscale Stacked Visualization
- **Per-Class Files**: `greyscale_class_{c}.png` generated for each of the 4 classes.
- **Improved Visuals**: Added thin 0.5px white divider rows between spectra for a natural "stack" look.
- **Indexing**: Included spectrum index numbers (1-100) on the left Y-axis for clear data point identification.
- **Real Wavelength Axis**: Integrated `wave.npy` to show real wavelength values on the X-axis.

### 3. Tier 1 Visualization Playbook (2x2 Panels)
I implemented 2x2 panel layouts for all key metrics to allow easy cross-class comparison:
- **Strength vs Density**: `tier1_ew_vs_density_2x2.png`
- **Feature Distribution**: `tier1_local_ew_dist_2x2.png` & `tier1_local_ew_kde_overlap.png`
- **Saturation Analysis**: `tier1_depth_vs_ew_2x2.png`
- **Clustering/Spacing**: `tier1_gap_dist_2x2.png` & `tier1_mean_gap_vs_density_2x2.png`
- **Complexity**: `tier1_depth_disp_vs_density_2x2.png`
- **Spatial Activity**: `tier1_activity_profile_2x2.png` & `tier1_activity_overlap.png`
- **Binned Correlation**: `tier1_activity_vs_density_binned_2x2.png`

## Task Checklist Status

- [x] Research and Understand Current EDA Implementation
- [x] Refactor Greyscale Stacked Visualization
    - [x] Generate per-class files
    - [x] Add dividers and indexing
- [x] Refactor Local Minima Statistical Analysis (Tier 1 Integration)
    - [x] Implement Tier 1 Metrics
    - [x] Implement 2x2 Panel Visualizations
- [x] Integrate Real Wavelength Data (wave.npy)
- [x] Verification
    - [x] Cleanup and Documentation Sync
