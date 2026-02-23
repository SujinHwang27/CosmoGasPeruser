# Tier 1 Metric Definitions

This document provides the mathematical definitions for the metrics implemented in the EDA pipeline.

## 1. Core Concepts

### Normalized Flux ($F$)
In absorption spectroscopy, the flux is normalized relative to the background continuum level.
$$F(\lambda) = \frac{I_{obs}(\lambda)}{I_{cont}(\lambda)}$$
*   $F = 1.0$: No absorption (continuum level).
*   $F = 0.0$: Total absorption (saturated line).

### Local Minimum ($\lambda_k$)
An absorption line center is defined as a local minimum in flux. In the implementation, these are found using `scipy.signal.find_peaks(-flux)`.
$$\left. \frac{dF}{d\lambda} \right|_{\lambda_k} = 0, \quad \left. \frac{d^2F}{d\lambda^2} \right|_{\lambda_k} > 0$$

### Absorption Line Count ($C_{lines}$)
The absolute number of absorption features (local minima) detected in a single spectrum.
$$C_{lines} = |\{\lambda_k\}|$$

---

## 2. Spectral Metrics

### Equivalent Width ($EW$)
The area between the absorption profile and the continuum level $F=1.0$.
*   **Total EW** ($EW_{total}$): Integrated across the entire spectral range.
    $$EW_{total} = \int_{\lambda_{min}}^{\lambda_{max}} (1 - F(\lambda)) \, d\lambda$$
*   **Local EW** ($EW_{local, k}$): Integrated around a specific minimum $\lambda_k$, bounded by neighboring saddle points (local maxima) $\lambda_{L}$ and $\lambda_{R}$.
    $$EW_{local, k} = \int_{\lambda_{L}}^{\lambda_{R}} (1 - F(\lambda)) \, d\lambda$$

### Absorption Depth ($D$)
The fractional depth of an absorption feature at its local minimum point.
$$D_k = 1 - F(\lambda_k)$$

### Line Density ($\rho_{lines}$)
The number of absorption lines detected per unit wavelength for a single spectrum.
$$\rho_{lines} = \frac{C_{lines}}{\lambda_{max} - \lambda_{min}}$$

### Gap ($G$)
The wavelength spacing between the centroids (local minima) of adjacent absorption features.
$$G_j = \lambda_{k+1} - \lambda_k$$

---

## 3. Statistical KPIs (Class-Level Means)

These metrics represent the average values calculated across all spectra ($N$) in a specific class.

| Metric | Formula |
| :--- | :--- |
| **Mean Total EW** | $\langle EW_{total} \rangle = \frac{1}{N} \sum_{i=1}^{N} EW_{total, i}$ |
| **Mean Depth** | $\langle D \rangle = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{C_{lines, i}} \sum_{k=1}^{C_{lines, i}} D_{k, i} \right)$ |
| **Mean Line Density** | $\langle \rho_{lines} \rangle = \frac{1}{N} \sum_{i=1}^{N} \rho_{lines, i}$ |
| **Mean Gap** | $\langle G \rangle = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{C_{lines, i} - 1} \sum_{j=1}^{C_{lines, i} - 1} G_{j, i} \right)$ |
| **Mean Absorption Line Count** | $\langle C_{lines} \rangle = \frac{1}{N} \sum_{i=1}^{N} C_{lines, i}$ |
