# Feature Extraction Pipeline for Synthetic Quasar Absorption Spectra
## Windowed DCT-II and Wavelet Decomposition for ML Classifiers

---

## Preliminary Decisions

### On Using 1 − F (Absorption Field)

This is worth doing. Raw flux $F = e^{-\tau}$ ranges from 0 (complete absorption) to 1 (no absorption). The signal you care about — the absorbing gas — lives in the *deviations from 1*. Transforming to:

$$A = 1 - F$$

means:

- No absorption → 0 (baseline silence, like background in an audio signal)
- Strong absorption → values approaching 1 (the "events" you want to detect)
- The DCT and wavelet coefficients will now directly encode *where and how much* absorption occurs, rather than encoding a near-constant signal with small dips

This is conceptually cleaner and practically better for feature extraction. The low-frequency DCT coefficients of $F$ are dominated by the mean flux level (~0.8 at $z \sim 2$), which carries some information but also swamps the variance of interest. With $A = 1 - F$, the DC component (coefficient 0) captures mean absorption, and higher coefficients capture structure — a more balanced decomposition.

**Caveat:** If your classifier needs to distinguish spectra partly by their mean flux level (e.g., different redshift bins have different mean absorption), keep the DC coefficient or add mean flux as an explicit separate feature rather than discarding it.

---

### On Ignoring Flux in the Range 0.95 − 1.0

Pixels with $F \in [0.95, 1.0]$ correspond to $A \in [0, 0.05]$ — very weak or no absorption. The question is whether zeroing these out (hard thresholding) is safe.

**Arguments for ignoring them:**
- These pixels are dominated by noise in real spectra
- For synthetic spectra they represent voids — physically uninteresting low-density regions
- Zeroing them out sharpens the contrast of absorption features

**Arguments against:**
- In synthetic spectra, noise is added artificially and controlled — weak absorption at $F = 0.97$ is a real physical signal, not noise
- Hard thresholding introduces discontinuities that create high-frequency artifacts in both DCT and wavelet coefficients — the sharp on/off boundary is itself a spurious feature
- The statistical distribution of weak absorption carries cosmological information (void statistics, UV background fluctuations)
- Soft thresholding or simply keeping all values is safer for an agnostic ML pipeline where you don't yet know what the classifier will use

**Recommendation:** Do not threshold for now. Keep all values of $A = 1 - F$. If later analysis shows that weak-absorption pixels add noise to classifier performance, apply **soft thresholding** (reduce values below a threshold toward zero gradually) rather than hard zeroing, to avoid spectral artifacts.

---

## Pipeline Architecture

```
Raw synthetic spectra (N spectra × 2048 pixels, flux normalized)
        │
        ▼
[1] Preprocessing
        │  · Compute A = 1 - F
        │  · Global standardization (zero mean, unit variance across dataset)
        │    or per-spectrum standardization depending on task
        │
        ▼
[2a] Windowed DCT-II Branch          [2b] Wavelet Branch
        │                                    │
        ▼                                    ▼
[3a] Feature Matrix                  [3b] Feature Matrix
        │                                    │
        ▼                                    ▼
[4] Post-processing (shared)
        │  · Coefficient selection / truncation
        │  · Optional: PCA for dimensionality reduction
        │  · L2 normalization of feature vectors
        │
        ▼
[5] Feature vectors → Classifier
```

---

## Branch 2a: Windowed DCT-II

### Concept

Rather than applying DCT-II to all 2048 pixels globally, divide the spectrum into overlapping windows and apply DCT-II to each. This gives **joint position-frequency information** — you know both what frequency content is present and roughly where in the spectrum it appears. This is the same principle as MFCCs in speech processing.

### Exact Configuration

**Window function:** Hann window. Before applying DCT-II to each segment, multiply by a Hann window:

$$w(n) = 0.5 \left(1 - \cos\frac{2\pi n}{L-1}\right)$$

This tapers the segment smoothly to zero at the edges, suppressing the spectral leakage and Gibbs ringing that would otherwise be caused by the sharp boundaries of each window. Without this, DCT-II of a raw rectangular segment will produce spurious high-frequency coefficients at every window edge.

**Window length $L$:** 256 pixels. At 2048 pixels total this gives a reasonable trade-off:
- Long enough to resolve velocity structures down to ~few hundred km/s (depending on pixel scale)
- Short enough to provide spatial localization across the spectrum
- Power of 2, which is efficient for DCT computation

**Hop size (step between windows):** 128 pixels (50% overlap). Overlapping windows ensure that absorption features near window boundaries are well-captured rather than split between two windows with attenuated edge weights.

**Number of windows:** With $L = 256$, hop $= 128$, and 2048 pixels:

$$n_{\text{windows}} = \left\lfloor \frac{2048 - 256}{128} \right\rfloor + 1 = 15$$

**Coefficients per window:** Keep the first 32 DCT-II coefficients out of 256. The higher coefficients capture sub-pixel oscillations that are noise-dominated in any realistic setting. This is analogous to keeping 13 MFCCs in speech — the exact cutoff is a hyperparameter to tune, but 32/256 (~12%) is a reasonable starting point.

**Feature vector size:** $15 \times 32 = 480$ features per spectrum.

**DCT-II formula applied per window:**

$$y_k = \sum_{n=0}^{L-1} w(n) \cdot A(n) \cdot \cos\!\left(\frac{\pi k (2n+1)}{2L}\right), \quad k = 0, 1, \ldots, 31$$

with `norm='ortho'` so all coefficients are on the same scale.

### Expected Behavior

The feature matrix for a single spectrum will be a $15 \times 32$ array. Visualized as a heatmap (windows on x-axis, frequency index on y-axis):

- The **top rows** (low $k$) capture slowly varying absorption — broad troughs, DLA wings
- The **lower rows** (high $k$) capture narrow features — individual Lyman-alpha lines, sharp metal line systems
- **Columns** trace how these structures evolve along the spectrum (position axis)
- A **DLA** will appear as a broad high-amplitude smear in the low-$k$ rows, localized to the windows covering its position
- A **dense Lyman-alpha forest region** will show elevated power across many $k$ values in the corresponding windows
- A **void** (no absorption, $A \approx 0$) will appear as near-zero across all $k$ for those windows

For a classifier, the flattened 480-dimensional vector carries both spectral and spatial information. Two spectra with the same global power spectrum but different spatial arrangement of absorbers will have different windowed DCT feature vectors — an advantage over global DCT-II.

---

## Branch 2b: Wavelet Decomposition

### Concept

The Discrete Wavelet Transform (DWT) decomposes the signal into **approximation coefficients** (low-frequency, coarse structure) and **detail coefficients** (high-frequency, fine structure) at multiple scales simultaneously. It is naturally suited to signals with localized features of varying widths — exactly what absorption spectra are.

### Exact Configuration

**Wavelet family:** Daubechies 8 (`db8`). The choice matters:

- `db4` or `db8` are standard for spectroscopic signals — they have compact support (localized in space), vanishing moments (they are blind to smooth polynomial trends, suppressing continuum contributions), and good frequency localization
- `db8` has 8 vanishing moments, meaning it will not respond to smooth continuum variations up to 7th-order polynomials — desirable because residual continuum shape is not the signal of interest
- Haar (`db1`) is too blocky and produces many edge artifacts
- Higher orders like `db16` improve frequency localization but at the cost of spatial localization and longer filter support

**Decomposition depth:** 6 levels. With 2048 pixels and 6 levels:

| Level | Scale (pixels) | Physical Interpretation |
|---|---|---|
| Detail 1 | 1–2 | Sub-resolution noise |
| Detail 2 | 2–4 | Narrowest absorption lines |
| Detail 3 | 4–8 | Typical Lyman-alpha line widths |
| Detail 4 | 8–16 | Broad lines, blended systems |
| Detail 5 | 16–32 | Large-scale clustering, DLA wings |
| Detail 6 | 32–64 | Large-scale structure |
| Approximation 6 | 64+ | Continuum shape, mean flux |

The approximation coefficients at level 6 capture the slowly varying continuum — these may be less informative for classifiers that have already seen continuum-normalized spectra, but should be retained until you know the task.

**Boundary handling:** `mode='periodization'` — this minimizes coefficient count at each level (output length equals input length divided by 2 at each level) and avoids edge padding artifacts. At 2048 pixels with periodization, coefficient counts are exactly: 1024, 512, 256, 128, 64, 32, 32 (for D1 through D6 and A6), totaling 2048 — the DWT is a perfect orthogonal decomposition.

**Feature vector construction:** Concatenate all detail coefficients and the approximation coefficients:

$$\mathbf{f} = [D_1, D_2, D_3, D_4, D_5, D_6, A_6]$$

This gives a 2048-dimensional vector — same size as the input. The DWT is an orthogonal transform so no information is lost at this stage.

**Dimensionality reduction options** (choose based on downstream task):

- Keep only $D_3$ through $D_6$ and $A_6$ (drop $D_1$ and $D_2$ as noise-dominated): reduces to 1024 features
- Apply energy thresholding per level: keep only coefficients whose squared magnitude exceeds a threshold (sparse representation)
- Apply PCA across the dataset to find the most discriminative directions

### Expected Behavior

Visualized as a multi-resolution plot (scalogram):

- **$D_1$, $D_2$:** dominated by noise — flat, near-Gaussian distribution across the spectrum
- **$D_3$, $D_4$:** individual absorption lines appear as localized spikes at their positions. The height of each spike is proportional to the line's equivalent width. A spectrum with dense Lyman-alpha forest will show many spikes distributed across the spectrum; a spectrum with a void will show near-zero coefficients in the corresponding region
- **$D_5$, $D_6$:** broad features — DLA wings produce large-amplitude, spatially extended coefficients. Redshift-space clustering of absorbers produces correlated groups of elevated coefficients
- **$A_6$:** smooth, slowly varying — captures the mean absorption level and any residual large-scale continuum shape

A **DLA** will be most visible as a large spike in $D_5$ and $D_6$ at its position, plus a characteristic pattern of suppressed $D_3$/$D_4$ coefficients in the saturated core region ($F \approx 0$, $A \approx 1$) flanked by elevated coefficients at the damping wing edges.

Two spectra from **different thermal histories** will differ primarily in the distribution of $D_3$/$D_4$ coefficient amplitudes — hotter IGM produces broader lines (larger $b$-parameters), shifting power from $D_3$ to $D_4$.

---

## Step 4: Post-processing (Shared)

**Coefficient selection:** (ignore for now)
- Windowed DCT-II: already truncated to 32 per window; tune this cutoff via cross-validation
- Wavelet: drop $D_1$ and $D_2$ as default; tune depth inclusion

**Optional PCA:** (ignore for now)
Apply PCA fitted on the training set only (never the test set) to reduce to 50–200 principal components. This both reduces dimensionality and removes correlated directions that carry no discriminative information.

**Normalization:** Apply L2 normalization (scale each feature vector to unit norm) before feeding to classifiers. This is important for distance-based classifiers (SVM, k-NN) and gradient-based models. Tree-based models (Random Forest, XGBoost) are scale-invariant and do not require this.

---

## Step 5: Micro Classifiers

We are going to create n classifiers where n is the number of **samples**. Each classifier will be trained with 4 samples of index i from each class. 

### Classifier Considerations

**Random Forest / Gradient Boosting (XGBoost, LightGBM):** (ignore for now)
Good baseline for both feature sets. Scale-invariant, handles correlated features, provides feature importance scores that can guide coefficient selection.

**SVM with RBF kernel:** Works well with L2-normalized wavelet or DCT features. The orthogonality of both transforms means features are already partially decorrelated, which benefits the RBF kernel's distance computations.

**Neural networks:** (ignore for now) If you have enough spectra (tens of thousands), a small MLP on the concatenated or PCA-reduced features is worth trying. At that point you might also consider bypassing the hand-crafted transforms entirely and using a 1D CNN directly on $A = 1 - F$ — the CNN will learn its own localized filters, which may outperform fixed DCT/wavelet bases.

Use SVM with RBF for now. 

---


## Step 6: Classifier Clustering

We are going to cluster the **n micro classifiers** from Step 5 using coefficients of the classifiers. The number of clusters k is not decided yet. 


---


## Experimental Comparison Strategy

Run both branches through the same classifier and compare:

1. **Classification accuracy / AUC** — which feature set is more discriminative?
2. **Feature importance** — which DCT windows / wavelet levels carry the most signal?
3. **Sensitivity to noise** — add varying levels of Gaussian noise to spectra and see how classification degrades for each feature set
4. **Interpretability** — can you map important features back to physical scales?

The wavelet branch is likely to outperform windowed DCT-II if the discriminative signal is **localized** (individual lines, DLAs), while windowed DCT-II may be competitive if the discriminative signal is more **global** (mean flux level, large-scale power spectrum shape). Running both and comparing is the cleanest way to determine which is better matched to your eventual task.

---

## Summary Table

| Property | Windowed DCT-II | Wavelet (db8, 6 levels) |
|---|---|---|
| Feature vector size | 480 (15 windows × 32 coeffs) | 2048 → 1024 (drop D1, D2) |
| Position sensitivity | Moderate (window-level) | Fine-grained (per-pixel at each scale) |
| Frequency resolution | Uniform across all windows | Multi-resolution (coarse to fine) |
| Best for | Global spectral structure, mean flux | Localized features, individual lines, DLAs |
| Noise sensitivity | Lower (windowed averaging) | Higher at fine scales (D1, D2) |
| Interpretability | Window position + frequency index | Physical scale per level |
| Hyperparameters to tune | Window size, hop, n_coeffs | Wavelet family, depth, level inclusion |
