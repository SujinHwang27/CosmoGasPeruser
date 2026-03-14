"""
Micro-probing module for signal clustering analysis.

Uses RBF SVM to compute separability vectors (24-dim decision distances)
for each sightline across 4 physics classes.
"""

import numpy as np
from sklearn.svm import SVC
from typing import List
from joblib import Parallel, delayed


# All one-vs-one class pairs (6 pairs for 4 classes)
OVO_PAIRS = [
    (0, 1),  # Class 1 vs Class 2
    (0, 2),  # Class 1 vs Class 3
    (0, 3),  # Class 1 vs Class 4
    (1, 2),  # Class 2 vs Class 3
    (1, 3),  # Class 2 vs Class 4
    (2, 3),  # Class 3 vs Class 4
]


def probe_sightline(s: int, X_per_class: List[np.ndarray]) -> np.ndarray:
    """
    Compute the 24-dimensional separability vector for a single sightline.

    For each of the 6 one-vs-one class pairs, fit an RBF SVM and record
    the signed decision distances from all 4 classes to the hyperplane.

    Args:
        s: Sightline index to probe
        X_per_class: List of 4 arrays, each (n_sightlines, n_features)

    Returns:
        np.ndarray shape (24,) - signed decision distances
    """
    # Extract the feature vectors for all 4 classes at sightline s
    # Each is a 1D feature vector of shape (n_features,)
    class_vectors = np.array([X_per_class[c][s] for c in range(4)])

    separability_vector = []

    for (p, q) in OVO_PAIRS:
        # Get the two class vectors for this pair
        X_pair = np.vstack([class_vectors[p], class_vectors[q]])
        y_pair = np.array([0, 1])

        # Fit RBF SVM
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(X_pair, y_pair)

        # Get decision distances for all 4 classes
        distances = svm.decision_function(class_vectors)
        separability_vector.extend(distances)

    return np.array(separability_vector, dtype=np.float64)


def run_probe(X_per_class: List[np.ndarray], n_jobs: int = -1) -> np.ndarray:
    """
    Run micro-probing on all sightlines.

    Args:
        X_per_class: List of 4 arrays, each (n_sightlines, n_features)
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        np.ndarray shape (n_sightlines, 24) - separability vectors
    """
    n_sightlines = X_per_class[0].shape[0]
    n_features = X_per_class[0].shape[1]

    print(f"Probing {n_sightlines} sightlines across {n_features} features...")

    # Parallelize over sightlines
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(probe_sightline)(s, X_per_class) for s in range(n_sightlines)
    )

    separability_vectors = np.array(results, dtype=np.float64)
    print(f"Separability vectors shape: {separability_vectors.shape}")

    return separability_vectors


def probe_single_feature_set(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Convenience function for probing from a concatenated feature array.

    Args:
        X: Concatenated feature array, shape (n_total, n_features)
        y: Class labels, shape (n_total,)
        n_jobs: Number of parallel jobs

    Returns:
        np.ndarray shape (n_sightlines, 24)
    """
    # Split into per-class arrays
    X_per_class = []
    for c in range(1, 5):
        X_per_class.append(X[y == c])

    return run_probe(X_per_class, n_jobs=n_jobs)
