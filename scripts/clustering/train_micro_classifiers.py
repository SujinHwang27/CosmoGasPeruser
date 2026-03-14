import numpy as np
import os
import argparse
from sklearn.svm import SVC
from joblib import Parallel, delayed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def train_single_svm_rbf(X_i, y, C=1.0, gamma='scale'):
    """
    Trains a series of RBF-kernel SVMs for 6 One-Vs-One pairs.
    Returns a 30-dimensional separability (sensitivity) vector:
      - 24 dims: decision function values for the 4 training points
                 across the 6 OvO binary classifiers.
      - 6 dims:  intercept (bias) terms of the 6 binary classifiers.

    This vector encodes *how much* the four physics realizations differ
    at this sightline index — measuring the intensity of contrast rather
    than the physical reason for it.
    """
    n_classes = 4
    all_decision_values = []
    all_intercepts = []

    # 6 pairs for 4 classes: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            idx = [i, j]
            X_pair = X_i[idx]
            y_pair = [0, 1]  # Binary targets for this pair

            clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            clf.fit(X_pair, y_pair)

            # decision_function returns distance from hyperplane for each sample
            # Shape: (2,) for binary SVC on 2 training points
            decision_vals = clf.decision_function(X_pair)  # (2,)
            all_decision_values.append(decision_vals)
            all_intercepts.append(clf.intercept_)  # (1,)

    # Concatenate: 6 * 2 decision values + 6 * 1 intercepts = 12 + 6 ... no,
    # wait — there are 4 training points total evaluated on each pair's classifier.
    # Re-evaluate all 4 points on each classifier for the full 24-dim vector.
    all_decision_values = []
    all_intercepts = []

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            idx = [i, j]
            X_pair = X_i[idx]
            y_pair = [0, 1]

            clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            clf.fit(X_pair, y_pair)

            # Evaluate all 4 class representatives on this binary classifier
            decision_vals = clf.decision_function(X_i)  # (4,)
            all_decision_values.append(decision_vals)
            all_intercepts.append(clf.intercept_)  # (1,)

    # 6 * 4 = 24 decision values + 6 intercepts = 30-dim vector
    return np.concatenate([
        np.concatenate(all_decision_values),   # (24,)
        np.concatenate(all_intercepts)         # (6,)
    ])


def train_all_micro_classifiers(data_dir, output_file, n_jobs=-1, C=1.0, gamma='scale'):
    """
    Trains 16384 micro-classifiers (RBF SVM) and saves their 30-dim
    separability vectors to a .npy file.

    Output shape: (n_probes, 30)
      - 24 dims: OvO decision function values for all 4 training points
                 across 6 binary classifiers
      - 6 dims:  intercepts of the 6 binary classifiers
    """
    print(f"Loading data from {data_dir}...")
    X_classes = []
    for c in range(1, 5):
        path = os.path.join(data_dir, str(c), "data.npy")
        X_classes.append(np.load(path, mmap_mode='r'))

    n_probes = X_classes[0].shape[0]
    print(f"Training {n_probes} micro-classifiers using RBF SVM (C={C}, gamma={gamma})...")

    y = np.array([0, 1, 2, 3])

    def worker(idx):
        X_idx = np.stack([X_classes[c][idx] for c in range(4)])  # (4, n_features)
        return train_single_svm_rbf(X_idx, y, C=C, gamma=gamma)

    # Use joblib to parallelize
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(i) for i in tqdm(range(n_probes))
    )

    results_array = np.array(results)
    print(f"Training complete. Separability vector shape: {results_array.shape}")
    assert results_array.shape == (n_probes, 30), \
        f"Expected shape ({n_probes}, 30), got {results_array.shape}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, results_array)
    print(f"Saved separability vectors to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RBF micro-classifiers and extract 30-dim separability vectors."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with class folders 1-4, each containing data.npy")
    parser.add_argument("--output", type=str,
                        default="data/feature_discovery/base_data/micro_classifier_params.npy",
                        help="Output .npy file for separability vectors")
    parser.add_argument("--C", type=float, default=1.0,
                        help="SVM regularization parameter")
    parser.add_argument("--gamma", type=str, default="scale",
                        help="RBF kernel bandwidth ('scale', 'auto', or float)")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 = all cores)")

    args = parser.parse_args()

    train_all_micro_classifiers(
        args.data_dir, args.output,
        n_jobs=args.jobs, C=args.C, gamma=args.gamma
    )
