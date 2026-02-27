import numpy as np
import os
import argparse
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def train_single_svm_l1(X_i, y, C=100.0):
    """
    Trains a series of L1-regularized Linear SVMs for 6 One-Vs-One pairs.
    Returns a sparse 3078-dimensional mechanism vector (6 * 512 weights + 6 intercepts).
    
    A high C (e.g., 100) combined with L1 penalty forces the model to find the 
    most compact set of physical features (wavelet pixels) that achieves 
    perfect separation between realizations.
    """
    n_classes = 4
    all_weights = []
    all_intercepts = []
    
    # 6 pairs for 4 classes (0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Extract only the two classes for this OVO pair
            idx = [i, j]
            X_pair = X_i[idx]
            y_pair = [0, 1] # Binary targets for the pair
            
            # LinearSVC with L1 penalty (requires dual=False)
            # We use a high C to approach a 'Hard Margin' while maintaining L1-sparsity
            clf = LinearSVC(penalty='l1', C=C, dual=False, random_state=42, max_iter=10000)
            clf.fit(X_pair, y_pair)
            
            all_weights.append(clf.coef_.flatten())
            all_intercepts.append(clf.intercept_)
            
    return np.concatenate([np.concatenate(all_weights), np.concatenate(all_intercepts)])

def train_all_micro_classifiers(data_dir, output_file, n_jobs=-1, C=100.0):
    """
    Trains 16384 micro-classifiers and saves their sparse mechanism fingerprints.
    """
    print(f"Loading data from {data_dir}...")
    X_classes = []
    for c in range(1, 5):
        path = os.path.join(data_dir, str(c), "data.npy")
        X_classes.append(np.load(path, mmap_mode='r'))
    
    n_probes = X_classes[0].shape[0]
    print(f"Training {n_probes} micro-classifiers using L1-Linear SVM (Hard Margin C={C})...")
    
    y = np.array([0, 1, 2, 3])
    
    def worker(idx):
        X_idx = np.stack([X_classes[c][idx] for c in range(4)])
        return train_single_svm_l1(X_idx, y, C=C)

    # Use joblib to parallelize
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(i) for i in tqdm(range(n_probes))
    )
    
    results_array = np.array(results)
    print(f"Training complete. Sparse Vector shape for clustering: {results_array.shape}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, results_array)
    print(f"Saved sparse fingerprints to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with class folders 1-4")
    parser.add_argument("--output", type=str, default="data/feature_discovery/base_data/params_l1_mechanism.npy", help="Output file")
    parser.add_argument("--C", type=float, default=100.0, help="Regularization strength (High = Harder Margin)")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    train_all_micro_classifiers(args.data_dir, args.output, n_jobs=args.jobs, C=args.C)
