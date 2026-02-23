import numpy as np
import os
import argparse
from sklearn.svm import SVC
from joblib import Parallel, delayed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def train_single_svm(X_i, y, kernel='rbf', C=1.0, gamma='scale'):
    """
    Trains an SVM on 4 samples and returns a flattened representation of its parameters.
    """
    clf = SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape='ovo')
    clf.fit(X_i, y)
    
    # Extract "coefficients" for clustering.
    # For OvO 4-class, we have 6 binary classifiers.
    # We'll use the dual coefficients and the intercepts.
    
    # dual_coef_ has shape (n_classes - 1, n_SV)
    # For 4 classes, it's (3, n_SV). 
    # This is a bit complex to parse because it's a concatenated representation.
    
    # Alternative: Use the decision function values on the 4 training points themselves.
    # This captures the 'behavior' of the classifier on the signals.
    # shape: (4 samples, 6 pairs) -> 24 values.
    # This is fixed size and perfectly describes the state of the classifier.
    behavior = clf.decision_function(X_i) # Shape (4, 6)
    
    # We can also add intercepts
    intercepts = clf.intercept_ # Shape (6,)
    
    return np.concatenate([behavior.flatten(), intercepts])

def train_all_micro_classifiers(data_dir, output_file, n_jobs=-1, kernel='rbf'):
    """
    Trains 16384 micro-classifiers and saves their behavioral vectors for clustering.
    """
    print(f"Loading data from {data_dir}...")
    X_classes = []
    for c in range(1, 5):
        path = os.path.join(data_dir, str(c), "data.npy")
        X_classes.append(np.load(path, mmap_mode='r'))
    
    n_probes = X_classes[0].shape[0]
    print(f"Training {n_probes} micro-classifiers using {kernel} kernel...")
    
    y = np.array([0, 1, 2, 3])
    
    def worker(i):
        X_i = np.stack([X_classes[c][i] for c in range(4)])
        return train_single_svm(X_i, y, kernel=kernel)

    # Use joblib to parallelize
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(i) for i in tqdm(range(n_probes))
    )
    
    results_array = np.array(results)
    print(f"Training complete. Feature vector shape for clustering: {results_array.shape}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, results_array)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with class folders 1-4")
    parser.add_argument("--output", type=str, default="data/feature_discovery/micro_classifier_params.npy", help="Output file")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    train_all_micro_classifiers(args.data_dir, args.output, n_jobs=args.jobs, kernel=args.kernel)
