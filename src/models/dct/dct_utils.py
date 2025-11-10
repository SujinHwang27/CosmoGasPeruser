import numpy as np
from scipy.fftpack import dct, idct
import os

def dct_high_freq(data, n_coefficients=2):
    """
    Extracts the last k coefficients (high-frequency) from Discrete Cosine Transform
    for each row of the input 2D numpy array.

    Args:
        data (np.ndarray): 2D array of size n x m (n rows, m features)
        n_coefficients (int): number of high-frequency coefficients to extract

    Returns:
        np.ndarray: 2D array of size n x k containing the last k DCT coefficients
    """
    try:

        # Apply DCT along each row
        dct_data = dct(data, type=2, norm='ortho', axis=1)

        # Extract the last n_coefficients (high-frequency)
        high_freq = dct_data[:, -n_coefficients:]

        return high_freq

    except Exception as e:
        print(f"DCT computation failed: {e}")
        raise

def dct_full(data):
    """
    Transforms data into full DCT coefficients
    """
    try:
        # Apply DCT along the row
        dct_data = dct(data, type=2, norm='ortho', axis=1)

        return dct_data

    except Exception as e:
        print(f"DCT computation failed: {e}")
        raise

def dct_dominant(data, n_coefficients=2):
    """
    Extracts the first k coefficients from Discrete Cosine Transform
    for each row of the input 2D numpy array.

    Args:
        data (np.ndarray): 2D array of size n x m (n rows, m features)
        n_coefficients (int): number of dominant coefficients to extract

    Returns:
        np.ndarray: 2D array of size n x k containing the last k DCT coefficients
    """
    try:

        # Apply DCT along each row
        dct_data = dct(data, type=2, norm='ortho', axis=1)

        # Extract the last n_coefficients (high-frequency)
        dominant = dct_data[:, :n_coefficients]

        return dominant

    except Exception as e:
        print(f"DCT computation failed: {e}")
        raise


def save_processed_data(X_reduced, y, out_base):
    """
    Saves processed data by class
    """
    for i in np.unique(y):
        class_data = X_reduced[y == i]
        out_dir = os.path.join(out_base, str(i))
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "data.npy")
        np.save(save_path, class_data)
        print(f"Saved {len(class_data)} samples to {save_path}")

def sort_dct_by_variance(sample_dct_list):
    """
    Sorts DCT coefficient indices of a Line of Sight by variance across classes.

    Args:
        sample_dct_list (list[np.ndarray]): list of DCT coefficient arrays 
                                     of a same Line of Sight(a data point) from each classes. 
    Returns:
        sorted_indices (np.ndarray): indices sorted by variance (descending)
        variances (np.ndarray): variance values corresponding to each coefficient
    """
    D = np.vstack(sample_dct_list)  # shape: (n_classes=4, n_coeffs=2048)
    variances = np.var(D, axis=0)
    sorted_indices = np.argsort(variances)[::-1]
    return sorted_indices, variances

def global_rank_by_variance(X_dct, y):
    """
    Aggregates and ranks coefficients globally based on variance.

    Args:
        X_dct (np.ndarray): X processed by Discrete Cosine Transform
        y (np.ndarray): class label of corresponding X

    Returns:
        global_sorted_idx (np.ndarray): global indices sorted by average variance
        mean_var (np.ndarray): mean variance values across datasets
    """
    # Group by sample index across classes
    unique_classes = np.unique(y)
    class_blocks = np.vsplit(X, len(unique_classes)) # shape (depth=n_classes, row=n_samples_per_class, col=n_features)

    # Stack along a new dimension
    dct_los_stacked = np.stack(class_blocks, axis=1) # shape (depth=n_samples_per_class, row=n_classes, col=n_features)

    all_variances = []

    for dct_los in dct_los_stacked:
        var = np.var(dct_los, axis=0)
        all_variances.append(var)

    all_variances = np.vstack(all_variances)
    mean_var = np.mean(all_variances, axis=0)

    global_sorted_idx = np.argsort(mean_var)[::-1]
    return global_sorted_idx, mean_var

