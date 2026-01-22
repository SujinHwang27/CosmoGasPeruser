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

