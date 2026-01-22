import numpy as np
import os


def load_sherwood_data(base_path):
    """
    Loads flux.npy files from Sherwood dataset.
    Returns:
        X: list of arrays (concatenated data)
        y: list of class labels
    """
    X_list = []
    y_list = []
    for i in range(1, 5):  # Classes 1 to 4
        file_path = os.path.join(base_path, str(i), "data.npy")
        flux = np.load(file_path)
        X_list.append(flux)
        y_list.append(np.full(len(flux), i))  # class label
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y