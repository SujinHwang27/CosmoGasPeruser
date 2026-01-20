import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(base_path=None):
    """
    Loads flux.npy files from Sherwood dataset.
    Returns:
        X: list of arrays (concatenated data)
        y: list of class labels
    """
    X_list = []
    y_list = []
    for i in range(1, 5):  # Classes 1 to 4 mapped into 0 to 3
        file_path = os.path.join(base_path, str(i), "data.npy")
        flux = np.load(file_path)
        X_list.append(flux)
        y_list.append(np.full(len(flux), i-1))  # class label
    X = np.vstack(X_list)
    X = X[:, :20]
    y = np.concatenate(y_list)


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/val/test: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

