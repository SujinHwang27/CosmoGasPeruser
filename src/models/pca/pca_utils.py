import sys
import numpy as np
import os


def uncentered_pca(data, n_components=2):
    """
    Performs a simple PCA to reduce dimensionality to top k principal components.

    The input is a comma-separated matrix (npy file) of size n x m.
    The output is a comma-separated matrix (npy file) of size n x k,
    representing the data projected onto the first k principal components.

    Args:
        data (numpy matrix): X (without label)

    """
    try:
        # 1. Load the data
        # already done through 'data' argument

        # 2. Compute Gram Matrix (no centering)
        gram_matrix = np.dot(data.T, data)

        # 3. Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

        # 4. Select the Top k Eigenvectors (Principal Components)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_indices = sorted_indices[:n_components]
        principal_components = eigenvectors[:, top_indices]


        # 5. Project the Data onto the 2 Principal Components
        data_projected = np.dot(data, principal_components)


        return data_projected

    except Exception as e:
        print(f"PCA computation failed: {e}")
        raise



def uncentered_pca_last(data, n_components=2):
    """
    Performs a simple PCA to reduce dimensionality to last k principal components.

    The input is a comma-separated matrix (npy file) of size n x m.
    The output is a comma-separated matrix (npy file) of size n x k,
    representing the data projected onto the last k principal components.

    Args:
        data (numpy matrix): X (without label)

    """
    try:
        # 1. Load the data
        # already done through 'data' argument

        # 2. Compute Gram Matrix (no centering)
        gram_matrix = np.dot(data.T, data)

        # 3. Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

        # 4. Select the Last k Eigenvectors (Principal Components)
        sorted_indices = np.argsort(eigenvalues)[:n_components]
        principal_components = eigenvectors[:, sorted_indices]


        # 5. Project the Data onto the 2 Principal Components
        data_projected = np.dot(data, principal_components)


        return data_projected

    except Exception as e:
        print(f"PCA computation failed: {e}")
        raise

        

def save_reduced_data(X_reduced, y, out_base):
    """
    Saves reduced 2D data by class
    """
    for i in np.unique(y):
        class_data = X_reduced[y == i]
        out_dir = os.path.join(out_base, str(i))
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "flux.npy")
        np.save(save_path, class_data)
        print(f"Saved {len(class_data)} samples to {save_path}")