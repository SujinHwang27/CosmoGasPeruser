import matplotlib.pyplot as plt
import numpy as np
import os

def plot_fisher(fisher_scores_matrix, title, save_path):
    """
    Plots all rows of a Fisher scores matrix as lines in a single figure.
    
    Args:
        fisher_scores_matrix (np.ndarray): shape (n_samples, n_features)
        title (str): plot title
        save_path (str): folder to save the figure
    Returns:
        out_path (str): path to saved figure
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))

    n_samples, n_features = fisher_scores_matrix.shape

    # Plot each row as a line
    for i in range(n_samples):
        plt.plot(range(n_features), fisher_scores_matrix[i], alpha=0.6, linewidth=1)

    plt.xlabel("Feature Index")
    plt.ylabel("Fisher Coefficient")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

    out_path = os.path.join(save_path, f"{title}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path

def plot_mean_fisher(mean_fisher, title="Mean Fisher Coefficient", save_path="plots"):
    """
    Plots the mean Fisher coefficient per feature as a line graph.

    Args:
        mean_fisher (np.ndarray): 1D array of mean Fisher coefficients per feature
        title (str): Title of the plot
        save_path (str): Folder to save the figure

    Returns:
        out_path (str): Path to the saved figure
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(mean_fisher)), mean_fisher, marker='o', linewidth=2, color='blue')
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Fisher Coefficient")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path
