import matplotlib.pyplot as plt


def plot_2d_data(X_2d, y, title):
    """
    Plots 2D PCA data with points colored by class.
    Returns the path to the saved plot.
    """
    out_path = f"{title}.png"
    plt.figure(figsize=(8, 6))
    colors = ["r", "g", "b", "m"]
    for i, color in enumerate(colors, start=1):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=color, label=f"Class {i}", alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path