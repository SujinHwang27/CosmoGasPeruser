import matplotlib.pyplot as plt
import plotly.express as px

import numpy as np
import pandas as pd
import os



def plot_2d_data(X_2d, y, title, save_path):
    """
    Plots 2D DCT-reduced data with points colored by class.
    Returns the path to the saved plot.
    """
    out_path = f"{save_path}/{title}.png"
    plt.figure(figsize=(8, 6))
    colors = ["r", "g", "b", "m"]
    for i, color in enumerate(colors, start=1):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=color, label=f"Class {i}", alpha=0.6)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"{title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_dct_by_los(dct_data, y, title, save_path, n_classes=4, n_samples_per_class=None):
    """
    Plots DCT coefficients of corresponding data points across classes.
    
    Args:
        dct_data (np.ndarray): DCT-transformed data of shape (n, m),
                               where rows are vertically concatenated from multiple classes.
        n_classes (int): Number of classes (default = 4)
        n_samples_per_class (int, optional): Number of data points per class.
                                             If None, inferred as n / n_classes.
    """
    try:
        n, m = dct_data.shape
        if n_samples_per_class is None:
            n_samples_per_class = n // n_classes

        if n_samples_per_class * n_classes != n:
            raise ValueError("Data size not evenly divisible by number of classes.")

        # Split by class (assuming vertical stacking)
        class_blocks = np.vsplit(dct_data, n_classes)

        # Fixed class color palette (consistent across plots)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        out_path_list = []

        # Loop through sample indices within each class
        for i in range(n_samples_per_class):
            plt.figure(figsize=(8, 5))
            for c in range(n_classes):
                plt.plot(
                    class_blocks[c][i, :],
                    label=f"Class {c+1}",
                    color=colors[c],
                    marker='o',
                    linewidth=1.3
                )

            plt.title(f"{title} — Sample #{i+1} Across Classes")
            plt.xlabel("DCT Coefficient Index (Frequency)")
            plt.ylabel("Coefficient Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            out_path = f"{save_path}/{title}_sample{i+1}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()

            out_path_list.append(out_path)


        print(f"Saved {n_samples_per_class} plots as '{title}_sample#.png'")
        return out_path_list

    except Exception as e:
        print(f"Plotting failed: {e}")
        raise


def plot_3d_interactive(X_3d, y, title, save_path):
    """
    Creates and saves an interactive 3D plot as an HTML file.

    Args:
        X_3d (np.ndarray): n x 3 array of data points
        y (np.ndarray): n-length label vector (1-based or 0-based class indices)
        title (str): title for the plot
        save_path (str): folder to save the HTML plot

    Returns:
        str: path to the saved interactive plot
    """
    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    # Convert to DataFrame for Plotly
    df = pd.DataFrame(X_3d, columns=['Comp1', 'Comp2', 'Comp3'])
    df['Class'] = y

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='Comp1',
        y='Comp2',
        z='Comp3',
        color='Class',
        symbol='Class',
        title=title,
        labels={'Comp1':'Component 1', 'Comp2':'Component 2', 'Comp3':'Component 3'},
        opacity=0.7
    )

    # Save as interactive HTML
    out_path = os.path.join(save_path, f"{title}_interactive.html")
    fig.write_html(out_path)

    return out_path


def plot_dct_1d_heatmap(dct_coeffs, title="DCT Coefficients", save_path=None):
    """
    Visualizes a 1D array of DCT coefficients as a horizontal heatmap.
    """
    dct_coeffs = np.array(dct_coeffs).reshape(1, -1)  # reshape for imshow
    
    plt.figure(figsize=(8, 1.2))
    plt.imshow(dct_coeffs, cmap='coolwarm', aspect='auto')
    plt.colorbar(label="Coefficient Value", orientation="vertical", shrink=0.8)
    plt.yticks([])
    plt.xticks(range(len(dct_coeffs[0])), labels=[f"{i}" for i in range(len(dct_coeffs[0]))])
    plt.title(title)
    plt.xlabel("Coefficient Index")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
