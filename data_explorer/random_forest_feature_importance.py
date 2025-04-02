import logging
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
# %matplotlib inline




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_normalize(values, threshold):
    """Normalize values so that threshold maps to 0.5."""
    values = np.array(values)
    min_val, max_val = np.min(values), np.max(values)
    
    # Split normalization into below and above threshold
    below_mask = values < threshold
    above_mask = values > threshold

    normalized_values = np.zeros_like(values, dtype=float)
    
    if np.any(below_mask):  # Avoid division by zero
        normalized_values[below_mask] = 0.5 * (values[below_mask] - min_val) / (threshold - min_val + 1e-9)
    if np.any(above_mask):  # Avoid division by zero
        normalized_values[above_mask] = 0.5 + 0.5 * (values[above_mask] - threshold) / (max_val - threshold + 1e-9)

    normalized_values[values == threshold] = 0.5  # Explicitly set threshold point

    return normalized_values

def plot_feature_importance(features, importance_scores, threshold=0.005):
    """Plot velocity values as x-axis with points colored by their importance scores."""
    plt.figure(figsize=(25, 1))
    
    # Apply custom normalization
    norm_importance = custom_normalize(importance_scores, threshold)
    
    # Define colors (Blue for low importance, Red for high importance)
    colors = np.array([[score, 0, 1 - score] for score in norm_importance])
    
    # Scatter plot
    plt.scatter(features, np.zeros_like(features), c=colors, s=50, edgecolors='k', alpha=0.7)

    # top_indices = np.argsort(importance_scores)[-15:]  # Get indices of top 15 importance values
    
    # # Annotate the points with the highest importance scores
    # for idx in top_indices:
    #     plt.text(velocity_data[idx], 0.02, f"{velocity_data[idx]:.2f}", 
    #              ha='center', fontsize=10, color='black')
    
    # Labels and formatting
    plt.xlabel("Velocity(or wavelength)")
    plt.yticks([])  # Hide y-axis since it's not relevant
    plt.title("Feature Importance Visualization")
    
    # Show plot
    plt.show()


    # # Get feature importances
    # importances = rf.feature_importances_
    
    # # Load velocity data
    # # velocity_path = "/home/sujin/CosmoGasPeruser/data/TargetedSpecML/no_feedback/SN40.0_nspec3000_seed10/velocity.npy"
    # wavelength = load_wavelength()
    
    # # Print important features
    # logger.info("Top 15 important features:")
    # sorted_indices = np.argsort(importances)[::-1]
    # for i in sorted_indices[:15]:
    #     logger.info(f"Wavelength {wavelength[i]}: {importances[i]:.4f}")

    # # Create velocity importance plot
    # plot_feature_importance(wavelength, importances, threshold=0.0009)
    # plt.show()

    # # Calculate and print permutation importance
    # perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    # sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    # print("\nPermutation Importance - Top 10 features:")
    # for i in sorted_indices[:10]:
    #     print(f"Feature {i}: {perm_importance.importances_mean[i]:.4f}")
