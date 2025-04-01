import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_importance(velocity_data, importance_scores, threshold=0.005):
    """
    Plot velocity values colored by their importance scores.
    
    Args:
        velocity_data (np.array): Array of velocity values
        importance_scores (np.array): Array of importance scores
        threshold (float): Threshold for importance scores to be highlighted
    """
    plt.figure(figsize=(15, 6))
    
    # Create color array based on importance scores
    colors = np.zeros((len(velocity_data), 3))
    for i in range(len(velocity_data)):
        if importance_scores[i] > threshold:
            # Use a color gradient from blue to red based on importance
            colors[i] = [importance_scores[i], 0, 1 - importance_scores[i]]
        else:
            colors[i] = [0.7, 0.7, 0.7]  # Gray for low importance
    
    # Plot velocity values
    plt.plot(velocity_data, color='gray', alpha=0.3, label='Low importance')
    
    # Plot highlighted points
    for i in range(len(velocity_data)):
        if importance_scores[i] > threshold:
            plt.scatter(i, velocity_data[i], 
                       c=[colors[i]], 
                       s=100,
                       alpha=0.7)
    
    plt.xlabel('Index')
    plt.ylabel('Velocity')
    plt.title('Velocity Values Colored by Feature Importance')
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r), 
                label='Importance Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf() 