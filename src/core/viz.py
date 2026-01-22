import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List

def plot_training_curves(history: Dict[str, List[float]], save_path: str = "training_plot.png"):
    """
    Plots training and validation loss/accuracy.
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot if present
    if 'val_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_feature_scores(scores: np.ndarray, feature_names: List[str] = None, save_path: str = "feature_scores.png"):
    """
    Plots feature importance scores.
    """
    plt.figure(figsize=(10, 6))
    if feature_names:
        plt.bar(feature_names, scores)
    else:
        plt.bar(range(len(scores)), scores)
    plt.title('Feature Importance Scores')
    plt.xlabel('Feature Index')
    plt.ylabel('Score')
    plt.savefig(save_path)
    plt.close()
