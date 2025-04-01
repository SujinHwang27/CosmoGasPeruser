from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import sys
import psutil
import logging
from src.data_loader import load_data, prepare_dataset
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt



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

def plot_velocity_importance(velocity_data, importance_scores, threshold=0.005):
    """Plot velocity values as x-axis with points colored by their importance scores."""
    plt.figure(figsize=(25, 1))
    
    # Apply custom normalization
    norm_importance = custom_normalize(importance_scores, threshold)
    
    # Define colors (Blue for low importance, Red for high importance)
    colors = np.array([[score, 0, 1 - score] for score in norm_importance])
    
    # Scatter plot
    plt.scatter(velocity_data, np.zeros_like(velocity_data), c=colors, s=50, edgecolors='k', alpha=0.7)

    # top_indices = np.argsort(importance_scores)[-15:]  # Get indices of top 15 importance values
    
    # # Annotate the points with the highest importance scores
    # for idx in top_indices:
    #     plt.text(velocity_data[idx], 0.02, f"{velocity_data[idx]:.2f}", 
    #              ha='center', fontsize=10, color='black')
    
    # Labels and formatting
    plt.xlabel("Velocity Data")
    plt.yticks([])  # Hide y-axis since it's not relevant
    plt.title("Velocity Importance Visualization")
    
    # Show plot
    plt.show()



def main():
    data = load_data()
    spectra, labels = prepare_dataset(data)
    
    # 2. Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, 
        labels, 
        test_size=0.2,  
        random_state=42,  
        stratify=labels  # maintain class distribution
    )
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
        
    # Measure memory before training
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Measure memory after training
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    print(f"Memory Usage Before Training: {mem_before:.2f} MB")
    print(f"Memory Usage After Training: {mem_after:.2f} MB")
    print(f"Memory Increase: {mem_after - mem_before:.2f} MB")

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Get feature importances
    importances = rf.feature_importances_
    
    # Load velocity data
    velocity_path = "/home/sujin/CosmoGasPeruser/data/TargetedSpecML/no_feedback/SN40.0_nspec3000_seed10/velocity.npy"
    velocity_data = np.load(velocity_path)
    
    # Print important features
    logger.info("Top 15 important features:")
    sorted_indices = np.argsort(importances)[::-1]
    for i in sorted_indices[:15]:
        logger.info(f"Velocity {velocity_data[i]}: {importances[i]:.4f}")

    # Create velocity importance plot
    plot_velocity_importance(velocity_data, importances, threshold=0.005)
    plt.show()

    # # Calculate and print permutation importance
    # perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    # sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    # print("\nPermutation Importance - Top 10 features:")
    # for i in sorted_indices[:10]:
    #     print(f"Feature {i}: {perm_importance.importances_mean[i]:.4f}")

# Accuracy: 0.6925
# INFO:__main__:Feature 73: 0.0083
# INFO:__main__:Feature 72: 0.0080
# INFO:__main__:Feature 124: 0.0077
# INFO:__main__:Feature 71: 0.0071
# INFO:__main__:Feature 69: 0.0069
# INFO:__main__:Feature 122: 0.0068
# INFO:__main__:Feature 132: 0.0068
# INFO:__main__:Feature 75: 0.0067
# INFO:__main__:Feature 189: 0.0067
# INFO:__main__:Feature 76: 0.0065



# import os
# import yaml
# import pandas as pd
# from src.model import RandomForestModel
# from src.utils.data_processor import DataProcessor

# def load_config(config_path: str) -> dict:
#     """Load configuration from YAML file"""
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# def setup_directories(config: dict):
#     """Create necessary directories if they don't exist"""
#     for dir_path in config['paths'].values():
#         os.makedirs(dir_path, exist_ok=True)

# def main():
#     # Load configuration
#     config = load_config('config/config.yaml')
    
#     # Setup directories
#     setup_directories(config)
    
#     # Initialize data processor
#     data_processor = DataProcessor()
    
#     # Load and preprocess data
#     data_path = os.path.join(config['paths']['data_dir'], 'data.csv')
#     data = pd.read_csv(data_path)
    
#     # Preprocess data
#     X, y = data_processor.preprocess_data(
#         data=data,
#         target_column=config['data']['target_column'],
#         categorical_columns=config['data']['categorical_columns'],
#         numerical_columns=config['data']['numerical_columns']
#     )
    
#     # Split data
#     X_train, X_test, y_train, y_test = data_processor.split_data(
#         X, y,
#         test_size=config['data']['test_size'],
#         random_state=config['data']['random_state']
#     )
    
#     # Initialize and train model
#     model = RandomForestModel(
#         task_type=config['model']['task_type'],
#         **{k: v for k, v in config['model'].items() if k != 'task_type'}
#     )
    
#     # Train model
#     model.train(X_train, y_train)
    
#     # Evaluate model
#     results = model.evaluate(X_test, y_test)
    
#     # Save model
#     model_path = os.path.join(config['paths']['model_dir'], 'model.joblib')
#     model.save_model(model_path)
    
#     # Save results
#     results_path = os.path.join(config['paths']['results_dir'], 'results.txt')
#     with open(results_path, 'w') as f:
#         f.write(str(results))
    
#     print("Training completed successfully!")
#     print(f"Model saved to: {model_path}")
#     print(f"Results saved to: {results_path}")
#     print("\nResults:")
#     print(results)

if __name__ == "__main__":
    main() 