from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import logging
from src.data_loader import load_data, prepare_dataset
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")



    # Sort and visualize top features
    importances = rf.feature_importances_
    feature_names = np.array(["Feature " + str(i) for i in range(spectra.shape[1])])

    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    for i in sorted_indices[:10]:  # Show top 10 features
        logger.info(f"{feature_names[i]}: {importances[i]:.4f}")


    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]

    for i in sorted_indices[:10]:
        print(f"{feature_names[i]}: {perm_importance.importances_mean[i]:.4f}")


    

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_indices[:10]], importances[sorted_indices[:10]], color='royalblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title("Top 10 Most Important Features (Random Forest)")
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.show()

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