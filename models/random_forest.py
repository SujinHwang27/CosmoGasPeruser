import os
import json
import joblib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42):
    """
    Trains a Random Forest classifier and evaluates performance.

    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        n_estimators (int, optional): Number of trees in the forest. Default is 100.
        max_depth (int, optional): Maximum depth of the trees. Default is None.
        random_state (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        dict: Performance metrics including confusion matrix.
    """

    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute performance metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()  # Convert to list for saving
    }

    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # Plot Confusion Matrix
    plot_confusion_matrix(metrics["confusion_matrix"], classes=np.unique(y_train))

    # Save the trained model and metadata
    save_model(model, metrics, {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state})

    return metrics

def plot_confusion_matrix(conf_matrix, classes):
    """
    Plots and saves the confusion matrix as an image.

    Args:
        conf_matrix (list): Confusion matrix as a list.
        classes (list): Unique class labels.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def save_model(model, metrics, hyperparameters):
    """
    Saves the trained model and metadata in 'saved_models/random_forest/'.

    Args:
        model (RandomForestClassifier): Trained model instance.
        metrics (dict): Performance metrics.
        hyperparameters (dict): Model hyperparameters.
    """
    save_dir = "saved_models/random_forest"
    os.makedirs(save_dir, exist_ok=True)

    # Generate a timestamped folder for this model run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"random_forest_{timestamp}")

    os.makedirs(model_path, exist_ok=True)

    # Save the model
    joblib.dump(model, os.path.join(model_path, "model.pkl"))

    # Save metadata (metrics and hyperparameters)
    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(model_path, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"Model and metadata saved in: {model_path}")
