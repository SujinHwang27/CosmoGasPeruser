import os
import json
import joblib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scripts.config import CLASSES


def train_svm(X_train, X_test, y_train, y_test, data_spec):
    """
    Trains an SVM classifier and evaluates performance.

    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        data_spec (dict): specifications of the data

    Returns:
        dict: Performance metrics including confusion matrix.
    """
    # Initialize and train the SVM classifier
    model = SVC(
        kernel='rbf',          # RBF kernel (can change to 'linear' or 'poly')
        C=1.0,                 # Regularization parameter
        gamma='scale',         # Kernel coefficient
        probability=True,      # Enable probability estimates
        random_state=42
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute performance metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "test_f1_score": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
    }

    print("\nTest Confusion Matrix:")
    print(np.array(metrics["test_confusion_matrix"]))

    # Save model and metadata
    hyperparameters = {"data": data_spec, "model": model.get_params()}
    folder_path = save_model(model, metrics, hyperparameters)

    # Plot confusion matrix
    plot_confusion_matrix(metrics["test_confusion_matrix"], classes=CLASSES, folder_path=folder_path)

    return metrics, folder_path


def plot_confusion_matrix(conf_matrix, classes, folder_path):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 22},
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.set_title("Confusion Matrix", fontsize=15)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    save_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


def save_model(model, metrics, hyperparameters):
    save_dir = "saved_models/svm"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"svm_{timestamp}")
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, os.path.join(model_path, "model.pkl"))

    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(model_path, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"Model and metadata saved in: {model_path}")
    return model_path
