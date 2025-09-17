import os
import json
import joblib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scripts.config import CLASSES

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_important_features(X_train, y_train, X_test, threshold="median"):
    """
    Selects important features using feature importances from Random Forest.

    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test data.
        threshold (str or float): Threshold for feature importance. Can be "mean", "median", or a float.

    Returns:
        X_train_selected, X_test_selected, selector
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]} using threshold = {threshold}")
    
    return X_train_selected, X_test_selected, selector


def train_random_forest(X_train, X_test, y_train, y_test, data_spec, feature_selection=False):
    """
    Trains a Random Forest classifier and evaluates performance.

    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        data_spec (dict): specifications of the data


    Returns:
        dict: Performance metrics including confusion matrix.
    """
    if feature_selection==True:
        X_train, X_test, selector = select_important_features(X_train=X_train, y_train=y_train, X_test=X_test)

    # Initialize and train the Random Forest classifier
    # model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model = RandomForestClassifier(
        bootstrap= True,
        ccp_alpha= 0.0,
        criterion= "gini",
        max_depth= 20,
        max_features= "log2",
        max_leaf_nodes= None,
        max_samples= None,
        min_impurity_decrease= 0.0,
        min_samples_leaf= 30,
        min_samples_split= 100,
        min_weight_fraction_leaf= 0.0,
        monotonic_cst= None,
        n_estimators= 200,
        n_jobs= -1,
        oob_score= False,
        random_state= 42,
        verbose= 0,
        warm_start= False,
        class_weight='balanced'
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
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()  # Convert to list for saving
    }

    # Print Confusion Matrix
    print("\nTest Confusion Matrix:")
    print(np.array(metrics["test_confusion_matrix"]))


    # Save the trained model and metadata
    hyperparameters = {"data": data_spec, "model":model.get_params()}
    folder_path = save_model(model, metrics, hyperparameters)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(metrics["test_confusion_matrix"], folder_path=folder_path)


    return metrics, folder_path

def plot_confusion_matrix(conf_matrix, folder_path, classes=None):
    """
    Plots and saves the confusion matrix as an image.
    """
    plt.figure(figsize=(8, 6))  # Increase size of figure
    ax = sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        # xticklabels=classes, 
        # yticklabels=classes,
        annot_kws={"size": 22},  # Font size for the numbers inside boxes
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.set_title("Confusion Matrix", fontsize=15)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()  # Better spacing
    save_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)  # Save as PNG
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

    return model_path




def main():
    # # Test Plot Confusion Matrix
    # # Confusion matrix
    # conf_matrix = np.array([
    #     [1500, 190, 158, 152],
    #     [199, 1485, 157, 159],
    #     [164, 176, 1481, 179],
    #     [132, 156, 104, 1608]
    # ])

    # # Multi-line labels
    # class_labels = ['no\nfeedback', 'stellar\nwind', 'wind+\nAGN', 'wind+\nstrongAGN']

    # # Plot
    # plt.figure(figsize=(8, 6))
    # ax = sns.heatmap(
    #     conf_matrix, 
    #     annot=True, 
    #     fmt="d", 
    #     cmap="Blues", 
    #     xticklabels=class_labels, 
    #     yticklabels=class_labels,
    #     annot_kws={"size": 22}
    # )

    # ax.set_xlabel("\nPredicted Label", fontsize=16)
    # ax.set_ylabel("True Label\n", fontsize=16)
    # # ax.set_title("Confusion Matrix", fontsize=18)
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='y', labelsize=20)

    # plt.tight_layout()
    # plt.savefig("confusion_matrix.png", dpi=300)
    # plt.show()
    return


if __name__ == "__main__":
    main()
