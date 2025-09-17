# main script to run custom random forest pipeline
import sys
import joblib
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.Spectra_for_Sujin.data_loader import resample_classes, load_data, load_data_with_nearest_galaxy
from data_preprocessing.train_test_split import split_data
from data_explorer.visualization import plot_pca_variance, plot_pca_projection
from data_preprocessing.pca import perform_pca_analysis, apply_pca_transformation
from models.random_forest import train_random_forest
from model_analysis.host_galaxy import plot_galaxy_mass

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



PCA_NCOMP = 0.90
DATA_NICKNAME = "Sherwood_IGM"
SPECTRUM_LEN = 2048
REDSHIFT = 0.3
SIZE_PER_CLASS = 16384
SN = 20
"""Random Forest for Sherwood IGM"""


def main():
    data_spec = {"name": DATA_NICKNAME,
                 "redshift": REDSHIFT,
                 "length": SPECTRUM_LEN,
                 "size per class": SIZE_PER_CLASS,
                 "nclass": 4,
                 "SN": SN,
                 "dimensionality reduction":{}
    }
                  

    # Load data
    # data, labels, headers = load_data_with_nearest_galaxy(redshift=REDSHIFT, size=20000)
    data, labels, headers = load_data(SN=SN, redshift=REDSHIFT, size=SIZE_PER_CLASS)
    # replace values over threshold with 1.0
    data[:, :-4][data[:, :-4] > 0.9] = 1.0
    # Round all values except the last 4 columns
    # data[:, :-4] = np.round(data[:, :-4], decimals=4)

    # class 4 vs not class 4
    resampled_idx = resample_classes(y=labels)
    data, labels = data[resampled_idx], labels[resampled_idx]
    labels = (labels == 4).astype(int)   # 1 if class4, 0 otherwise

    # Split train and test set
    train_idx, test_idx = split_data(labels)

    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # pca, n_comp_dict = perform_pca_analysis(X_train[:, :SPECTRUM_LEN], show_plot=False)
    # data_spec["dimensionality reduction"]["n comp dict"] = n_comp_dict

    # # pca transformation and save the pca model
    # X_train_pca, X_test_pca = apply_pca_transformation(X_train[:, :SPECTRUM_LEN], X_test[:, :SPECTRUM_LEN], ncomp=PCA_NCOMP, pca_name=DATA_NICKNAME)
    # data_spec["dimensionality reduction"]= {"name":"PCA", "var":PCA_NCOMP}

    
    # train rf and save model parameters
    # pca_metrics, pca_model_folder_path = train_random_forest(X_train_pca, X_test_pca, y_train, y_test, data_spec=data_spec)
    metrics, model_folder_path = train_random_forest(X_train[:, :SPECTRUM_LEN], X_test[:, :SPECTRUM_LEN], 
                                                     y_train, y_test, 
                                                     feature_selection=False, 
                                                     data_spec=data_spec)

    # load the saved rf param
    # pca_model = joblib.load(os.path.join(pca_model_folder_path, "model.pkl"))
    # no_pca_model = joblib.load(os.path.join(model_folder_path, "model.pkl"))

    # y_train_pred = no_pca_model.predict(X_train[:, :SPECTRUM_LEN])
    # y_test_pred = no_pca_model.predict(X_test[:, :SPECTRUM_LEN])

    # y_pca_train_pred = pca_model.predict(X_train_pca)
    # y_pca_test_pred = pca_model.predict(X_test_pca)

    # y_train_final = np.where(y_train_pred == y_pca_train_pred, y_pca_train_pred, y_train_pred)
    # y_test_final = np.where(y_test_pred == y_pca_test_pred, y_pca_test_pred, y_test_pred)


    # # Assuming y_true and y_final_pred are NumPy arrays
    # train_conf_matrix = confusion_matrix(y_train, y_train_final)
    # test_conf_matrix = confusion_matrix(y_test, y_test_final)

    # # Print the raw confusion matrix
    # print("Train Confusion Matrix:")
    # print(train_conf_matrix)

    # # Print the raw confusion matrix
    # print("Test Confusion Matrix:")
    # print(test_conf_matrix)

    # # Optional: Plot the confusion matrix
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix: y_true vs y_final_pred")
    # plt.tight_layout()
    # plt.show()


    # train_confusion_3d = np.zeros((2, 2, 2), dtype=int)
    # for yt, yp, ypp in zip(y_train, y_train_pred, y_pca_train_pred):
    #     train_confusion_3d[yt, yp, ypp] += 1

    # test_confusion_3d = np.zeros((2, 2, 2), dtype=int)
    # for yt, yp, ypp in zip(y_test, y_test_pred, y_pca_test_pred):
    #     test_confusion_3d[yt, yp, ypp] += 1

    # print("Train")
    # print(train_confusion_3d)

    # print("Test")
    # print(test_confusion_3d)


    # plot_galaxy_mass(y_train_pred, y_train, X_train[:, SPECTRUM_LEN:], headers[SPECTRUM_LEN:])
    # plot_galaxy_mass(y_test_pred, y_test, X_test[:, SPECTRUM_LEN:], headers[SPECTRUM_LEN:])
   

if __name__ == "__main__":
    main()