# main script to run custom random forest pipeline

from data.Spectra_for_Sujin.data_loader import load_data, load_wavelength
from data_preprocessing.train_test_split import split_data
from data_explorer.visualization import plot_pca_variance, plot_pca_projection
from data_preprocessing.pca import perform_pca_analysis, apply_pca_transformation
from models.random_forest import train_random_forest



PCA_NCOMP = 0.8
DATA_NICKNAME = "Sherwood_IGM"


def main():
    # Load data
    spectra, labels = load_data(0.3)

    # Split train and test set
    X_train, X_test, y_train, y_test = split_data(spectra, labels)

    pca, n_comp_dict = perform_pca_analysis(X_train)

    # pca transformation and save the pca model
    X_train_pca, X_test_pca = apply_pca_transformation(X_train, X_test, ncomp=PCA_NCOMP, pca_name=DATA_NICKNAME)

    train_random_forest(X_train, X_test, y_train, y_test)




if __name__ == "__main__":
    main()