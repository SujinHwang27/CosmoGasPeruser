# TODO : rename this file 

# param : base_dir (location of data)

# Data Loader : Load dataset of a specific redshift. (param: redshift value). should return loaded data (original)

# eda : Perform EDA and save into the report

# Data Preprocessing : preprocess data using pca. first analyze and save pca-transformed data. should return var-ncomp dictionary

# SVM Tune Hyperparam : (param: search grid, k value, data) 
#                       perform grid search + kfold cv. fix the data-subset size. should return gs result excel file and best hyperparam combination. 

# SVM Full Training : perform SVM training with the given hyperparameter on full data. should save model weights and train/test score

from data_loader import load_data
from eda import visualize_spectrum
from data_preprocessing import perform_pca
from svm_training import svm_tune_hyperparam

def main(base_dir):
    # Load data
    physics_values = [1, 2, 3, 4]
    redshift_values = [0.1, 0.3, 2.2, 2.4]
    data = load_data(base_dir, physics_values, redshift_values)

    # Perform PCA
    n_components = 10
    pca_data = perform_pca(data, n_components)

    # Train SVM
    param_grid = {"C": [1, 10], "kernel": ["linear", "rbf"]}
    results = svm_tune_hyperparam(param_grid, pca_data, "svm_model.pkl")

if __name__ == "__main__":
    main("/path/to/base_dir")
