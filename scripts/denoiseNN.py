from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import load_model
import os
import sys


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
import pandas as pd




PCA_NCOMP = 0.90
DATA_NICKNAME = "Sherwood_IGM"
SPECTRUM_LEN = 2048
REDSHIFT = 0.3
SIZE_PER_CLASS = 16384
SN = 20
"""Random Forest for Sherwood IGM"""


def psnr(clean, denoised):
    mse = np.mean((clean - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(clean)
    return 20 * np.log10(max_val / np.sqrt(mse))




def main():
    data_spec = {"name": DATA_NICKNAME,
                 "redshift": REDSHIFT,
                 "length": SPECTRUM_LEN,
                 "size per class": SIZE_PER_CLASS,
                 "nclass": 2,
                 "SN": SN,
                 "etc": "denoised",
                 "dimensionality reduction":{}
    }
                  

    # Load data
    # data, labels, headers = load_data_with_nearest_galaxy(redshift=REDSHIFT, size=20000)
    clean_data, labels, headers = load_data(SN="inf", redshift=REDSHIFT, size=SIZE_PER_CLASS)
    noisy_data, labels, headers = load_data(SN=SN, redshift=REDSHIFT, size=SIZE_PER_CLASS)
    
    # replace values over threshold with 1.0
    # data[:, :-4][data[:, :-4] > 0.95] = 1.0
    # Round all values except the last 4 columns
    # data[:, :-4] = np.round(data[:, :-4], decimals=4)

    # # class 4 vs not class 4
    resampled_idx = resample_classes(y=labels)
    clean_data, noisy_data, labels = clean_data[resampled_idx], noisy_data[resampled_idx], labels[resampled_idx]
    clean_data = clean_data[:, :SPECTRUM_LEN]
    noisy_data = noisy_data[:, :SPECTRUM_LEN]
    labels = (labels == 4).astype(int)   # 1 if class4, 0 otherwise

    # Split train and test set
    train_idx, test_idx = split_data(labels)

    X_clean_train, X_clean_test = clean_data[train_idx], clean_data[test_idx]
    X_noisy_train, X_noisy_test = noisy_data[train_idx], noisy_data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    input_dim = SPECTRUM_LEN

    """Autoencoder with l2 reg and dropout
    l2_reg = regularizers.l2(1e-5)  # L2 weight decay
    dropout_rate = 0.2              # Dropout rate

    # Encoder
    input_layer = layers.Input(shape=(SPECTRUM_LEN,))
    x = layers.Dense(1024, activation='relu', kernel_regularizer=l2_reg)(input_layer)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    encoded = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)

    # Decoder
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(encoded)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(SPECTRUM_LEN, activation='linear')(x)


    autoencoder = models.Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_noisy_train, X_clean_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.2
    )
    
    X_denoised_train = autoencoder.predict(X_noisy_train)
    X_denoised_test = autoencoder.predict(X_noisy_test)

    # # low dimensional representation
    # encoder = models.Model(input_layer, encoded)
    # X_encoded = encoder.predict(X_clean)  # or X_noisy
    """

    """1D Convolutional Denoising Autoencoder (1D-CDAE)"""
    # Encoder
    input_layer = layers.Input(shape=(input_dim, 1))
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(16, kernel_size=5, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Decoder
    x = layers.Conv1D(16, kernel_size=5, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    decoded = layers.Conv1D(1, kernel_size=5, activation='linear', padding='same')(x)

    model = models.Model(input_layer, decoded)
    model.compile(optimizer='adam', loss='mse')

    X_noisy_train = X_noisy_train[..., np.newaxis]  #(n_samples, 2048, 1)
    X_noisy_test = X_noisy_test[..., np.newaxis]  #(n_samples, 2048, 1)
    X_clean_train = X_clean_train[..., np.newaxis]  #(n_samples, 2048, 1)
    X_clean_test = X_clean_test[..., np.newaxis]  #(n_samples, 2048, 1)

    # model.fit(X_noisy_train, X_clean_train, batch_size=64, epochs=5, validation_split=0.1)  
    # model.save("denoising_autoencoder.keras")

    model = load_model("denoising_autoencoder.keras")

    
    X_denoised_train = model.predict(X_noisy_train)
    X_denoised_test = model.predict(X_noisy_test)

    psnr_train_score = psnr(X_clean_train, X_denoised_train)
    psnr_test_score = psnr(X_clean_test, X_denoised_test)

    print(f"PSNR train: {psnr_train_score:.2f} dB")
    print(f"PSNR test: {psnr_test_score:.2f} dB")

    X_denoised_train = X_denoised_train.reshape((X_denoised_train.shape[0], X_denoised_train.shape[1]))  # shape: (n_samples, 2052)
    X_denoised_test = X_denoised_test.reshape((X_denoised_test.shape[0], X_denoised_test.shape[1]))  # shape: (n_samples, 2052)
    X_clean_train = X_clean_train.reshape((X_clean_train.shape[0], X_clean_train.shape[1]))  # shape: (n_samples, 2052)
    X_clean_test = X_clean_test.reshape((X_clean_test.shape[0], X_clean_test.shape[1]))  # shape: (n_samples, 2052)
    X_noisy_test = X_noisy_test.reshape((X_noisy_test.shape[0], X_noisy_test.shape[1]))  # shape: (n_samples, 2052)
    X_noisy_train = X_noisy_train.reshape((X_noisy_train.shape[0], X_noisy_train.shape[1]))  # shape: (n_samples, 2052)


    mse_values = np.mean((X_clean_test - X_denoised_test) ** 2, axis=1)
    df_report = pd.DataFrame({
        "Sample Index": np.arange(len(mse_values)),
        "MSE": mse_values
    })


    # Get best-denoised sample indices (lowest MSE)
    best_indices = df_report.sort_values(by="MSE", ascending=True).head(3)["Sample Index"].values
    plot_denoising_results(X_clean_test, X_denoised_test, X_noisy_test, indices=best_indices)

    worst_indices = df_report.sort_values(by="MSE", ascending=False).head(3)["Sample Index"].values
    plot_denoising_results(X_clean_test, X_denoised_test, X_noisy_test, indices=worst_indices)


    # metrics, model_folder_path = train_random_forest(X_denoised_train, X_denoised_test, 
    #                                                  y_train, y_test, 
    #                                                  feature_selection=False, 
    #                                                  data_spec=data_spec)


def plot_denoising_results(X_clean, X_denoised, X_noisy, indices=[0, 1, 2], figsize=(12, 4)):
    for idx in indices:
        plt.figure(figsize=figsize)
        plt.plot(X_noisy[idx], label='Noisy Signal', color='orange', linewidth=1.5)
        plt.plot(X_clean[idx], label='Clean Signal', color='black', linewidth=2)
        plt.plot(X_denoised[idx], label='Denoised Signal', color='blue', linewidth=1.5, linestyle='--')

        plt.title(f'Sample {idx}')
        plt.xlabel('Spectral Feature Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()