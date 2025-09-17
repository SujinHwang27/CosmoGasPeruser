import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.Spectra_for_Sujin.data_loader import load_data, load_wavelength
from data_explorer.visualization import plot_spectra




def main():
    # Load data
    spectra, labels = load_data(0.3)
    wavelength = load_wavelength()

    plot_spectra(spectra, labels, wavelength, sample_idx=1234)
    #4000, 5000, 6000, 10000
    #7000, 12000


if __name__ == "__main__":
    main()