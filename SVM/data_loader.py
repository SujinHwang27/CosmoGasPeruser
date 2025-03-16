import os
import numpy as np
import matplotlib.pyplot as plt

# TODO : rename physics values
# TODO : should be able to specify redshift load only that
def load_data(base_dir, redshift):
    """
    Parameters
    -------------------------------
    base_dir : str
        directory path of data

    redshift : float
        redshift value
    

    Returns 
    -----------------------------------
    data: dictionary
        {'nofeedback' : [[.....]], 'stellarwind':[[.....]], ...} 

    """


    physics_dict = {1:'nofeedback', 2:'stellarwind', 3:'windAGN', 4:'windstrongAGN'}
    data = {str(physics):[] for physics in physics_dict.values()}

    #data_by_redshift = {str(redshift): {str(physics): [] for physics in physics_values} for redshift in redshift_values}

    # Iterate through each folder
    for physics_num, physics in physics_dict.items():
        folder_name = f"{str(physics_num)}_{redshift}"
        folder_path = os.path.join(base_dir, folder_name)
        print(f"Processing folder: {folder_name}")

        try:
            # Load flux data from the folder
            flux = np.load(os.path.join(folder_path, "flux.npy"))
            wavelength = np.load(os.path.join(folder_path, "wave.npy"))
            data[physics] = flux
            print(f"Flux data shape for {physics}: {flux.shape}")
        
        except FileNotFoundError as e:
            print(f"Files not found in folder {folder_path}: {e}")
            del data[physics]
            continue
            
        except Exception as e:
            print(f"An error occurred while processing folder {folder_path}: {e}")
            continue

    sample_spectrum = data['windstrongAGN'][0]
    print(f'Example of a spectrum in redshift {redshift}: \n{sample_spectrum}')
    print(f'Shape: {sample_spectrum.shape}')


    # Plot the spectrum
    plt.figure(figsize=(20, 4))
    plt.plot(wavelength, flux, label=f'Spectrum (windstrongAGN, z=0.3)', color='black', lw=1.5)
    plt.xlabel('Wavelength', fontsize=12)  
    plt.ylabel('Flux', fontsize=12)
    plt.title(f'Example of a spectrum of z={redshift} in wind+strongAGN universe', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

    return data
    