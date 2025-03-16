# modules for data preprocessing, such as applying PCA
import numpy as np

from sklearn.decomposition import PCA


# TODO : pca analyzer of var-ncomp relationship of a data subset of a redshift
#        also pca performer (do pca transformation)


def pca_transform(data, n_components):
    """
    Parameters 
    -------------------
    data: dictionary
        {'nofeedback' : [[.....]], 'stellarwind':[[.....]], ...} 

    n_components: float 
        (0, 1)

    Returns
    ---------------
    pca_data : dictionary
        {'nofeedback' : [[..]], 'stellarwind':[[..]], ...} 

    """
    pca_transformed_data = {}

    pca = PCA(n_components=n_components, svd_solver='full')

    # TODO : validate the code below
    pca.fit([spectrum for spectra in data.values() for spectrum in spectra])

    for physics, spectra in data.items():
        transformed = pca.transform(spectra)
        pca_transformed_data[physics] = np.array(transformed)

    return pca_transformed_data

