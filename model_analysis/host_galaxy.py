import numpy as np
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_galaxy_mass(y_pred, y_true, host_galaxy, headers):
    """
    Compares host galaxy properties (mass, impact parameter) between
    correctly vs incorrectly classified data, and between classes.
    
    Parameters:
    - y_pred: array-like, shape (n_samples,)
    - y_true: array-like, shape (n_samples,)
    - host_galaxy: array-like, shape (n_samples, 8)
    - headers: list of strings, length = 8
    """
    logger.info("Generating host galaxy analysis report")
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    host_galaxy = np.array(host_galaxy)

    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    # --- Extract relevant host galaxy properties ---
    mass_total = host_galaxy[:, headers.index('mass_gas')] + \
                 host_galaxy[:, headers.index('mass_dm')] + \
                 host_galaxy[:, headers.index('mass_stars')] + \
                 host_galaxy[:, headers.index('mass_bh')]

    impact_param = host_galaxy[:, headers.index('impact_param')]

    # --- Plot total mass and impact parameter: correct vs incorrect ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(mass_total[correct_mask], bins=50, alpha=0.6, label='Correct')
    axs[0].hist(mass_total[incorrect_mask], bins=50, alpha=0.6, label='Incorrect')
    axs[0].set_title('Host Galaxy Total Mass')
    axs[0].set_xlabel('Mass')
    axs[0].set_ylabel('Count')
    axs[0].legend()

    axs[1].hist(impact_param[correct_mask], bins=50, alpha=0.6, label='Correct')
    axs[1].hist(impact_param[incorrect_mask], bins=50, alpha=0.6, label='Incorrect')
    axs[1].set_title('Impact Parameter')
    axs[1].set_xlabel('Impact Parameter')
    axs[1].set_ylabel('Count')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # --- Per-class comparison of total mass ---
    classes = np.unique(y_true)
    fig, axs = plt.subplots(2, len(classes), figsize=(5 * len(classes), 8))

    for i, cls in enumerate(classes):
        class_mask = y_true == cls

        axs[0, i].hist(mass_total[class_mask], bins=50, color='steelblue')
        axs[0, i].set_title(f"Total Mass (Class {cls})")
        axs[0, i].set_xlabel("Mass")
        axs[0, i].set_ylabel("Count")

        axs[1, i].hist(impact_param[class_mask], bins=50, color='seagreen')
        axs[1, i].set_title(f"Impact Param (Class {cls})")
        axs[1, i].set_xlabel("Impact Parameter")
        axs[1, i].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
