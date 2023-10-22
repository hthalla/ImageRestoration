"""
    Helper function to plot the images.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_images(images: np.ndarray):
    """
        Function to plot the images.
        input:
            images: Array of images to plot.
    """
    plt.figure()
    plt.title('Rainy image')
    plt.imshow((images+1)*0.5)
    plt.close()
