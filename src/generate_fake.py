"""
    Generate fake samples from the model.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_fake_samples(g_model, dataset, patch_shape, count):
    """
        Generate fake samples using the generator model.
        inputs:
            g_model: Generator model.
            dataset: Input dataset.
            patch_shape: Required patch shape.
        outputs:
            X: Fake samples.
            y: Target images.
    """
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))

    if count%250 == 0:
        plt.figure(figsize=(12, 12))
        display_list = [dataset[0], X[0]]
        title = ['Rainy Image', 'Restored Image']
        for i in range(2):
            plt.subplot(2, 1, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow((display_list[i] + 1) * 0.5)
            plt.axis('off')
        plt.savefig("images/Iter_"+str(count)+".png")
        plt.close()
    return X, y
