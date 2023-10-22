"""
    Randomly selects real samples from the dataset.
"""

import numpy as np

def generate_real_samples(dataset, n_samples, patch_shape):
    """
        Generate real samples.
        inputs:
            dataset: Images dataset.
            n_samples: Number of samples.
            patch_shape: Required patch shape.
        outputs:
            X: Images
            y: Target
    """
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y
