"""
    Helper function to load the dataset.
"""

import cv2

import numpy as np

def load_dataset(image_paths: str, images: np.ndarray):
    """
        Function to load the dataset.
        inputs:
            image_paths: Path of the dataset to load.
            images: Empty numpy array.
        output:
            images: Numpy array with loaded images.
    """
    for ind, file in enumerate(image_paths):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        img = np.array((img/127.5) - 1)
        images[ind] = img
        ind += 1
        if ind%2000 == 0:
            break
    return images
