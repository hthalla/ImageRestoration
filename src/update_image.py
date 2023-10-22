"""
    Updating the image pool for fake images.
"""

import numpy as np

def update_image_pool(pool, images, max_size=50):
    """
        Update image pool.
        inputs:
            pool: Image pool
            images: Images for updation.
            max_size: Maximum size of pool.
        output:
            selected: Array of selected images.
    """
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)
