"""
    This is the complete implementation of Cycle-GAN architecture for image restoration.
"""

import glob
import random

import numpy as np
import tensorflow as tf

from src.train import train
from src.generator import generator
from src.plot_images import plot_images
from src.load_dataset import load_dataset
from src.discriminator import discriminator
from src.composite_model import composite_model
from src.generate_real import generate_real_samples

# Loading the dataset
RAIN = r"/dataset/rain"
CITY = r"/dataset/city"

# TODO: Need to implement dynamic loading of images as this
# approach requires huge memory to store images.
images_r = np.zeros(shape=(2000,512,512,3))
images_c = np.zeros(shape=(2000,512,512,3))

image_paths = glob.glob(RAIN + "/*.png")
random.shuffle(image_paths)
images_r = load_dataset(image_paths, images_r)

image_paths = glob.glob(CITY + "/*.png")
random.shuffle(image_paths)
images_r = load_dataset(image_paths, images_c)

train_X = images_r[:int(0.9*len(images_r))]
test_X = images_r[int(0.9*len(images_r)):]
train_y = images_c[:int(0.9*len(images_c))]
test_y = images_c[int(0.9*len(images_c)):]

# Print the sample images
plot_images(train_X[100])
plot_images(train_y[100])

tf.random.set_seed(0)

# input shape
image_shape = (512,512,3)
# generator: A -> B
g_model_AtoB = generator(image_shape)
# generator: B -> A
g_model_BtoA = generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

# select a batch of real samples
X_realA, y_realA = generate_real_samples(train_X, 1, 24)
X_realB, y_realB = generate_real_samples(train_y, 1, 24)

# load a dataset as a list of two numpy arrays
dataset = [train_X, train_y]

# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
