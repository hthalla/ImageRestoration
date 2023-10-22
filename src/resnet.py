"""
    Defining the resnet block.
"""

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization

def resnet_block(n_filters, input_layer):
    """
        Creates a residual network block
        inputs:
            n_filters: Number of filters
            input_layer: Input layer
        outputs:
            res: Residual network
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    res = Conv2D(n_filters, (5,5), padding='same', kernel_initializer=init)(input_layer)
    res = InstanceNormalization(axis=-1)(res)
    res = Activation('relu')(res)
    # second convolutional layer
    res = Conv2D(n_filters, (5,5), padding='same', kernel_initializer=init)(res)
    res = InstanceNormalization(axis=-1)(res)
    # concatenate merge channel-wise with input layer
    res = Concatenate()([res, input_layer])
    return res
