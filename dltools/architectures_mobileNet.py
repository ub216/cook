from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from dltools.depthwise_conv2d import DepthwiseConvolution2D

"""
# Reference:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
"""

class mobileNetBuilder(object):
    """
    Instantiates the MobileNet Network 
    """
    def __init__(self,num_classes=1,alpha=1):
        """
        Initializes a new instance of the network.

        :param num_classes: The number of output classes.
        :param alpha: Control the width of the network
        """
        self.num_classes = num_classes
        self.alpha = alpha


    def build(self,dropout=False):
        """
        Builds the MobileNet Network 

            # Arguments
                input_tensor: Keras tensor (i.e. output of `layers.Input()`)
                    to use as image input for the model (channel last).
                shallow: optional parameter for making network smaller
                    into.
            # Returns
                A Keras model instance.

        """
        alpha = self.alpha
        model = Sequential()

        model.add(Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False, input_shape=(224, 224, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout:
            model.add(Dropout(0.25))

        model.add(DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout:
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('relu'))

        model.add(Dense(self.num_classes))
        model.add(Activation('sigmoid'))

        return model

