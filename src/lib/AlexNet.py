
import numpy as np

from convnetskeras.customlayers import crosschannelnormalization

from convnetskeras.customlayers import Softmax4D

from convnetskeras.customlayers import splittensor

from convnetskeras.imagenet_tool import synset_to_dfs_ids

from keras.layers import Activation

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import Input

from keras.layers import merge

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.convolutional import ZeroPadding2D

from keras.models import Model

from keras.models import Sequential

from keras.optimizers import SGD

from scipy.misc import imread

from scipy.misc import imresize

def model_generator(inputshape,weights_path = None):

    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=inputshape))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))



    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))



    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))



    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))



    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))



    model.add(Flatten(name='flatten'))

    model.add(Dense(4096, activation='relu', name='dense_1'))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu', name='dense_2'))

    model.add(Dropout(0.5))

    model.add(Dense(1000, name='dense_3'))

    model.add(Activation('softmax', name='softmax'))



    if weights_path:

        model.load_weights(weights_path)

    return model