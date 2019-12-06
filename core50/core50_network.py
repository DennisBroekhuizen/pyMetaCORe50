"""
Author:         Dennis Broekhuizen, Tilburg University
Credits:        Giacomo Spigler, pyMeta: https://github.com/spiglerg/pyMeta
Description:    Pre-defined CORe50 network model.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Activation, \
                                    GlobalAveragePooling2D, Input, ZeroPadding2D, Convolution2D, Dropout, MaxPooling2D


def make_core50_cnn_model(num_output_classes):
    model = tf.keras.models.Sequential()
    for i in range(4):
        if i == 0:
            model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=None,
                             input_shape=[128, 128, 3]))
        else:
            model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=None))

        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=2, strides=2, padding="same"))
        model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(num_output_classes, activation='softmax'))

    return model
