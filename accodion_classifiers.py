# MNIST Classifier
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def accordion_mnist_classifier(model_name, accordions, compression, decompression):
    model = tf.keras.Sequential(name=model_name)
    model.add(layers.Flatten(input_shape=(28,28)))

    for _ in range(accordions):
        model.add(layers.Dense(compression, activation='relu'))
        model.add(layers.Dense(decompression, activation='relu'))

    model.add(layers.Dense(compression, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
