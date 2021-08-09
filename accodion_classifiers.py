# MNIST Classifier
import tensorflow as tf
import math

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

def eval_accordion_mnist_classifier(model_name, accordions, compression, decompression):
    num_nodes = (accordions * (compression + decompression)) + compression
    num_layers = accordions*2 + 1
    nodes_per_layer = round(num_nodes/num_layers)

    eval_model = tf.keras.Sequential(name=model_name)
    eval_model.add(layers.Flatten(input_shape=(28,28)))

    for _ in range(num_layers):
        eval_model.add(layers.Dense(nodes_per_layer, activation='relu'))

    eval_model.add(layers.Dense(10, activation='softmax'))
    eval_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return eval_model

