import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def accordion_model(input_dim, accordions=1, compression=1, decompression=1):
    model = tf.keras.Sequential(name=f'accordion{accordions:02}')
    model.add(tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)))

    for i in range(accordions):
        model.add(tf.keras.layers.Dense(compression, activation='elu'))
        model.add(tf.keras.layers.Dense(decompression, activation='elu'))


    model.add(tf.keras.layers.Dense(compression, activation='elu'))
    model.add(tf.keras.layers.Dense(input_dim, activation='elu'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    return model

