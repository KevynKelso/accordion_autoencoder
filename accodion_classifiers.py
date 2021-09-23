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

# baseline x = 128, x2 = 64, x3 = 32
def baseline_mnist(x, x2, x3):
    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(x, activation='relu')(input_img) # 128
    encoded = layers.Dense(x2, activation='relu')(encoded)

    encoded = layers.Dense(x3, activation='relu')(encoded)

    decoded = layers.Dense(x2, activation='relu')(encoded)
    decoded = layers.Dense(x, activation='relu')(decoded)
    decoded = layers.Dense(10, activation='softmax')(decoded)

    model = keras.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# baseline x1 = 16, x2 = 8, x3 = 4, latent = 2
def baseline_fraud(x1,x2,x3,latent):
    input_vec = keras.Input(shape=(29,))
    # deconstruct / encode
    encoded = layers.Dense(x1, activation='elu')(input_vec)
    encoded = layers.Dense(x2, activation='elu')(encoded)
    encoded = layers.Dense(x3, activation='elu')(encoded)

    encoded = layers.Dense(latent, activation='elu')(encoded)

    # reconstruction / decode
    decoded = layers.Dense(x3, activation='elu')(encoded)
    decoded = layers.Dense(x2, activation='elu')(decoded)
    decoded = layers.Dense(x1, activation='elu')(decoded)
    tf.keras.layers.Dense(29, activation='elu')(decoded)

    model = keras.Model(input_vec, decoded)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

# def baseline_classifier_ae_3l(x):
    # input_img = keras.Input(shape=(784,))
    # encoded = layers.Dense(x, activation='relu')(input_img)

    # encoded = layers.Dense(32, activation='relu')(encoded)

    # decoded = layers.Dense(x, activation='relu')(encoded)
    # decoded = layers.Dense(10, activation='softmax')(decoded)

    # model = keras.Model(input_img, decoded)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # return model
