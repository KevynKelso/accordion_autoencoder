import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# def accordion_sequential(input_dim, accordions=1, compression=1, decompression=1):
    # model = tf.keras.Sequential(name=f'accordion{accordions:02}')
    # model.add(tf.keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)))

    # for i in range(accordions):
        # model.add(tf.keras.layers.Dense(compression, activation='relu'))
        # model.add(tf.keras.layers.Dense(decompression, activation='relu'))


    # model.add(tf.keras.layers.Dense(compression, activation='relu'))
    # model.add(tf.keras.layers.Dense(input_dim, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    # return model


# def accordion_covnet(input_dim, accordions=1, compression=1, decompression=1):
    # # grayscale images only
    # model = tf.keras.Sequential(name=f'accordion_covnet{accordions:02}')
    # model.add(tf.keras.layers.Conv2D(compression, activation='relu', input_shape=(input_dim,input_dim,1)))

    # for _ in range(accordions):
        # model.add(tf.keras.layers.Conv2D(decompression, activation='relu'))
        # model.add(tf.keras.layers.Conv2D(decompression, activation='relu'))

    # input_img = keras.Input(shape=(28, 28, 1))

    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    # x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    # x = layers.UpSampling2D((2, 2))(x)
    # decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # return keras.Model(input_img, decoded)

# def deep():
    # input_img = keras.Input(shape=(784,))
    # encoded = layers.Dense(128, activation='relu')(input_img)
    # encoded = layers.Dense(64, activation='relu')(encoded)
    # encoded = layers.Dense(32, activation='relu')(encoded)

    # decoded = layers.Dense(64, activation='relu')(encoded)
    # decoded = layers.Dense(128, activation='relu')(decoded)
    # decoded = layers.Dense(784, activation='sigmoid')(decoded)
    # return keras.Model(input_img, decoded)

def vanilla():
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(784, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    return autoencoder

# def covnet():
    # input_img = keras.Input(shape=(28, 28, 1))

    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    # x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    # x = layers.UpSampling2D((2, 2))(x)
    # decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # autoencoder = keras.Model(input_img, decoded)
    # return autoencoder
