import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def accordion_sequential(input_dim, accordions=1, compression=1, decompression=1):
    model = tf.keras.Sequential(name=f'accordion{accordions:02}')
    model.add(tf.keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)))

    for i in range(accordions):
        model.add(tf.keras.layers.Dense(compression, activation='relu'))
        model.add(tf.keras.layers.Dense(decompression, activation='relu'))


    model.add(tf.keras.layers.Dense(compression, activation='relu'))
    model.add(tf.keras.layers.Dense(input_dim, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    return model

def accordion_covnet(input_dim, accordions=1, compression=1, decompression=1):
        input_img = keras.Input(shape=(28, 28, 1))

        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        return keras.Model(input_img, decoded)

