from tensorflow import keras
from tensorflow.keras import layers

def double_latent_model(x, x2, x3):
    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(x, activation='relu')(input_img) # 128
    encoded = layers.Dense(x2, activation='relu')(encoded)

    encoded = layers.Dense(x3, activation='relu')(encoded)
    encoded = layers.Dense(x3, activation='relu')(encoded)

    decoded = layers.Dense(x2, activation='relu')(encoded)
    decoded = layers.Dense(x, activation='relu')(decoded)
    decoded = layers.Dense(10, activation='softmax')(decoded)

    model = keras.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

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

    decoded = layers.Dense(29, activation='elu')(decoded)

    model = keras.Model(input_vec, decoded)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

