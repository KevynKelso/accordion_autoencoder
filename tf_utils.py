import tensorflow as tf
import click
import numpy as np

def add_noise(data, noise_factor):
    data_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(data_noisy, 0., 1.)

def early_stop():
    # early_stop, so we don't have to wait around.
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)


def initialize_train(autoencoder, training_data, validation_data, noisy=0.0, loss='binary_crossentropy', model_name='', num_epoch=0):
    if model_name == '':
        model_name = click.prompt('Enter model name', type=str)
    if num_epoch == 0:
        num_epoch = click.prompt('Enter number of epochs', type=int)

    autoencoder.compile(optimizer='adam', loss=loss, metrics=["accuracy"])

    # early_stop, so we don't have to wait around.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    # Training:
    training_data_noisy = add_noise(training_data, noisy)
    validation_data_noisy = add_noise(validation_data, noisy)

    if 'covnet' in model_name:
        training_data_noisy = training_data_noisy.reshape(len(training_data_noisy),30,1)
        training_data = training_data.reshape(len(training_data),30,1)

        validation_data_noisy = validation_data_noisy.reshape(len(validation_data_noisy),30,1)
        validation_data = validation_data.reshape(len(validation_data), 30,1)

    r = autoencoder.fit(training_data_noisy, training_data, epochs=num_epoch, batch_size=256, shuffle=True,
            validation_data=(validation_data_noisy, validation_data),
            callbacks=[early_stop])

    acual_epoch = len(r.history['loss'])

