import tensorflow as tf
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


def fit_model_fraud(model, training_data, validation_data, model_name, noisy=0.0, num_epoch=0):
    # Training:
    training_data_noisy = add_noise(training_data, noisy)
    validation_data_noisy = add_noise(validation_data, noisy)

    r = model.fit(training_data_noisy, training_data, epochs=num_epoch, batch_size=256, shuffle=True,
            validation_data=(validation_data_noisy, validation_data),
            callbacks=[early_stop()])

    actual_epoch = len(r.history['loss'])
    model_file = f'{actual_epoch}p-{model_name}-{round(min(r.history["loss"]),3)}'
    model.save(f'models/fraud_models/{model_file}.h5')

    return r

def fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test):
    model.summary()

    r = model.fit(x_train, y_train, epochs=5, shuffle=True, validation_data=(x_test, y_test), callbacks=[early_stop()])

    actual_epoch = len(r.history['loss'])
    model_file = f'{actual_epoch}p-{model_name}-{round(min(r.history["loss"]),3)}'
    model.save(f'models/mnist_accordion_classification_models/{model_file}.h5')

    return r
