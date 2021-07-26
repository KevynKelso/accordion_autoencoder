import tensorflow as tf


def early_stop():
    # early_stop, so we don't have to wait around.
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1, 
        mode='min',
        restore_best_weights=True)

