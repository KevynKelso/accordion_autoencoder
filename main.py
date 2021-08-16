import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from creditcarddata import get_creditcard_data_normalized
from autoencoder_designs import *
from accodion_classifiers import *
from accordion_mnist import get_formatted_mnist_classification_data
from tf_utils import early_stop
from plots import *

from sklearn.model_selection import GridSearchCV

def grid_search_mnist():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()

    accodions_to_test = [2]
    compressions_to_test = [32]
    decompression_to_test = [10, 11, 12, 13, 14, 15, 16]

    parameters = {
            "accordions": 2,
            "compression": 32,
            "decompression": 64,
    }

    best_loss = 1
    for acc in accodions_to_test:
        parameters["accordions"] = acc
        for comp in compressions_to_test:
            parameters["compression"] = comp
            for decomp in decompression_to_test:
                parameters["decompression"] = decomp

                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()

                model_name = f'mnist_class_accordion{parameters["accordions"]}-{parameters["compression"]}-{parameters["decompression"]}'

                model = accordion_mnist_classifier(model_name, parameters["accordions"], parameters["compression"], parameters["decompression"])
                r = fit_model(model, model_name, x_train, y_train, x_test, y_test)

                with open("baseline_tuning.csv", "a") as f:
                    f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                            f'{max(r.history["val_accuracy"])}\n')


def parameter_tuning_baseline():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()

    for i in range(1, 129):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        model_name = f'3_layer128l->{i}'
        model = baseline_classifier_ae_3l(i)

        r = fit_model(model, model_name, x_train, y_train, x_test, y_test)

        with open("baseline_tuning.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])}\n')



def fit_model(model, model_name, x_train, y_train, x_test, y_test):
    model.summary()

    r = model.fit(x_train, y_train, epochs=200, shuffle=True, validation_data=(x_test, y_test), callbacks=[early_stop()])

    actual_epoch = len(r.history['loss'])
    model_file = f'{actual_epoch}p-{model_name}-{round(min(r.history["loss"]),3)}'
    model.save(f'models/mnist_accordion_classification_models/{model_file}.h5')

    return r

def main():
    # grid_search_mnist()
    # parameter_tuning_baseline()
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()

    model_name = f'putting-it-all-together-27-10'
    model = baseline_classifier_ae(27, 10)
    r = fit_model(model, model_name, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()

