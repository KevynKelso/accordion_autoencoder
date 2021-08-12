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

    accodions_to_test = [2, 3, 4, 5, 6]
    compressions_to_test = [4, 8, 16]
    decompression_to_test = [512, 1024]

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
                eval_model_name = f'{model_name}_eval'

                model = accordion_mnist_classifier(model_name, parameters["accordions"], parameters["compression"], parameters["decompression"])
                r = fit_model(model, model_name, x_train, y_train, x_test, y_test)

                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()

                eval_model = eval_accordion_mnist_classifier(eval_model_name, parameters["accordions"], parameters["compression"], parameters["decompression"])
                r_eval = fit_model(eval_model, eval_model_name, x_train, y_train, x_test, y_test)

                # file will be accordions, compressions, decompressions, loss, accuracy, val_loss, val_accuracy, eval_loss, eval_accuracy, eval_val_loss, eval_val_accuracy
                with open("mnist_class_data2.csv", "a") as f:
                    f.write(f'{parameters["accordions"]},{parameters["compression"]},{parameters["decompression"]},' +
                            f'{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                            f'{max(r.history["val_accuracy"])},{min(r_eval.history["loss"])},{max(r_eval.history["accuracy"])},' +
                            f'{min(r_eval.history["val_loss"])},{max(r_eval.history["val_accuracy"])}\n')
                # reconstructed_imgs = model.predict(testing_data)

                # plot_original_vs_reconstructed_imgs(parameters, testing_data, reconstructed_imgs)


def fit_model(model, model_name, x_train, y_train, x_test, y_test):
    model.summary()

    r = model.fit(x_train, y_train, epochs=200, shuffle=True, validation_data=(x_test, y_test), callbacks=[early_stop()])

    actual_epoch = len(r.history['loss'])
    model_file = f'{actual_epoch}p-{model_name}-{round(min(r.history["loss"]),3)}'
    model.save(f'models/mnist_accordion_classification_models/{model_file}.h5')

    return r

def main():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()
    model_name = f'best_fraud_architecture'
    model = accordion_mnist_classifier(model_name, 2, 4, 9)
    r = fit_model(model, model_name, x_train, y_train, x_test, y_test)

    with open("fraud_to_mnist_tuning.csv", "a") as f:
        f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                f'{max(r.history["val_accuracy"])}\n')

    # grid_search_mnist()


if __name__ == '__main__':
    main()

