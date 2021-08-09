import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from creditcarddata import get_creditcard_data_normalized
from autoencoder_designs import *
from accodion_classifiers import accordion_mnist_classifier
from accordion_mnist import get_formatted_mnist_classification_data
from tf_utils import early_stop
from plots import *

from sklearn.model_selection import GridSearchCV

def grid_search_mnist():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()

    accodions_to_test = [2, 3, 4, 5]
    compressions_to_test = [32, 48, 64]
    decompression_to_test = [48, 64, 128]

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
                model.summary()

                r = model.fit(x_train, y_train, epochs=200, shuffle=True, validation_data=(x_test, y_test), callbacks=[early_stop()])

                min_loss = min(r.history['loss'])
                if min_loss < best_loss:
                    best_loss = min_loss

                actual_epoch = len(r.history['loss'])
                model_file = f'{actual_epoch}p-{model_name}-{round(min_loss,3)}'
                model.save(f'models/mnist_accordion_models/{model_file}.h5')

                # file will be accordions, compressions, decompressions, loss, accuracy, val_loss, val_accuracy
                with open("mnist_class_data1.csv", "a") as f:
                    f.write(f'{parameters["accordions"]},{parameters["compression"]},{parameters["decompression"]},' +
                            f'{round(min_loss,3)},{round(max(r.history["accuracy"]),3)},{round(min(r.history["val_loss"]),3)},' +
                            f'{round(max(r.history["val_accuracy"]),3)}\n')

                # reconstructed_imgs = model.predict(testing_data)

                # plot_original_vs_reconstructed_imgs(parameters, testing_data, reconstructed_imgs)

    with open("mnist_class_data1.csv", "a") as f:
        f.write(best_loss)

def summary_best_models():
    model = vanilla()
    model.summary()
    model = deep()
    model.summary()
    model = covnet()
    model.summary()

def main():
    # summary_best_models()
    grid_search_mnist()


if __name__ == '__main__':
    main()

