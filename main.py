import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from creditcarddata import get_creditcard_data_normalized
from autoencoder_designs import accordion_sequential
from accordion_mnist import get_formatted_mnist_data
from tf_utils import early_stop
from itertools import permutations
from plots import *

from sklearn.model_selection import GridSearchCV


# TODO: this function does not work currently
def grid_search_fraud():
    training_data, validation_data, testing_data, _ = get_creditcard_data_normalized()

    clf = GridSearchCV(AccordionAutoencoder(build_fn=accordion_fraud, epochs=10, batch_size=256), {
        'accordions': [1,2,3,4,5],
        'alpha': [1,2,3],
        'beta': [1,2,3]
    }, scoring='f1')

    clf.fit(training_data, np.ones(len(training_data)))
    df = pd.DataFrame(clf.cv_results_)
    print(df)


def grid_search_mnist():
    training_data, testing_data = get_formatted_mnist_data()
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

                model_name = f'accordion{parameters["accordions"]}-{parameters["compression"]}-{parameters["decompression"]}'
                # TODO: implement GridSearchCV

                # TODO: find good model that doesn't take forever to train
                model = accordion_sequential(784, parameters["accordions"], parameters["compression"], parameters["decompression"])
                model.summary()

                r = model.fit(training_data, training_data, epochs=200, batch_size=256, shuffle=True,
                    validation_data=(testing_data, testing_data), callbacks=[early_stop()])

                min_loss = min(r.history['loss'])
                if min_loss < best_loss:
                    best_loss = min_loss

                actual_epoch = len(r.history['loss'])
                model_file = f'{actual_epoch}p-{model_name}-{round(min_loss,3)}'
                model.save(f'models/mnist_accordion_models/{model_file}.h5')

                with open("data.txt", "a") as f:
                    f.write(f'{model_file}\n')

                reconstructed_imgs = model.predict(testing_data)

                plot_original_vs_reconstructed_imgs(parameters, testing_data, reconstructed_imgs)

    with open("data.txt", "a") as f:
        f.write(best_loss)


def main():
    grid_search_mnist()


if __name__ == '__main__':
    main()

