import numpy as np
import pandas as pd

from creditcarddata import get_creditcard_data_normalized
from autoencoder_designs import accordion_model
from accordion_mnist import get_formatted_mnist_data
from tf_utils import early_stop
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
    accordions = 2
    compression = 32
    decompression = 128
    model_name = f'accordion{accordions}-{compression}-{decompression}'
    # TODO: implement GridSearchCV

    # TODO: find good model that doesn't take forever to train
    model = accordion_model(784, accordions, compression, decompression)
    model.summary()

    r = model.fit(training_data, training_data, epochs=100, batch_size=256, shuffle=True,
        validation_data=(testing_data, testing_data), callbacks=[early_stop()])

    actual_epoch = len(r.history['loss'])
    model.save(f'models/mnist_accordion_models/{actual_epoch}p-{model_name}.h5')

    reconstructed_imgs = model.predict(testing_data)

    plot_original_vs_reconstructed_imgs(testing_data, reconstructed_imgs)



def main():
    grid_search_mnist()


if __name__ == '__main__':
    main()

