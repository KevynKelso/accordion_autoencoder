import tensorflow as tf

from autoencoder_designs import *
from accodion_classifiers import *
from accordion_mnist import get_formatted_mnist_classification_data
from tf_utils import fit_model_fraud, fit_model_mnist
from plots import *

from sklearn.metrics import (confusion_matrix,
                             precision_recall_curve,
                             precision_score,
                             recall_score,
                             f1_score)

from creditcarddata import get_creditcard_data_normalized
from utils import mad_score



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
                r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)

                with open("baseline_tuning.csv", "a") as f:
                    f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                            f'{max(r.history["val_accuracy"])}\n')


def parameter_tuning_baseline_mnist():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()

    for i in range(1, 64):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        model_name = f'baseline_l32->{i}'
        model = baseline_classifier_ae(128, 64, i)

        r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)

        with open("baseline_tuning.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])}\n')

def parameter_tuning_baseline_fraud():
    training_data, validation_data, testing_data, y_data = get_creditcard_data_normalized()

    for i in range(1, 25):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        model_name = f'baseline_4-x-4-2-4-x-4->{i}'
        model = baseline_fraud(4, i, 4, 2)

        model.summary()

        r = fit_model_fraud(model, training_data, validation_data, model_name=model_name, num_epoch=200)

        precision, recall, f1 = test_model(model, training_data, validation_data, testing_data, y_data)
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

        # header for this file: name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
        with open("baseline_tuning_fraud.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')


def print_baseline_models():
    baseline_fraud(4,9,4,2).summary()
    baseline_classifier_ae(128, 64, 32).summary()

def test_model(model, training_data, validation_data, testing_data, y_data):
    decoded_data = model.predict(testing_data)

    if len(decoded_data.shape) == 3: # covnet data
        decoded_data = decoded_data.reshape(len(decoded_data), 30)

    mse = np.mean(np.power(testing_data - decoded_data, 2), axis=1)

    z_scores = mad_score(mse)
    THRESHOLD = 3

    outliers = z_scores > THRESHOLD

    precision = precision_score(y_data, outliers)
    recall = recall_score(y_data, outliers)
    f1 = f1_score(y_data, outliers)

    return precision, recall, f1


def test_baseline():
    training_data, validation_data, testing_data, y_data = get_creditcard_data_normalized()

    model = baseline_fraud()
    fit_model_fraud(model, training_data, validation_data, loss='mse', model_name=f'Baseline_1', num_epoch=100)

    test_model(model, training_data, validation_data, testing_data, y_data)

def main():
    # grid_search_mnist()
    parameter_tuning_baseline_mnist()
    # print_baseline_models()
    # test_baseline()
    # (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()

    # model_name = f'putting-it-all-together-27-10'
    # model = baseline_classifier_ae(27, 10)
    # r = fit_model(model, model_name, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()

