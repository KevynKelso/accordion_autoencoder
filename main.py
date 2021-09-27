import tensorflow as tf

from accodion_classifiers import baseline_mnist, baseline_fraud
from accordion_mnist import get_formatted_mnist_classification_data
from tf_utils import fit_model_fraud, fit_model_mnist
from plots import *

from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score)

from creditcarddata import get_creditcard_data_normalized
from utils import mad_score


def parameter_tuning_baseline_mnist():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist_classification_data()
    # model_names_to_test = 'x-64-32-64-x 128-x-32-x-128 128-64-x-64-128'.split(' ')

    # for t_name in model_names_to_test:
        # model = baseline_mnist(1, 1, 1)

        # for i in range(1, 129):
            # tf.keras.backend.clear_session()
            # tf.compat.v1.reset_default_graph()
            # if '128' in t_name and '64' in t_name:
                # model = baseline_mnist(128, 64, i)

            # if '64' in t_name and '32' in t_name:
                # model = baseline_mnist(i, 64, 32)

            # if '128' in t_name and '32' in t_name:
                # model = baseline_mnist(128, i, 32)

            # model_name = f'baseline_{t_name}->{i}'

            # r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)

            # precision, recall, f1 = test_model_fraud_precision_recall_f1(model, testing_data, y_data)

            # trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

            # header for this file: name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    for i in range(1, 129):
        model = baseline_mnist(i,64,32)
        model_name = f'baseline_x-64-32-64-x->{i}'

        r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        precision, recall, f1 = test_model_mnist_precision_recall_f1(model, x_test, y_test)

        with open("baseline_tuning_mnist_10_epoch.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')


def parameter_tuning_baseline_fraud():
    training_data, validation_data, testing_data, y_data = get_creditcard_data_normalized()

    for i in range(1, 26):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        # model_name = f'baseline_4-x-4-2-4-x-4->{i}'
        # model_name = f'baseline_4-9-x-2-x-9-4->{i}'
        # model_name = f'baseline_4-2-x-2-x-2-4->{i}'
        # model_name = f'baseline_4-2-6-x-6-2-4->{i}'
        model_name = f'baseline_x-2-6-14-6-2-x->{i}'
        model = baseline_fraud(i, 2, 6, 14)

        model.summary()

        r = fit_model_fraud(model, training_data, validation_data, model_name=model_name, num_epoch=200)

        precision, recall, f1 = test_model_fraud_precision_recall_f1(model, testing_data, y_data)

        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

        # header for this file: name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
        with open("baseline_tuning_fraud.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')

def test_ind_model_fraud():
    training_data, validation_data, testing_data, y_data = get_creditcard_data_normalized()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model_name = f'test_4-2-6-2-6-2-4'
    model = baseline_fraud(4, 2, 6, 2)

    model.summary()

    r = fit_model_fraud(model, training_data, validation_data, model_name=model_name, num_epoch=200)

    precision, recall, f1 = test_model_fraud_precision_recall_f1(model, testing_data, y_data)

    print(f'precision: {precision}, recall: {recall}, f1: {f1}')


def print_baseline_models():
    baseline_fraud(4,2,4,2).summary()
    # baseline_mnist(128, 64, 32).summary()

def test_model_mnist_precision_recall_f1(model, testing_data, y_data):
    y_test_pred = model.predict(testing_data)
    y_test_pred = np.argmax(y_test_pred, axis=1)

    precision = precision_score(y_data, y_test_pred, average='weighted')
    recall = recall_score(y_data, y_test_pred, average='weighted')
    f1 = f1_score(y_data, y_test_pred, average='weighted')

    return precision, recall, f1

def test_model_fraud_precision_recall_f1(model, testing_data, y_data):
    decoded_data = model.predict(testing_data)

    mse = np.mean(np.power(testing_data - decoded_data, 2), axis=1)

    z_scores = mad_score(mse)
    THRESHOLD = 3

    outliers = z_scores > THRESHOLD

    precision = precision_score(y_data, outliers)
    recall = recall_score(y_data, outliers)
    f1 = f1_score(y_data, outliers)

    return precision, recall, f1

def main():
    # grid_search_mnist()
    # parameter_tuning_baseline_fraud()
    parameter_tuning_baseline_mnist()
    # print_baseline_models()
    # test_baseline()
    # test_ind_model_fraud()


if __name__ == '__main__':
    main()

