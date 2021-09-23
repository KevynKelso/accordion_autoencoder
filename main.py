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

    for i in range(1, 64):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        model_name = f'baseline_l32->{i}'
        model = baseline_mnist(128, 64, i)

        r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)

        with open("baseline_tuning.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])}\n')

def delete_me():
    training_data, validation_data, testing_data, y_data = get_creditcard_data_normalized()

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    # model_name = f'baseline_4-x-4-2-4-x-4->{i}'
    model_name = f'baseline_4-9-x-2-x-9-4->25'
    # model = baseline_fraud(4, i, 4, 2)
    model = baseline_fraud(4, 9, 25, 2)

    model.summary()

    r = fit_model_fraud(model, training_data, validation_data, model_name=model_name, num_epoch=200)

    precision, recall, f1 = test_model_fraud_precision_recall_f1(model, testing_data, y_data)

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

    # header for this file: name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
    with open("baseline_tuning_fraud.csv", "a") as f:
        f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model_name = f'baseline_4-x-4-2-4-x-4->25'
    # model_name = f'baseline_4-9-x-2-x-9-4->{i}'
    model = baseline_fraud(4, 25, 4, 2)
    # model = baseline_fraud(4, 9, 25, 2)

    model.summary()

    r = fit_model_fraud(model, training_data, validation_data, model_name=model_name, num_epoch=200)

    precision, recall, f1 = test_model_fraud_precision_recall_f1(model, testing_data, y_data)

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

    # header for this file: name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
    with open("baseline_tuning_fraud.csv", "a") as f:
        f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')

def parameter_tuning_baseline_fraud():
    training_data, validation_data, testing_data, y_data = get_creditcard_data_normalized()

    for i in range(1, 25):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        # model_name = f'baseline_4-x-4-2-4-x-4->{i}'
        model_name = f'baseline_4-9-x-2-x-9-4->{i}'
        # model = baseline_fraud(4, i, 4, 2)
        model = baseline_fraud(4, 9, i, 2)

        model.summary()

        r = fit_model_fraud(model, training_data, validation_data, model_name=model_name, num_epoch=200)

        precision, recall, f1 = test_model_fraud_precision_recall_f1(model, testing_data, y_data)

        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

        # header for this file: name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
        with open("baseline_tuning_fraud.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')


def print_baseline_models():
    baseline_fraud(4,9,4,2).summary()
    baseline_mnist(128, 64, 32).summary()

def test_model_fraud_precision_recall_f1(model, testing_data, y_data):
    decoded_data = model.predict(testing_data)

    # Not sure if this is needed
    # if len(decoded_data.shape) == 3: # covnet data
        # decoded_data = decoded_data.reshape(len(decoded_data), 30)

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
    delete_me()
    # print_baseline_models()
    # test_baseline()


if __name__ == '__main__':
    main()

