import tensorflow as tf
import numpy as np

from accodion_classifiers import double_latent_model
from accordion_mnist import get_formatted_fashion_mnist
from tf_utils import fit_model_mnist
from utils import mad_score, write_csv_std

from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score)

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

def batch_train_models(lower, upper, model_spec, data_func, model_func, test_func, csv_file_name):
    (x_train, y_train), (x_test, y_test) = data_func()

    for i in range (lower, upper+1):
        model_name = f'{model_spec}->{i}'

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        x1 = int(model_name.split('-')[0])
        x2 = int(model_name.split('-')[1])
        model = model_func(x1,x2,i)

        r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)

        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        precision, recall, f1 = test_func(model, x_test, y_test)

        write_csv_std(csv_file_name, model_name, r, precision, recall, f1, trainable_params)

def main():
    batch_train_models(
            37, 64, '128-64-x-x-64-128', get_formatted_fashion_mnist,
            double_latent_model, test_model_mnist_precision_recall_f1,
            'fashion_mnist.csv'
    )

if __name__ == '__main__':
    main()

