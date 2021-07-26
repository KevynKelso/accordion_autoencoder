import numpy as np

from tensorflow.keras.datasets import mnist


def normalize_image_data_8_bit(data):
    return data.astype('float32') / 255.


# converts data from 28 x 28 image into vector of 784 pixels.
def convert_to_vector(data):
    return data.reshape((len(data), np.prod(data.shape[1:])))


def get_formatted_mnist_data():
    (training_data, _), (testing_data, _) = mnist.load_data()

    training_data = convert_to_vector(training_data)
    testing_data = convert_to_vector(testing_data)

    return normalize_image_data_8_bit(training_data), normalize_image_data_8_bit(testing_data)

