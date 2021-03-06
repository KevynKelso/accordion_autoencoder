import tensorflow as tf
import numpy as np
import glob

from accordion_mnist import get_formatted_mnist

from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score)

def get_model_name(file_path):
    file_name = file_path.split('/')[-1]
    x = file_name.split('->')[-1].split('-')[0]

    if 'l128' in file_name:
        return f'baseline_x-64-32-64-x->{x}'

    if 'l64' in file_name:
        return f'baseline_128-x-32-x-128->{x}'

    if 'l32' in file_name:
        return f'baseline_128-64-x-64-128->{x}'

    return f'Err_{file_name}'

def main():
    baseline_mnist_models = glob.glob('./models/mnist_accordion_classification_models/*baseline*.h5')
    (_, _), (x_test, y_test) = get_formatted_mnist()

    with open('./baseline_tuning_mnist.csv', 'a') as f:
        f.write('name,val_loss,val_accuracy,precision,recall,f1,complexity')

    for model_file in baseline_mnist_models:
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        model = tf.keras.models.load_model(model_file)
        model.summary()

        loss, accuracy = model.evaluate(x_test,y_test)
        y_test_pred = model.predict(x_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)

        # parameters for csv need: name,val_loss,val_accuracy,precision,recall,f1,complexity
        model_name = get_model_name(model_file)
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')

        print_line = f'{model_name},{loss},{accuracy},{precision},{recall},{f1},{trainableParams}'

        with open('./baseline_tuning_mnist.csv', 'a') as f:
            f.write(print_line)

if __name__ == "__main__":
    main()
