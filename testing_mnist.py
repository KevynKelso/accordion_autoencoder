import tensorflow as tf
import glob

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

    for model_file in baseline_mnist_models:
        print(get_model_name(model_file))
        model = tf.keras.models.load_model(model_file)
        model.summary()
        break

    names = [get_model_name(x) for x in baseline_mnist_models]
    names.sort()

    # load data and test
    for name in names:
        print(name)

if __name__ == "__main__":
    main()
