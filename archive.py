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


def accordion_mnist_classifier(model_name, accordions, compression, decompression):
    model = tf.keras.Sequential(name=model_name)
    model.add(layers.Flatten(input_shape=(28,28)))

    for _ in range(accordions):
        model.add(layers.Dense(compression, activation='relu'))
        model.add(layers.Dense(decompression, activation='relu'))

    model.add(layers.Dense(compression, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def eval_accordion_mnist_classifier(model_name, accordions, compression, decompression):
    num_nodes = (accordions * (compression + decompression)) + compression
    num_layers = accordions*2 + 1
    nodes_per_layer = round(num_nodes/num_layers)

    eval_model = tf.keras.Sequential(name=model_name)
    eval_model.add(layers.Flatten(input_shape=(28,28)))

    for _ in range(num_layers):
        eval_model.add(layers.Dense(nodes_per_layer, activation='relu'))

    eval_model.add(layers.Dense(10, activation='softmax'))
    eval_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return eval_model

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

def parameter_tuning_baseline_mnist():
    (x_train, y_train), (x_test, y_test) = get_formatted_mnist()

    model_names = '64-128-32-128-64'.split(' ')

    for model_name in model_names:
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        x1 = int(model_name.split('-')[0])
        x2 = int(model_name.split('-')[1])
        x3 = int(model_name.split('-')[2])
        model = baseline_mnist(x1,x2,x3)

        r = fit_model_mnist(model, model_name, x_train, y_train, x_test, y_test)
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        precision, recall, f1 = test_model_mnist_precision_recall_f1(model, x_test, y_test)

        with open("test_unique_arch_mnist.csv", "a") as f:
            f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                    f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainableParams}\n')

def parse_layer_nodes_from_name_mnist(layer_nodes, df):
    df[f'l{layer_nodes}_nodes'] = df['name'].apply(
            lambda x: int(x.split('>')[-1]) if f'l{layer_nodes}' in x else int(layer_nodes)
    )

    df['accordion'] = df['name'].apply(
            lambda x: True if 'l128' in x and int(x.split('>')[-1]) <= 64 or 'l64' in x and int(x.split('>')[-1]) <= 32 else False
    )

    df['color'] = df[f'l{layer_nodes}_nodes'].apply(
            lambda x: red if x == int(layer_nodes) else '#ff7c43' if x <= 32  else blue
    )

    return df

def parse_layer_nodes_from_name_fraud(layer_nodes, df):
    df[layer_nodes] = df['name'].apply(
            lambda x: int(x.split('->')[-1]) if f'' in x else int(layer_nodes)
    )

    return df

