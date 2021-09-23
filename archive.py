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

