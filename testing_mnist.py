import glob

def main():
    baseline_mnist_models = glob.glob('./models/mnist_accordion_classification_models/*baseline*.h5')
    print(baseline_mnist_models)

if __name__ == "__main__":
    main()
