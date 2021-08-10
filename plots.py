import numpy as np
import matplotlib.pyplot as plt


def plot_original_vs_reconstructed_imgs(parameters, original, reconstructed):
    n = 10 # 20 digits to display
    plt.figure()

    for i in range(n):
        # original image
        ax = plt.subplot(2, n, i+1)
        if i == (n / 2):
            ax.set_title('Original Images')

        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed image
        ax = plt.subplot(2, n, i+1+n)
        if i == (n / 2):
            ax.set_title('Reconstructed Images')
        plt.imshow(reconstructed[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f'{parameters["accordions"]}accords-{parameters["compression"]}c-{parameters["decompression"]}d.png')


# expecting data to be a list of {'loss': 0.366, 'decompression': 9, 'compression': 4, 'acc': 2}
# want 4 different plots
def plot_accordion_model_data(data):
    plots = [plot_accuracy_vs_accordions, plot_accuracy_vs_d_c_difference, plot_accuracy_vs_compression, plot_accuracy_vs_decompression]

    for i, plot in enumerate(plots):
        plt.subplot(2,2,i+1)
        plot(data)

    plt.tight_layout()
    plt.show()

def plot_accuracy_by_model_and_eval_model(data):
    N = len(data)
    model_accuracy = [d["accuracy"] for d in data]
    eval_model_accuracy = [d["eval_accuracy"] for d in data]
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_accuracy, width, label='Accordion Model')
    plt.bar(ind + width, eval_model_accuracy, width, label='Flattened Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by model and model type')
    plt.legend(loc='best')
    plt.show()

def plot_loss_vs_accordions(data):
    # naming goes y vs x
    plt.title('Loss vs. Accordions')
    plt.scatter([d['accordions'] for d in data], [d['loss'] for d in data], s=7, c='red')
    plt.xlabel('Number of Accordions')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_accordions(data):
    # models will be linked if they have the same number of other parameters.
    plt.title('Validation Accuracy vs. Accordions')
    plt.scatter([d['accordions'] for d in data], [d['val_accuracy'] for d in data], s=7, c='red')
    plt.xlabel('Number of Accordions')
    plt.ylabel('Final Validation Accuracy')

def plot_loss_vs_d_c_difference(data):
    plt.title('Loss vs. Decompression - Compression')
    plt.scatter([d['decompression'] - d['compression'] for d in data], [d['loss'] for d in data], s=7, c='red')
    plt.xlabel('#Decompression Nodes - #Compression Nodes')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_d_c_difference(data):
    plt.title('Validation Accuracy vs. Decompression - Compression')
    plt.scatter([d['decompression'] - d['compression'] for d in data], [d['val_accuracy'] for d in data], s=7, c='red')
    plt.xlabel('#Decompression Nodes - #Compression Nodes')
    plt.ylabel('Final Validation Accuracy')

def plot_loss_vs_compression(data):
    plt.title('Loss vs. Compression Nodes')
    plt.scatter([d['compression'] for d in data], [d['loss'] for d in data], s=7, c='red')
    plt.xlabel('Number of Compression Nodes')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_compression(data):
    plt.title('Validation Accuracy vs. Compression Nodes')
    plt.scatter([d['compression'] for d in data], [d['val_accuracy'] for d in data], s=7, c='red')
    plt.xlabel('Number of Compression Nodes')
    plt.ylabel('Final Validation Accuracy')

def plot_loss_vs_decompression(data):
    plt.title('Loss vs. Decompression Nodes')
    plt.scatter([d['decompression'] for d in data], [d['loss'] for d in data], s=7, c='red')
    plt.xlabel('Number of Compression Nodes')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_decompression(data):
    plt.title('Validation Accuracy vs. Decompression Nodes')
    plt.scatter([d['decompression'] for d in data], [d['val_accuracy'] for d in data], s=7, c='red')
    plt.xlabel('Number of Compression Nodes')
    plt.ylabel('Final Validation Accuracy')


def plot_loss_vs_epoch(data):
    plt.title('Loss vs. Number of Epoch')
    plt.scatter([d['epoch'] for d in data], [d['loss'] for d in data], s=7, c='red')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Final Loss')
    plt.show()

