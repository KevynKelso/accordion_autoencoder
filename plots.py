import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from plots_utils import assign_colors_to_tracked_data_points


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

def plot_accuracy_vs_layer_nodes(data, layer_nodes, title):
    plt.title(title)
    plt.ylim((90, 100))
    plt.xticks(np.arange(0,128+1,10))
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy (%)')
    plt.bar(data[f'l{layer_nodes}_nodes'], data['val_accuracy']*100, color=data['color'])
    blue_patch = mpatches.Patch(color='#003f5c', label='Modified Model')
    red_patch = mpatches.Patch(color='#bc5090', label='Baseline Model')
    orange_patch = mpatches.Patch(color='#ff7c43', label='Accordion Model')
    plt.legend(handles=[red_patch, blue_patch, orange_patch])

# expecting accuracies to be an array of length 2
def plot_mean_accuracy_vs_model_type(mean_accuracies):
    plt.title('Mean Accuracy Accordion vs. Baseline Models')
    plt.ylim((90,100))
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Model Type')
    plt.bar(['Baseline', 'Accordion'], np.array(mean_accuracies)*100, color=['#bc5090', '#003f5c'])

def plot_loss_vs_accordions(data):
    # naming goes y vs x
    plt.title('Loss vs. Accordions')
    plt.scatter(data['accordions'], data['loss'], s=10, c='red')
    plt.xlabel('Number of Accordions')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_accordions(data):
    # models will be linked if they have the same number of other parameters.
    plt.title('Validation Accuracy vs. Accordions')
    # for d in data:
        # plt.scatter(d['accordions'], d['val_accuracy'], s=10, c=d['color'], label=d['label'])
    plt.scatter(data['accordions'], data['accuracy'], s=10, c=data['color'])
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), loc='upper left')
    plt.xlabel('Number of Accordions')
    plt.ylabel('Final Validation Accuracy')
    plt.tight_layout()

def plot_loss_vs_d_c_difference(data):
    plt.title('Loss vs. Decompression - Compression')
    plt.scatter([d['decompression'] - d['compression'] for d in data], [d['loss'] for d in data], s=10, c='red')
    plt.xlabel('#Decompression Nodes - #Compression Nodes')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_d_c_difference(data):
    plt.title('Validation Accuracy vs. Decompression - Compression')
    plt.scatter([d['decompression'] - d['compression'] for d in data], [d['val_accuracy'] for d in data], s=10, c='red')
    plt.xlabel('#Decompression Nodes - #Compression Nodes')
    plt.ylabel('Final Validation Accuracy')

def plot_loss_vs_compression(data):
    plt.title('Loss vs. Compression Nodes')
    plt.scatter([d['compression'] for d in data], [d['loss'] for d in data], s=10, c='red')
    plt.xlabel('Number of Compression Nodes')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_compression(data):
    data = assign_colors_to_tracked_data_points('compression', data)
    plt.title('Validation Accuracy vs. Compression Nodes')
    for d in data:
        plt.scatter(d['compression'], d['val_accuracy'], s=10, c=d['color'], label=d['label'])
    # plt.scatter([d['compression'] for d in data], [d['val_accuracy'] for d in data], s=10, c=[d['color'] for d in data])
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), loc='upper left')
    plt.xlabel('Number of Compression Nodes')
    plt.ylabel('Final Validation Accuracy')

def plot_loss_vs_decompression(data):
    plt.title('Loss vs. Decompression Nodes')
    plt.scatter([d['decompression'] for d in data], [d['loss'] for d in data], s=10, c='red')
    plt.xlabel('Number of Decompression Nodes')
    plt.ylabel('Final Loss')

def plot_accuracy_vs_decompression(data):
    data = assign_colors_to_tracked_data_points('decompression', data)
    plt.title('Validation Accuracy vs. Decompression Nodes')
    for d in data:
        plt.scatter(d['decompression'], d['val_accuracy'], s=10, c=d['color'], label=d['label'])
    # plt.scatter([d['decompression'] for d in data], [d['val_accuracy'] for d in data], s=10, c=[d['color'] for d in data])
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), loc='upper left')
    plt.xlabel('Number of Decompression Nodes')
    plt.ylabel('Final Validation Accuracy')


def plot_loss_vs_epoch(data):
    plt.title('Loss vs. Number of Epoch')
    plt.scatter([d['epoch'] for d in data], [d['loss'] for d in data], s=10, c='red')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Final Loss')
    plt.show()

