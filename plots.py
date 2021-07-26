import matplotlib.pyplot as plt


def plot_original_vs_reconstructed_imgs(original, reconstructed):
    n = 10 # 20 digits to display
    plt.figure(figsize=(20, 6))

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
    plt.savefig('original_vs_reconstructed.png')

