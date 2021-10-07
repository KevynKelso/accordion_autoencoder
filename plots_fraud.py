import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

blue = '#003f5c'
red = '#bc5090'

# want a plot for f1 score
# requires a color assigned
def plot_model_f1(df, title, save=False):
    plt.bar(df['change_node'], df['f1'], color=df['color'])

    blue_patch = mpatches.Patch(color=blue, label='Modified Model')
    red_patch = mpatches.Patch(color=red, label='Baseline Model')
    plt.legend(handles=[red_patch,blue_patch])

    plt.ylim([0,1])
    plt.title(title)
    plt.xlabel('Layer nodes')
    plt.ylabel('F1 score')

    if save:
        plt.savefig('./figures/{title}')
        return

    plt.show()

def plot_model_sensitivity(df, title, metric, save=False):
    plt.bar(df['change_node'], df[metric], color=df['color'])

    blue_patch = mpatches.Patch(color=blue, label='Modified Model')
    red_patch = mpatches.Patch(color=red, label='Baseline Model')
    plt.legend(handles=[red_patch,blue_patch])

    plt.ylim([0,1])
    if 'complexity' in metric:
        plt.autoscale()

    plt.title(title)
    plt.xlabel('Layer nodes')
    plt.ylabel(f'{metric} score')

    if save:
        plt.savefig(f'./figures/{title}')
        return

    plt.show()
