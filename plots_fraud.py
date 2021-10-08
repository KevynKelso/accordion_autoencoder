import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

blue = '#003f5c'
red = '#bc5090'

def plot_model_sensitivity(df, title, metric, save=False):
    plt.clf()
    plt.bar(df['change_node'], df[metric], color=df['color'])

    blue_patch = mpatches.Patch(color=blue, label='Modified Model')
    red_patch = mpatches.Patch(color=red, label='Baseline Model')
    plt.legend(handles=[red_patch,blue_patch])

    plt.ylim([0,1])
    if 'complexity' in metric: plt.autoscale()

    plt.title(title)
    plt.xlabel('Layer nodes')
    plt.ylabel(f'{metric} score')

    if save:
        dir_name = df['name'][0].split('->')[0]
        save_path = f'./figures/{dir_name}'
        if not os.path.exists(save_path): os.makedirs(save_path)
        plt.savefig(f'./figures/{dir_name}/{title}')
        return

    plt.show()
