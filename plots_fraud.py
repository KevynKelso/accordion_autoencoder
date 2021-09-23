import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

blue = '#003f5c'
red = '#bc5090'

# want a plot for f1 score
# requires a color assigned
def plot_model_f1(df, title):
    plt.bar(df['change_node'], df['f1'], color=df['color'])

    blue_patch = mpatches.Patch(color=blue, label='Modified Model')
    red_patch = mpatches.Patch(color=red, label='Modified Model')
    plt.legend(handles=[red_patch,blue_patch])

    plt.ylim([0,1])
    plt.title(title)
    plt.xlabel('Layer nodes')
    plt.ylabel('F1 score')
    plt.show()
