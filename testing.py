import matplotlib.pyplot as plt
import pandas as pd
import sys

from plots import *

def parse_layer_nodes_from_name(layer_nodes, df):
    df[f'l{layer_nodes}_nodes'] = df['name'].apply(
            lambda x: int(x.split('>')[-1]) if f'l{layer_nodes}' in x else int(layer_nodes)
    )

    df['accordion'] = df['name'].apply(
            lambda x: True if 'l128' in x and int(x.split('>')[-1]) <= 64 or 'l64' in x and int(x.split('>')[-1]) <= 32 else False
    )

    df['color'] = df[f'l{layer_nodes}_nodes'].apply(
            lambda x: '#bc5090' if x == int(layer_nodes) else '#ff7c43' if x <= 32  else '#003f5c'
    )

    return df

def process_data_and_plot_acc_vs_model_type(df):
    median_model_accs = df.groupby('accordion', as_index=False)['val_accuracy'].median()

    # Baseline, Accordion is returned
    return median_model_accs['val_accuracy'].tolist()


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 testing.py <csv file>')
        return

    file_name = sys.argv[1]
    df = pd.read_csv(file_name)

    df = parse_layer_nodes_from_name('128', df)
    plot_mean_accuracy_vs_model_type(process_data_and_plot_acc_vs_model_type(df))
    plt.show()

    # df['decompression'] = df['name'].apply(lambda x: int(x.split('-')[-1]))
    # df['compression'] = df['name'].apply(lambda x: int(x.split('-')[-2]))
    # df['accordions'] = df['name'].apply(
            # lambda x: int(re.sub('[a-zA-Z_]', '', x.split('-')[-3]))
    # )

    # colors = ['#C71585', '#8B0000', '#FFA07A', '#FF8C00', '#BDB76B', '#FFD700', '#000080', '#0000FF', '#B0C4DE', '#E0FFFF', '#006400', '#808000', '#00FF00']
    # df['color'] = df['decompression'].apply(
            # lambda deco: colors[deco % len(colors)]
        # )
        #'Accuracy vs. Number of Nodes in 1st Compression Layer (3 layer network)'

    plot_accuracy_vs_layer_nodes(df, '128', 'Accuracy vs. Number of Nodes Changed in Layer 2 and Layer 4')
    plt.show()

if __name__ == '__main__':
    main()

