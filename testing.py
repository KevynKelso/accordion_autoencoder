import matplotlib.pyplot as plt
import pandas as pd
import random
import re
import sys

from plots import *

def parse_layer_nodes_from_name_3layer(layer_nodes, df):
    df[f'l{layer_nodes}_nodes'] = df['name'].apply(
            lambda x: int(x.split('>')[-1]) if f'{layer_nodes}l' in x else int(layer_nodes)
    )

    df['color'] = df[f'l{layer_nodes}_nodes'].apply(
            lambda x: '#bc5090' if x == int(layer_nodes) else '#003f5c'
    )

    return df

def parse_layer_nodes_from_name_baseline(layer_nodes, df):
    # parse information from the name
    df[f'l{layer_nodes}_nodes'] = df['name'].apply(
            lambda x: int(x.split('>')[-1]) if f'l{layer_nodes}' in x else int(layer_nodes)
    )

    df['color'] = df[f'l{layer_nodes}_nodes'].apply(
            lambda x: '#bc5090' if x == int(layer_nodes) else '#003f5c'
    )

    return df

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 testing.py <csv file>')
        return

    file_name = sys.argv[1]
    df = pd.read_csv(file_name)

    df = parse_layer_nodes_from_name_3layer('128', df)

    # df['decompression'] = df['name'].apply(lambda x: int(x.split('-')[-1]))
    # df['compression'] = df['name'].apply(lambda x: int(x.split('-')[-2]))
    # df['accordions'] = df['name'].apply(
            # lambda x: int(re.sub('[a-zA-Z_]', '', x.split('-')[-3]))
    # )

    # colors = ['#C71585', '#8B0000', '#FFA07A', '#FF8C00', '#BDB76B', '#FFD700', '#000080', '#0000FF', '#B0C4DE', '#E0FFFF', '#006400', '#808000', '#00FF00']
    # df['color'] = df['decompression'].apply(
            # lambda deco: colors[deco % len(colors)]
    # )
    plot_accuracy_vs_layer_nodes(df, '128')
    plt.show()

if __name__ == '__main__':
    main()

