import matplotlib.pyplot as plt
import pandas as pd
import random
import re
import sys

from plots import *

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 testing.py <csv file>')
        return

    file_name = sys.argv[1]
    df = pd.read_csv(file_name)

    # parse information from the name
    df['decompression'] = df['name'].apply(lambda x: int(x.split('-')[-1]))
    df['compression'] = df['name'].apply(lambda x: int(x.split('-')[-2]))
    df['accordions'] = df['name'].apply(
            lambda x: int(re.sub('[a-zA-Z_]', '', x.split('-')[-3]))
    )

    colors = ['#C71585', '#8B0000', '#FFA07A', '#FF8C00', '#BDB76B', '#FFD700', '#000080', '#0000FF', '#B0C4DE', '#E0FFFF', '#006400', '#808000', '#00FF00']
    df['color'] = df['decompression'].apply(
            lambda deco: colors[deco % len(colors)]
    )

    plot_accuracy_vs_accordions(df)
    plt.show()

if __name__ == '__main__':
    main()

