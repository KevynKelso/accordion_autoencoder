import matplotlib.pyplot as plt
import pandas as pd
import sys

from plots import *
from plots_fraud import plot_model_f1

blue = '#003f5c'
red = '#bc5090'

def parse_layer_nodes_from_name_mnist(layer_nodes, df):
    df[f'l{layer_nodes}_nodes'] = df['name'].apply(
            lambda x: int(x.split('>')[-1]) if f'l{layer_nodes}' in x else int(layer_nodes)
    )

    df['accordion'] = df['name'].apply(
            lambda x: True if 'l128' in x and int(x.split('>')[-1]) <= 64 or 'l64' in x and int(x.split('>')[-1]) <= 32 else False
    )

    df['color'] = df[f'l{layer_nodes}_nodes'].apply(
            lambda x: red if x == int(layer_nodes) else '#ff7c43' if x <= 32  else blue
    )

    return df

def parse_layer_nodes_from_name_fraud(layer_nodes, df):
    df[layer_nodes] = df['name'].apply(
            lambda x: int(x.split('>')[-1]) if f'' in x else int(layer_nodes)
    )

    return df

def dataframe_preprocessing_fraud(df, baseline_param):
    df['change_node'] = df['name'].apply(lambda x: int(x.split('>')[-1]))

    df['color'] = df['change_node'].apply(
            lambda x: red if x == baseline_param else blue
    )

    return df


def get_rows_matching_name_pattern(name_pattern, df):
    return df[df['name'].str.contains(name_pattern)]

# Not sure if this is useful
def process_data_and_plot_acc_vs_model_type(df):
    median_model_accs = df.groupby('accordion', as_index=False)['val_accuracy'].median()

    # Baseline, Accordion is returned
    return median_model_accs['val_accuracy'].tolist()

def plot_all_metrics_fraud(df):
    opt1 = dataframe_preprocessing_fraud(get_rows_matching_name_pattern('4-x-4-2-4-x-4',df), 8)
    print(opt1)
    opt2 = dataframe_preprocessing_fraud(get_rows_matching_name_pattern('4-9-x-2-x-9-4',df), 4)
    opt3 = dataframe_preprocessing_fraud(get_rows_matching_name_pattern('4-9-4-x-4-9-4',df), 2)

    # plot_model_f1(opt1, '4-x-4-2-4-x-4 F1 Scores')
    # plot_model_f1(opt2, '4-9-x-2-x-9-4 F1 Scores')
    plot_model_f1(opt3, '4-9-4-x-4-9-4 F1 Scores')


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 testing.py <csv file>')
        return

    file_name = sys.argv[1]
    df = pd.read_csv(file_name)

    plot_all_metrics_fraud(df)


    # get_rows_matching_name_pattern('4-x-4-2-4-x-4',df)

    # df = parse_layer_nodes_from_name_mnist('layer', df)


    # plot_accuracy_vs_layer_nodes_mnist(df, '64', 'Accuracy vs. Number of Nodes Changed in Layer 2 and 4')
    # plt.show()

if __name__ == '__main__':
    main()

