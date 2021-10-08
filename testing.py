import pandas as pd
import sys

from plots_fraud import plot_model_sensitivity
from utils import fatal_check_args_testing

blue = '#003f5c'
red = '#bc5090'

def dataframe_preprocessing(df, baseline_param):
    df['change_node'] = df['name'].apply(lambda x: int(x.split('>')[-1]))

    df['color'] = df['change_node'].apply(
            lambda x: red if x == baseline_param else blue
    )

    return df

def get_rows_matching_name_pattern(name_pattern, df):
    return df[df['name'].str.contains(name_pattern)]

# expecting dataframe with standard format of:
# name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
# where name is something like a-b-x-b-a->i
def plot_all_metrics_std_csv(df, pattern, baseline_nodes, save=False):
    metric_map = {
            'F1 Scores': 'f1',
            'Accuracy Scores': 'val_accuracy',
            'Loss Scores': 'val_loss',
            'Precision Scores': 'precision',
            'Recall Scores': 'recall',
            'Complexity': 'complexity',
    }
    opt = dataframe_preprocessing(get_rows_matching_name_pattern(pattern,df),baseline_nodes)

    for title, metric in metric_map.items():
        plot_model_sensitivity(opt, f'{pattern} {title}', metric, save=save)

def main():
    fatal_check_args_testing(sys.argv)

    file_name = sys.argv[1]
    df = pd.read_csv(file_name)

    plot_all_metrics_std_csv(df, '128-64-x-x-64-128', 32, save=True)

if __name__ == '__main__':
    main()

