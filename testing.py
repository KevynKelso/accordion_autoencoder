import pandas as pd
import sys

from plots_fraud import plot_model_sensitivity
from utils import fatal_check_args_testing

blue = '#003f5c'
red = '#bc5090'

def dataframe_preprocessing(df, baseline_param):
    df['change_node'] = df['name'].apply(lambda x: int(x.split('->')[-1]))

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

# expecting dataframe with standard format of:
# name, loss, accuracy, val_loss, val_accuracy, precision, recall, f1, complexity
# where name is something like a-b-x-b-a->i
def plot_all_metrics_std_csv(df, pattern, baseline_nodes, save=False):
    opt = dataframe_preprocessing(get_rows_matching_name_pattern(pattern,df),baseline_nodes)

    plot_model_sensitivity(opt, f'{pattern} F1 Scores', 'f1', save=save)
    plot_model_sensitivity(opt, f'{pattern} Accuracy Scores', 'val_accuracy', save=save)
    plot_model_sensitivity(opt, f'{pattern} Loss Scores', 'val_loss', save=save)
    plot_model_sensitivity(opt, f'{pattern} Precision Scores', 'precision', save=save)
    plot_model_sensitivity(opt, f'{pattern} Recall Scores', 'recall', save=save)
    plot_model_sensitivity(opt, f'{pattern} Complexity', 'complexity', save=save)

def main():
    fatal_check_args_testing(sys.argv)

    file_name = sys.argv[1]
    df = pd.read_csv(file_name)

    plot_all_metrics_std_csv(df, '128-64-x-x-64-128', 32, save=True)

if __name__ == '__main__':
    main()

