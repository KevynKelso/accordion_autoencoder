import numpy as np

def mad_score(points):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    return 0.6745 * ad / mad

def fatal_check_args_testing(args):
    if len(args) < 2:
        print('Usage: python3 testing.py <csv file>')
        exit(1)

def write_csv_std(file_name, model_name, r, precision, recall, f1, trainable_params):
    with open(file_name, "a") as f:
        f.write(f'{model_name},{min(r.history["loss"])},{max(r.history["accuracy"])},{min(r.history["val_loss"])},' +
                f'{max(r.history["val_accuracy"])},{precision},{recall},{f1},{trainable_params}\n')
