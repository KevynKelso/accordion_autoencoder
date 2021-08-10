import sys

from plots import *
# want to test y axis = loss
# x axis = number of accordions
# x axis = difference between compression and decompression
# x axis = compression nodes
# x axis = decompression nodes
def extract_data_from_name(name):
    # expecting lines to look like 151p-accordion3-32-48-0.093.h5
    name = name.replace('.h5', '')
    loss = float(name.split('-')[-1].strip())
    decompression = int(name.split('-')[-2].strip())
    compression = int(name.split('-')[-3].strip())
    acc = int(name.split('-')[-4].strip().replace('accordion', ''))
    epoch = int(name.split('-')[0].replace('p', ''))

    return {"epoch": epoch, "loss": loss, "decompression": decompression, "compression": compression, "acc": acc}

# file will be accordions, compressions, decompressions, loss, accuracy, val_loss, val_accuracy, eval_loss, eval_accuracy, eval_val_loss, eval_val_accuracy
# want data to be a list of {'loss': 0.366, 'decompression': 9, 'compression': 4, 'acc': 2}
def parse_csv_accordion_metrics(file_line):
    data = file_line.split(',')
    if len(data) != 11:
        print(f'Error, expected data to have 11 elements, got {len(data)}.\n{data}\n\n')
        return

    # TODO: each instance needs a color so it can be kept track of.
    return {'accordions': int(data[0]), 'compression': int(data[1]),
            'decompression': int(data[2]), 'loss': float(data[3]),
            'accuracy': float(data[4]), 'val_loss': float(data[5]),
            'val_accuracy': float(data[6]), 'eval_loss': float(data[7]),
            'eval_accuracy': float(data[8]), 'eval_val_loss': float(data[9]),
            'eval_val_accuracy': float(data[10])}


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 testing.py <csv file>')
        return

    file_name = sys.argv[1]
    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(parse_csv_accordion_metrics(line))

    plot_accordion_model_data(data)
    plot_accuracy_by_model_and_eval_model(data)
    # plot_loss_vs_epoch(data)


if __name__ == '__main__':
    main()

