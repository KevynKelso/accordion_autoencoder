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


def main():
    lines = []
    with open('data.txt', 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(extract_data_from_name(line))

    plot_accordion_model_data(data)
    plot_loss_vs_epoch(data)


if __name__ == '__main__':
    main()

