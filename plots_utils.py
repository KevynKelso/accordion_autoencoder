def extract_data_from_name1(name):
    # expecting lines to look like 151p-accordion3-32-48-0.093.h5
    name = name.replace('.h5', '')
    loss = float(name.split('-')[-1].strip())
    decompression = int(name.split('-')[-2].strip())
    compression = int(name.split('-')[-3].strip())
    acc = int(name.split('-')[-4].strip().replace('accordion', ''))
    epoch = int(name.split('-')[0].replace('p', ''))

    return {"epoch": epoch, "loss": loss, "decompression": decompression, "compression": compression, "acc": acc}

# def apply_color_to_same_data(variable, constant1, constant2):
    # # find all the datas with the same compression and decompression
    # if constant1 == constant2:



def assign_colors_to_tracked_data_points(key, data):
    colors = ['#C71585', '#8B0000', '#FFA07A', '#FF8C00', '#BDB76B', '#FFD700', '#000080', '#0000FF', '#B0C4DE', '#E0FFFF', '#006400', '#808000', '#00FF00']
    if key not in data[0].keys():
        print(f'Error, {key} not in data.')
        return

    value_to_track = data[0][key]
    keys = ['accordions', 'compression', 'decompression']
    labels = []
    keys.remove(key)

    for d in data:
        if d[key] == value_to_track:
            if len(colors) == 0:
                color = "#%06x" % random.randint(0, 0xFFFFFF)
            else:
                color = colors[0]
                colors.remove(color)

            val1 = d[keys[0]]
            val2 = d[keys[1]]
            for d2 in data:
                if d2[keys[0]] == val1 and d2[keys[1]] == val2:
                    d2['color'] = color
                    d2['label'] = ""

                    label = f'{d2[keys[0]]} {keys[0]} {d2[keys[1]]} {keys[1]}'
                    if label not in labels:
                        d2['label'] = label
                        labels.append(label)

    return data

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
