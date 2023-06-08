import numpy as np


def process_data_a9a(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Split the line by whitespace
            elements = line.split()
            try:
                label = int(elements[0])
            except:
                continue
            features = [int(feature.split(':')[0]) for feature in elements[1:]]
            #For some reason, the number of features is not consistent...
            if len(features) != 14:
                continue
            data.append([label] + features)
    return data