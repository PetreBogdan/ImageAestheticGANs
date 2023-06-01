import pandas as pd
import numpy as np


def normalize(array):
    array_norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array_norm

def get_median(attribute, csv, test=False):

    # csv_file = csv + 'Dataset.csv' if not test else csv + 'Dataset_test.csv'
    label_csv = pd.read_csv(csv_file, delimiter=",").drop(['ImageFile'], axis=1)

    values = label_csv[attribute].values

    return np.mean(values)

attributes = [
        "BalacingElements",
        "ColorHarmony",
        "Content",
        "DoF",
        "Light",
        "MotionBlur",
        "Object",
        "Repetition",
        "RuleOfThirds",
        "Symmetry",
        "VividColor",
        "score",
        # "real"          # this is always 0
    ]

# csv_file = "F:\Projects\Disertatie\ImageAestheticsGANs\AADB\Dataset.csv"
# label_csv = pd.read_csv(csv_file, delimiter=",", usecols=attributes)
# # print(label_csv.head())
#
# for column in label_csv:
#     print(column)

path = "F:\Projects\Disertatie\ImageAestheticsGANs\AVA_dataset\style_image_lists\\test.multilab"
with open(path, 'r') as file:
    labels = []
    with open(path, 'r') as l:
        for line in l:
            print(line)
            label = line.split()
            print(label)
            labels.append([int(att) for att in label])

print(labels)
