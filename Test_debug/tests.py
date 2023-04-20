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

csv_file = "F:\Projects\Disertatie\ImageAestheticsGANs\AADB\Dataset.csv"
label_csv = pd.read_csv(csv_file, delimiter=",").drop(['ImageFile', 'score'], axis=1)
# print(label_csv.head())

for index, label in label_csv.drop(['ImageFile'], axis=1).iterrows():  # this is for moving score to the last value
    label = list(label.values)
    label.append(label.pop(9))
    labels.append(label)
