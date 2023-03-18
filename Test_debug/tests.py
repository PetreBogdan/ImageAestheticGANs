import pandas as pd
import numpy as np


def normalize(array):
    array_norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array_norm

csv_file = "F:\Projects\Disertatie\Dataset.csv"
label_csv = pd.read_csv(csv_file, delimiter=",").drop(['ImageFile', 'score'], axis=1)
print(label_csv.head())

# for index, value in label_csv.iterrows():
#     print(value)
#     for item in value:
#         print(item)
#
#     break