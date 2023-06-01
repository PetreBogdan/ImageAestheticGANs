import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from ImageAestheticsGANs.utils.utils import *


class AADB(Dataset):

    attributes = [
        "BalacingElements",
        "ColorHarmony", # asta am pus
        "Content",
        "DoF",
        "Light",    # asta am pus
        "MotionBlur",
        "Object",
        "Repetition",
        "RuleOfThirds", # asta am pus
        "Symmetry",
        "VividColor",   # asta am pus
        # "score",
    ]

    def __init__(self, data_path, test=False):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, "ImageAesthetics_ECCV2016", "datasetImages_warp256")
        self.test=test
        self.transform = T.Compose([T.Resize(size=(128, 128)),
                                    #T.RandomCrop((64, 64)),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()])
                                    #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.files, self.labels = self.load_data(self.image_dir, self.data_path, self.test)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
#         print("Image Name: {}".format(self.files[index].split("/")[-1]))
        image = self.transform(image)
        label = torch.from_numpy(self.labels[index])
        return image, label

    def load_data(self, image_dir, data_path, test=False):
        csv_file = os.path.join(data_path, 'Dataset.csv') if not test else os.path.join(data_path, 'Dataset_test.csv')
        label_csv = pd.read_csv(csv_file, delimiter=",", usecols=AADB.attributes)
        images_csv = pd.read_csv(csv_file, delimiter=",", usecols=['ImageFile'])
        files = [os.path.join(image_dir, f) for f in images_csv['ImageFile']]
        labels = np.asarray([label.values for index, label in label_csv.iterrows()])
        return files, labels

    def normalize(self, array):
        array_norm = (array - np.min(array)) / (np.max(array) - np.min(array))
        return array_norm

    def get_num_classes(self):
        return len(AADB.attributes)

    def get_classes_histogram_graphs(self, attribute):

        csv_file = os.path.join(self.data_path, 'Dataset.csv') if not self.test else os.path.join(self.data_path, 'Dataset_test.csv')
        label_csv = pd.read_csv(csv_file, delimiter=",").drop(['ImageFile'], axis=1)

        if attribute == "all":
            fig, axs = plt.subplots(nrows=label_csv.shape[1], ncols=1, figsize=(10, 50))
            for i, col in enumerate(label_csv.columns):
                axs[i].hist(label_csv[col])
                axs[i].set_title(col)

            # Show the figure
            plt.tight_layout()
            plt.show()
        else:
            # Create a histogram of the selected column
            plt.hist(label_csv[attribute])

            # Set the title and axis labels
            plt.title(attribute)
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # Show the plot
            plt.show()

    def get_median(self, attribute, csv_file):

        values = csv_file[attribute].values

        return np.median(values)

    def get_mean(self, attribute, csv_file):

        values = csv_file[attribute].values

        return np.mean(values)

class AADB_binaries(AADB):

    binary_data = None

    def __init__(self, data_path, test=False):
        super(AADB_binaries, self).__init__(data_path, test)

    def load_data(self, image_dir, data_path, test=False):
        csv_file = os.path.join(data_path, 'Dataset.csv') if not test else os.path.join(data_path, 'Dataset_test.csv')
        label_csv = pd.read_csv(csv_file, delimiter=",", usecols=AADB.attributes)
        images_csv = pd.read_csv(csv_file, delimiter=",", usecols=['ImageFile'])
        files = [os.path.join(image_dir, f) for f in images_csv['ImageFile']]
        AADB_binaries.binary_data = self.binarize_data(label_csv)
        labels = np.asarray([label.values for index, label in self.binary_data.iterrows()])
        return files, labels

    def binarize_data(self, csv_file):

        for column in csv_file:
            # threshold = self.get_median(column, csv_file)
            threshold = self.get_mean(column, csv_file)
            bin_values = [0 if val <= threshold else 1 for val in csv_file[column].values]
            csv_file[column] = bin_values

        return csv_file

    def get_percent_binary_data(self, attribute=None):

        if attribute == 'all' or attribute is None:
            for column in  AADB_binaries.binary_data:
                print(f"{column}: {np.sum(self.binary_data[column].values) / len(self.binary_data[column].values)}")

        else:
            print(f"{attribute}: {np.sum(self.binary_data[attribute].values) / len(self.binary_data[attribute].values)}")

if __name__ == "__main__":


    data_path = "F:\Projects\Disertatie\ImageAestheticsGANs\AADB\\"
    aadb = AADB_binaries(data_path, test=False)
    # aadb.get_classes_histogram_graphs("all")
    #
    # show_example(*aadb[0])
    # plt.show()

    aadb.get_percent_binary_data()



