import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


class AADB(Dataset):

    attributes = [
        "balancing_elements",
        "color_harmony",
        "content",
        "depth_of_field",
        "light",
        "motion_blur",
        "object",
        "repetition",
        "rule_of_thirds",
        "symmetry",
        "vivid_color",
    ]

    def __init__(self, image_dir, label_csv_path, test=False, drop_score=False):
        self.label_csv_path = label_csv_path
        self.image_dir = image_dir
        self.test=test
        self.drop_score = drop_score
        self.transform = T.Compose([T.Resize(size=(256, 256)),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()])
        self.files, self.labels = self.load_data(self.image_dir, self.label_csv_path, self.test)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
#         print("Image Name: {}".format(self.files[index].split("/")[-1]))
        image = self.transform(image)
        label = torch.from_numpy(self.labels[index])
        return image, label

    def load_data(self, image_dir, csv_path, test=False):
        csv_file = csv_path + 'Dataset.csv' if not test else csv_path + 'Dataset_test.csv'
        label_csv = pd.read_csv(csv_file, delimiter=",")
        files = [os.path.join(image_dir, f) for f in label_csv['ImageFile']]
        if self.drop_score:
            labels = np.asarray([label.values for index, label in label_csv.drop(['ImageFile', 'score'], axis=1).iterrows()])
        else:
            labels = []
            for index, label in label_csv.drop(['ImageFile'], axis=1).iterrows():  # this is for moving score to the last value
                label = list(label.values)
                label.append(label.pop(9))
                labels.append(label)

            labels = np.asarray(labels)
        return files, labels


    def normalize(self, array):
        array_norm = (array - np.min(array)) / (np.max(array) - np.min(array))
        return array_norm

    def get_classes(self):
        return len(AADB.attributes)

    def get_classes_histogram_graphs(self, attribute):

        csv_file = self.label_csv_path + 'Dataset.csv' if not self.test else self.label_csv_path + 'Dataset_test.csv'
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

    def get_median(self, attribute):

        csv_file = self.label_csv_path + 'Dataset.csv' if not self.test else self.label_csv_path + 'Dataset_test.csv'
        label_csv = pd.read_csv(csv_file, delimiter=",").drop(['ImageFile'], axis=1)

        values = label_csv[attribute].values

        return np.mean(values)

class AADB_classes(AADB):

    def __init__(self, image_dir, label_csv_path, test=False):
        super(AADB_classes, self).__init__(image_dir, label_csv_path, test)

    def load_data(self, image_dir, csv_path, test=False):
        csv_file = csv_path + 'Dataset.csv' if not test else csv_path + 'Dataset_test.csv'
        label_csv = pd.read_csv(csv_file, delimiter=",")
        files = [os.path.join(image_dir, f) for f in label_csv['ImageFile']]
        label_csv_normed = self.prepare_csv_file(label_csv)     # this is for norming values
        label_csv_rounded = self.round_labels(label_csv_normed)
        labels = np.asarray([label.values for index, label in label_csv_rounded.iterrows()])
        return files, labels

    def prepare_csv_file(self, csv_file):
        """
        Moves the score column to the last and normalize all the values between [0, 1]
        :param csv_file:
        :return: normalized csv_file
        """
        columns = csv_file.columns.tolist()
        columns.pop(columns.index('score'))
        columns.append('score')
        csv_file = csv_file.reindex(columns=columns)

        for column in csv_file.drop(['ImageFile'], axis=1):
            normed_values = self.normalize(csv_file[column].values)
            csv_file[column] = normed_values

        return csv_file.drop(['ImageFile', 'score'], axis=1)

    def round_labels(self, csv_file):

        for index, label in csv_file.iterrows():
            for value in csv_file.columns:
                if np.logical_and(label[value] >= 0, label[value] <0.2):
                    label[value] = 0
                elif np.logical_and(label[value] >= 0.2, label[value] <0.4):
                    label[value] = 1
                elif np.logical_and(label[value] >= 0.4, label[value] <0.6):
                    label[value] = 2
                elif np.logical_and(label[value] >= 0.6, label[value] <0.8):
                    label[value] = 3
                elif np.logical_and(label[value] >= 0.8, label[value] <=1):
                    label[value] = 4

        return csv_file

    @staticmethod
    def get_classes():
        return len(AADB_classes.attributes)

class AADB_binaries(AADB):

    def __init__(self, image_dir, label_csv_path, test=False):
        super(AADB_binaries, self).__init__(image_dir, label_csv_path, test)

    def load_data(self, image_dir, csv_path, test=False):
        csv_file = csv_path + 'Dataset.csv' if not test else csv_path + 'Dataset_test.csv'
        label_csv = pd.read_csv(csv_file, delimiter=",")
        files = [os.path.join(image_dir, f) for f in label_csv['ImageFile']]
        label_csv_binaries = self.binarize_data(label_csv)     # this is for norming values
        labels = np.asarray([label.values for index, label in label_csv_binaries.iterrows()])
        return files, labels

    # def binarize_data(self, csv_file):



if __name__ == "__main__":


    image_dir = "F:\Projects\Disertatie\ImageAestheticsGANs\AADB\ImageAesthetics_ECCV2016\datasetImages_warp256"
    csv_path = "F:\Projects\Disertatie\ImageAestheticsGANs\AADB\\"
    aadb = AADB(image_dir, csv_path, test=False)

    aadb.get_classes_histogram_graphs('BalacingElements')
    print(aadb.get_median('BalacingElements'))