import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from ImageAestheticsGANs.utils.utils import *


class AVA(Dataset):

    attributes = [
        "Complementary_Colors",
        "Duotones",
        "HDR",
        "Image_Grain",
        "Light_On_White",
        "Long_Exposure",
        "Macro",
        "Motion_Blur",
        "Negative_Image",
        "Rule_of_Thirds",
        "Shallow_DOF",
        "Silhouettes",
        "Soft_Focus",
        "Vanishing_Point",
        # "score",
    ]

    def __init__(self, data_path, test=False):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, "images", "images")
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
        files = os.path.join(data_path, "style_image_lists", "train.jpgl") if not test else os.path.join(data_path, "style_image_lists", "test.jpgl")
        labels_file = os.path.join(data_path, "style_image_lists", "train.lab") if not test else os.path.join(data_path, "style_image_lists", "test.multilab")
        with open(files, 'r') as f:
            files_paths = []
            for line in f:
                line = line.rstrip('\n')
                files_paths.append(os.path.join(image_dir, f"{line}.jpg"))
        if not self.test:
            labels = []
            with open(labels_file, 'r') as l:
                for line in l:
                    label = np.zeros(self.get_num_classes())
                    label[int(line) - 1] = 1
                    labels.append(label)
        else:
            labels = []
            with open(labels_file, 'r') as l:
                for line in l:
                    label = line.split()
                    labels.append([int(att) for att in label])

        labels = np.asarray(labels)
        return files_paths, labels

    def get_num_classes(self):
        return len(AVA.attributes)

if __name__ == "__main__":

    data_path = "F:\Projects\Disertatie\ImageAestheticsGANs\AVA_dataset"
    ava = AVA(data_path, test=False)

    show_example(*ava[1])
    plt.show()