import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from ImageAestheticsGANs.utils.utils import *


class SUN(Dataset):

    images_path = "F:\Projects\Disertatie\ImageAestheticsGANs\TheSUN\SUN2012\Images"

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
        "real"          #this is always 1 and the rest are 0
    ]

    def __init__(self):
        self.image_dir = SUN.images_path
        self.transform = T.Compose([T.Resize(size=(128, 128)),
                                    T.CenterCrop(128),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.files, self.labels = self.load_data(self.image_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
#         print("Image Name: {}".format(self.files[index].split("/")[-1]))
        image = self.transform(image)
        label = torch.from_numpy(self.labels[index])
        return image, label

    def load_data(self, image_dir):
        files = []
        labels = []
        navigator = [image_dir]

        for letter in os.listdir(navigator[-1]):
            navigator.append(letter)
            category_path = os.path.join(*navigator)
            for category in os.listdir(category_path):
                navigator.append(category)
                if category.find('.jpg') > -1:
                    files.append(os.path.join(*navigator))
                    labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                else:
                    images_path = os.path.join(*navigator)
                    for file in os.listdir(images_path):
                        navigator.append(file)
                        if file.find('.jpg') > -1:
                            files.append(os.path.join(*navigator))
                            labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                        else:
                            in_out_path = os.path.join(*navigator)
                            for image in os.listdir(in_out_path):
                                navigator.append(image)
                                files.append(os.path.join(*navigator))
                                labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                                navigator = navigator[:-1]
                        navigator = navigator[:-1]
                navigator = navigator[:-1]
            navigator = navigator[:-1]



        labels = np.asarray(labels)
        return files, labels

    def get_classes(self):
        return len(SUN.attributes)

if __name__ == "__main__":

    sun = SUN()
    show_example(*sun[0])