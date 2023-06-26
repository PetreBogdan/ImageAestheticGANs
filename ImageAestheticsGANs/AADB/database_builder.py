import time
import os
import pandas as pd

start_time = time.time()

label_folder = "ImageAesthetics_ECCV2016/imgListFiles_label/"
datasetDict_train = {}
dataset_train = []

datasetDict_test = {}
dataset_test = []

files = os.listdir(label_folder)
for file in files[2:]:
    if file.find("Train") > -1 or file.find("Validation") > -1:
        with open(label_folder + file, 'r') as f:
            for line in f.readlines():
                imageDict = {}
                imageDict[file.split("_")[1].replace('.txt', '')] = line.split()[1]

                if line.split()[0] in datasetDict_train.keys():
                    datasetDict_train[line.split()[0]].update(imageDict)
                else:
                    datasetDict_train[line.split()[0]] = imageDict

    elif file.find("TestNew") > -1:
        with open(label_folder + file, 'r') as f:
            for line in f.readlines():
                imageDict = {}
                imageDict[file.split("_")[1].replace('.txt', '')] = line.split()[1]

                if line.split()[0] in datasetDict_test.keys():
                    datasetDict_test[line.split()[0]].update(imageDict)
                else:
                    datasetDict_test[line.split()[0]] = imageDict

for key in datasetDict_train:
    new_dict = {}
    new_dict['ImageFile'] = key
    new_dict.update(datasetDict_train[key])
    dataset_train.append(new_dict)

for key in datasetDict_test:
    new_dict = {}
    new_dict['ImageFile'] = key
    new_dict.update(datasetDict_test[key])
    dataset_test.append(new_dict)


df = pd.DataFrame.from_dict(dataset_train)
df.to_csv(r'Dataset.csv', index = False, header=True)

df = pd.DataFrame.from_dict(dataset_test)
df.to_csv(r'Dataset_test.csv', index = False, header=True)

print("---Execution Time is %s seconds ---" % (time.time() -start_time))
