import os
import random

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from ImageAestheticsGANs.AADB.AADB import AADB, AADB_binaries
from tqdm import tqdm
from ImageAestheticsGANs.utils.utils import *
from ImageAestheticsGANs.models.ResNet18 import RegressionNetwork
from ImageAestheticsGANs.loss_functions.rank_loss import RegRankLoss
import torch.nn as nn


batch_size = 64
epochs = 200
load = False
ckpt = 'F:\Projects\Disertatie\RESULTS\ResNet_classificaiton_11_attributes\AADB_epoch_199_loss_0.7456_.pt'
lr=0.0002
beta = 0.5  # Adam

data_path = '/ImageAestheticsGANs/AADB\\'

aadb = AADB_binaries(data_path)
aadb_test = AADB_binaries(data_path, test=True)
n_classes = aadb.get_num_classes()

val_size = 500
train_size = len(aadb) - val_size

train_ds, val_ds = random_split(aadb, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
valid_dl = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)

def get_default_device():
    '''Pick GPU if available'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    '''Move tensors to chosen device'''
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True).to(torch.float32)

class DeviceDataLoader():
    def __init__(self, dl ,device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

criterion = nn.BCEWithLogitsLoss()

model = RegressionNetwork(backbone='resnet18', num_attributes=n_classes, pretrained=True)
model = model.to('cuda')

# opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta, 0.999))
opt = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)

if load:
    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt)
    last_epoch = checkpoint['epoch']

    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    loss = train_losses[-1]

    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['optimizer'])
    model.eval()

else:
    last_epoch = 0

    train_losses = []
    val_losses = []

for epoch in range(last_epoch, epochs):

    # Training Phase
    model.train()

    pbar = tqdm(enumerate(train_dl), total=len(train_dl))
    for batch, (images, labels) in pbar:

        opt.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # predicted = outputs.detach() > 0.5

        # correct = (predicted == labels.type(torch.uint8))

        # accuracy = correct.sum().item() / (len(correct) * n_classes)

        opt.step()

        # pbar.set_description("Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}".format(
        #     epoch, float(loss), float(accuracy)))
        pbar.set_description("Epoch {}, Loss: {:.4f}".format(
            epoch, float(loss)))
    train_losses.append(loss)


    # Evaluation Phase
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(enumerate(valid_dl), total=len(valid_dl))
    for batch, (images, labels) in pbar:
        with torch.no_grad():
            outputs = model(images)

            predicted = outputs > 0.5

            correct += (predicted == labels.type(torch.uint8)).sum().item()
            total += len(labels) * n_classes

    accuracy = correct / total
    val_losses.append(accuracy)
    print('Accuracy of all test images: %.3f' % (accuracy * 100))
    filename = "{}_epoch_{}_loss_{:.4f}_.pt".format('AADB', epoch, accuracy)
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
             }, os.path.join('/ImageAestheticsGANs/classifier_ckpt', filename))
    load = False

