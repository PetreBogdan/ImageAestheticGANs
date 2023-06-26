import os
import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader, random_split

from ImageAestheticsGANs.AADB.AADB import AADB_binaries
from tqdm import tqdm
from ImageAestheticsGANs.models.ResNet18 import RegressionNetwork
import torch.nn as nn
from ImageAestheticsGANs.loss_functions.focal_loss import FocalLoss
import argparse

parser = argparse.ArgumentParser(description="Arguments for training loop")
parser.add_argument('--batch_size', type=int, help="Number of batches")
parser.add_argument('--epochs', type=int, default=200, help="Number of epochs")
parser.add_argument('--image_size', type=int,default=64, help="Image dimensions")
parser.add_argument('--load', type=bool, default=False, help="Loading model?")
parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate")
parser.add_argument('--ckpt', type=str, help="Checkpoint for loading")
parser.add_argument('--beta', type=float, default=0.5, help="Beta for Adam optimizer")
parser.add_argument('--optim', type=str, default='sgd', help="Optimizer for the algorithm (adam/sgd)")
parser.add_argument('--criterion', type=str, default='bcelogits', help="Loss function (cross/bcelogits/focal)")
parser.add_argument('--results', type=str, help="Results folder")
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
load = args.load
ckpt = args.ckpt
lr = args.lr
beta = args.beta
image_size = args.image_size

data_path = 'F:\Projects\Disertatie\ImageAestheticsGANs\AADB'

aadb = AADB_binaries(data_path, image_size)
aadb_test = AADB_binaries(data_path, image_size, test=True)
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
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True).to(torch.float32)


class DeviceDataLoader():
    def __init__(self, dl, device):
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

if args.criterion == "bcelogits":
    criterion = nn.BCEWithLogitsLoss()
elif args.criterion == "focal":
    criterion = FocalLoss()
elif args.criterion == "cross":
    criterion = nn.CrossEntropyLoss()

model = RegressionNetwork(backbone='resnet18', num_attributes=n_classes, pretrained=True)
model = model.to('cuda')

if args.optim == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta, 0.999))
elif args.optim == 'sgd':
    opt = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)

if load:
    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt)
    last_epoch = checkpoint['epoch'] + 1

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
            val_loss = loss = criterion(outputs, labels)

            predicted = outputs > 0.5

            correct += (predicted == labels.type(torch.uint8)).sum().item()
            total += len(labels) * n_classes

    accuracy = correct / total
    val_losses.append(val_loss)
    print('Accuracy of all test images: %.3f' % (accuracy * 100))
    if epoch % 10 == 0:
        filename = "{}_epoch_{}_accuracy_{:.4f}_.pt".format('AADB', epoch, accuracy)
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses
                    }, os.path.join(args.results, filename))
    load = False
