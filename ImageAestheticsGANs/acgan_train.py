import os
import sys
sys.path.append(".")

from AADB.AADB import AADB_binaries
import torch
from torch.utils.data import  DataLoader
import torchvision.utils as vutils
import torch.optim as optim
import torch.autograd as autograd
from models.ACGAN import Generator, Discriminator
from utils.utils import *
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description="Arguments for training loop")
parser.add_argument('--batch_size', type=int, help="Number of batches")
parser.add_argument('--max_epochs', type=int, default=200, help="Number of epochs")
parser.add_argument('--latent_dim', type=int, default=128, help="Latent dimension")
parser.add_argument('--n_channels', type=int, default=3, help="Number of channels")
parser.add_argument('--image_size', type=int,default=64, help="Image dimensions")
parser.add_argument('--ngf', type=int, default=128, help="Feature map for generator")
parser.add_argument('--ndf', type=int, default=128, help="Feature map for discriminator")
parser.add_argument('--beta', type=float, default=0.5, help="Beta for Adam optimizer")
parser.add_argument('--lrg', type=float, default=0.0002, help="Learning rate of the generator")
parser.add_argument('--lrd', type=float, default=0.0002, help="Learning rate for the discriminator")
parser.add_argument('--is_load', type=bool, default=False, help="Load model?")
parser.add_argument('--ckpt_path', type=str, help="Checkpoint for loading")
parser.add_argument('--n_critic', type=int, default=5, help="Iterations of the critic")
parser.add_argument('--lam_gp', type=int, default=10, help="Lambda Gradient Penalty")
parser.add_argument('--sample_path', type=str, help="Results folder")

args = parser.parse_args()


batch_size = args.batch_size
max_epochs = args.max_epochs
latent_dim = args.latent_dim
n_channels = args.n_channels
image_size = args.image_size
ngf = args.ngf   # feature map gen
ndf = args.ndf   # feature map disc
beta = args.beta  # Adam
lrg = args.lrg  # Learning rate for optimizers
lrd = args.lrd
is_load = args.is_load
ckpt_path = args.ckpt_path
n_critic = args.n_critic
lam_gp = args.lam_gp
data_path = 'F:\Projects\Disertatie\ImageAestheticsGANs\AADB\\'
samples_path = args.sample_path
os.makedirs(samples_path, exist_ok=True)

aadb = AADB_binaries(data_path, image_size)
aadb_test = AADB_binaries(data_path, image_size, test=True)
n_classes = aadb.get_num_classes()
aadb = aadb + aadb_test

train_dl = DataLoader(aadb, batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

n_sample = 64
sample_noise = torch.randn(n_sample, latent_dim, device=device)
sample_labels = torch.randint(2, (n_sample, n_classes), dtype=torch.float32, device=device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

net_G = Generator(ngf, n_classes, image_size, n_channels, latent_dim).to(device)
net_D = Discriminator(ndf, n_channels, image_size, n_classes).to(device)

net_G.apply(weights_init)
net_D.apply(weights_init)

def MBCE(input, target, esp=1e-19):
    loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
        (1 - target) * torch.log((1 - input).clamp_min(esp)))
    return loss

optimizer_D = optim.Adam(net_D.parameters(), lr=lrd, betas=(beta, 0.999))
optimizer_G = optim.Adam(net_G.parameters(), lr=lrg, betas=(beta, 0.999))

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device=device)
    z = x + alpha * (y - x)

    z.requires_grad = True
    o = f(z)[0]
    ones = torch.ones(o.size(), device=device)
    g = autograd.grad(o, z, grad_outputs=ones, create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp

def save_diagram_loss(list_1, list_2,
                 label1="D", label2="G",
                 title="Generator and Discriminator loss During Training",
                 x_label="iterations", y_label="Loss",
                 path=samples_path,
                 name='loss.jpg'
                 ):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(list_1, label=label1)
    plt.plot(list_2, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(path, name))
    plt.close()

if is_load:
    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt_path)
    last_epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    last_i = checkpoint['last_current_iteration']
    sample_noise = checkpoint['sample_noise']

    list_loss_D = checkpoint['list_loss_D']
    list_loss_G = checkpoint['list_loss_G']

    loss_D = list_loss_D[-1]
    loss_G = list_loss_G[-1]

    net_D.load_state_dict(checkpoint['netD_state_dict'])
    net_G.load_state_dict(checkpoint['netG_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    # net_D.eval()
    # net_G.eval()

else:
    last_epoch = 0
    iteration = 0

    list_loss_G = []
    list_loss_D = []

print("Starting Training Loop...")
for epoch in range(last_epoch, max_epochs):
    str_i = 0
    if is_load:
        str_i = last_i
    for i, (real_img, real_c) in enumerate(train_dl, str_i):

        # -----------------------------------------------------------
        # Initial batch
        real_img, real_c = real_img.to(device), real_c.to(device)
        real_batch_size = real_img.size(0)
        noise = torch.randn(real_batch_size, latent_dim, device=device)
        # random label
        fake_c = torch.randint(2, (real_batch_size, n_classes), dtype=torch.float32, device=device)
        fake_img = net_G(noise, fake_c)

        # -----------------------------------------------------------
        # Update D network: minimize: -(D(x) - D(G(z)))+ lambda_gp * gp + class_loss (gradient penalty)
        net_D.zero_grad()

        v, c = net_D(real_img)
        loss_real = (- torch.mean(v) + MBCE(c, real_c)) * 0.5
        v, c = net_D(fake_img.detach())
        loss_fake = (torch.mean(v) + MBCE(c, fake_c)) * 0.5
        gp = gradient_penalty(real_img.detach(), fake_img.detach(), net_D)
        loss_D = (loss_real + loss_fake) * 0.5 + lam_gp * gp  # total loss of D

        # Update D
        loss_D.backward()
        optimizer_D.step()

        # -----------------------------------------------------------
        # Update G network: maximize D(G(z)) , equal to minimize - D(G(z))
        if i % n_critic == 0:
            net_G.zero_grad()

            # Calculate G loss
            v, c = net_D(fake_img)

            loss_G = (- torch.mean(v) + MBCE(c, fake_c)) * 0.5

            # Update G
            loss_G.backward()
            optimizer_G.step()

        # -----------------------------------------------------------
        # Output training stats
        with torch.no_grad():
            list_loss_D.append(loss_D.item())

            if type(loss_G) == float:
                list_loss_G.append(loss_G)
            else:
                list_loss_G.append(loss_G.item())

            if i % 100 == 0: # batches
                print(
                    '[%d/%d][%2d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch, max_epochs, i, len(train_dl),
                       list_loss_D[-1], list_loss_G[-1]))

            # Output sample noise
            if (iteration % 500 == 0) or ((epoch == max_epochs - 1) and (i == len(train_dl) - 1)):
                samples = net_G(sample_noise, sample_labels).cpu()
                vutils.save_image(samples, os.path.join(samples_path, '%d.jpg' % iteration), padding=2, normalize=True)
                save_diagram_loss(list_loss_D, list_loss_G, name='loss.jpg')

            # Save model after 5000 iterations
            if (iteration % 5000 == 0) or ((epoch == max_epochs - 1) and (i == len(train_dl) - 1)):
                save_path = os.path.join(samples_path, 'checkpoint_iteration_%d.tar' % iteration)
                torch.save({
                    'epoch': epoch,
                    'iteration': iteration,
                    'last_current_iteration': i,
                    'netD_state_dict': net_D.state_dict(),
                    'netG_state_dict': net_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'list_loss_D': list_loss_D,
                    'list_loss_G': list_loss_G,
                    'sample_noise': sample_noise
                }, save_path)

        # iteration: total iteration, i: iteration of current epoch
        iteration += 1
        is_load = False
