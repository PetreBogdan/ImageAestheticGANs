import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)

class Generator(nn.Module):
    def __init__(self, ngf, n_classes, image_size, n_channels, latent_dim):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.latent_dim = latent_dim
        self.latent_class_dim = 28
        self.exp = nn.Linear(n_classes, self.latent_class_dim)
        self.main = nn.Sequential(

            nn.Linear(self.latent_dim + self.latent_class_dim, self.ngf * 8 * (image_size // 16) ** 2),
            Reshape(self.ngf * 8, image_size // 16, image_size // 16),
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x (image_size//8) x (image_size//8)

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x (image_size//4) x (image_size//4)

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # (ngf) x (image_size//2) x (image_size//2)

            nn.ConvTranspose2d(self.ngf, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x image_size x image_size
        )

    def forward(self, z, c):
        return self.main(torch.cat((z, self.exp(c)), 1))


class Discriminator(nn.Module):
    def __init__(self, ndf, n_channels, image_size, n_classes):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            # (nc) x image_size x image_size
            nn.Conv2d(n_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x (image_size//2) x (image_size//2)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 2, image_size // 4, image_size // 4]),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x (image_size//4) x (image_size//4)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 4, image_size // 8, image_size // 8]),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x (image_size//8) x (image_size//8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 8, image_size // 16, image_size // 16]),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x (image_size//16) x (image_size//16)
            Reshape(ndf * 8 * (image_size // 16) ** 2),
        )

        self.adv = nn.Sequential(
            nn.Linear(ndf * 8 * (image_size // 16) ** 2, 1),
        )

        self.aux = nn.Sequential(
            nn.Linear(ndf * 8 * (image_size // 16) ** 2, n_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        feature = self.main(input)
        v = self.adv(feature)
        c = self.aux(feature)
        return v, c
