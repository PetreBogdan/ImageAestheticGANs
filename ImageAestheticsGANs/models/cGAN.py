import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # 自动取得batch维
        return x.view((x.size(0),) + self.shape)
        # 若使用下式，batch的维数只能用-1代指
        # return x.view(self.shape)


class Generator(nn.Module):
    def __init__(self, ngf, n_classes, image_size, n_channels, latent_dim):
        super(Generator, self).__init__()
        self.latent_class_dim = 10  # 包含分类信息的噪声维数
        self.ngf = ngf
        self.latent_dim = latent_dim
        self.latent_class_dim = n_classes
        # 如果输入c为1维int型,升到2维到且第2维为latent_class_dim
        # self.emb = nn.Embedding(n_classes, self.latent_class_dim)
        # 如果输入c为one-hot，第2维扩张到latent_class_dim
        self.exp = nn.Linear(n_classes, self.latent_class_dim)
        self.main = nn.Sequential(

            nn.Linear(latent_dim + self.latent_class_dim, ngf * 8 * (image_size // 16) ** 2),
            Reshape(ngf * 8, image_size // 16, image_size // 16),
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x (image_size//8) x (image_size//8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x (image_size//4) x (image_size//4)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x (image_size//2) x (image_size//2)

            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x image_size x image_size
        )

    def forward(self, z, c):
        # c为一维int型
        # cat = torch.cat([z, self.emb(c)], 1)
        # c为one-hot
        cat = torch.cat((z, self.exp(c)), 1)
        return self.main(cat)


class Discriminator(nn.Module):
    def __init__(self, ndf, n_channels, image_size, n_classes):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.image_size = image_size
        # self.emb = nn.Embedding(n_classes, image_size * image_size)
        self.exp = nn.Linear(n_classes, image_size * image_size)
        self.main = nn.Sequential(
            # input is (nc) x image_size x image_size
            nn.Conv2d(n_channels + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x (image_size//2) x (image_size//2)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x (image_size//4) x (image_size//4)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x (image_size//8) x (image_size//8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x (image_size//16) x (image_size//16)

            # Reshape(-1, ndf * 8 * (image_size // 16) ** 2),
            Reshape(ndf * 8 * (image_size // 16) ** 2),
            nn.Linear(ndf * 8 * (image_size // 16) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, c):
        output = self.exp(c)
        # output = self.emb(c)
        output = output.view(c.size(0), 1, self.image_size, self.image_size)
        output = torch.cat((img, output), 1)
        output = self.main(output)
        return output