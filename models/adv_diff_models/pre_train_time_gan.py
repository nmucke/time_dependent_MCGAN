import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch


import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
                nn.LeakyReLU(),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
                nn.LeakyReLU(),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=5,
                              padding=2, bias=False),
                    nn.LeakyReLU(),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=5,
                              padding=2, bias=False),
                    nn.LeakyReLU()
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels, batch_norm=batch_norm)
            nn.Conv2d(in_channels, out_channels, kernel_size=5,
                      padding=2, bias=False),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=32, in_channels=1, hidden_channels=2, bilinear=False):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.bilinear = bilinear
        self.activation = nn.LeakyReLU()
        #self.sigmoid = nn.Sigmoid()

        batch_norm = True

        self.hidden_channels = [hidden_channels*2**i for i in range(5)]


        self.inc = DoubleConv(in_channels, self.hidden_channels[0], batch_norm=batch_norm)
        self.down1 = Down(self.hidden_channels[0], self.hidden_channels[1], batch_norm=batch_norm)
        self.down2 = Down(self.hidden_channels[1], self.hidden_channels[2], batch_norm=batch_norm)
        self.down3 = Down(self.hidden_channels[2], self.hidden_channels[3], batch_norm=batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.hidden_channels[3],
                          self.hidden_channels[4] // factor,
                          batch_norm=batch_norm)
        self.out_dense = nn.Linear(self.hidden_channels[4]*8*2, self.latent_dim)



    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.out_dense(x)
        return x

class ForecastingNet(nn.Module):
    def __init__(self, latent_dim=32, in_channels=1, hidden_channels=2, bilinear=False):
        super(ForecastingNet, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.latent_channels = latent_dim
        self.bilinear = bilinear
        self.activation = nn.LeakyReLU()

        self.encoder = Encoder(
                latent_dim=latent_dim,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                bilinear=bilinear
        )

        self.out_dense_1 = nn.Linear(latent_dim, latent_dim, bias=True)
        self.out_conv_1 = nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=latent_dim//2,
                kernel_size=8,
                stride=4,
                bias=True
        )
        self.out_conv_2 = nn.ConvTranspose1d(
                in_channels=latent_dim//2,
                out_channels=latent_dim//4,
                kernel_size=8,
                stride=4,
                bias=True,
                padding=2
        )
        self.out_conv_3 = nn.ConvTranspose1d(
                in_channels=latent_dim//4,
                out_channels=1,
                kernel_size=8,
                stride=4,
                bias=False,
                padding=2
        )
        #self.out_dense_2 = nn.Linear(64, 128, bias=False)

    def forward(self, x, return_input=False):
        x_pred = self.encoder(x)
        x_pred = self.activation(x_pred)
        x_pred = self.out_dense_1(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = x_pred.view(x.size(0), self.latent_dim, 1)
        x_pred = self.out_conv_1(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = self.out_conv_2(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = self.out_conv_3(x_pred)
        x_pred = x_pred.transpose(1, 2)
        x_pred = x_pred.unsqueeze(1)
        x_pred += x[:, :, :, -1:]

        #x_pred = self.out_dense_2(x_pred)
        if return_input:
            return torch.cat((x, x_pred), dim=-1)
        else:
            return x_pred


class ConditionalGenerator(nn.Module):
    def __init__(self,
                 z_latent_dim=32,
                 in_channels=1,
                 hidden_channels=2,
                 encoder=None,
                 bilinear=False
                 ):
        super(ConditionalGenerator, self).__init__()
        self.in_channels = in_channels
        self.z_latent_dim = z_latent_dim
        self.bilinear = bilinear
        self.activation = nn.Sigmoid()

        self.encoder = encoder
        self.conditional_latent_dim = self.encoder.latent_dim

        self.full_latent_dim = self.conditional_latent_dim + self.z_latent_dim

        self.z_latent_dense = nn.Linear(self.z_latent_dim, self.z_latent_dim)

        self.out_dense_1 = nn.Linear(
                in_features=self.full_latent_dim,
                #out_features=self.full_latent_dim,
                out_features=64,
                bias=True)
        self.out_dense_2 = nn.Linear(
                in_features=64,
                #out_features=self.full_latent_dim,
                out_features=64,
                bias=True)

        self.out_dense_3 = nn.Linear(
                in_features=64,
                #out_features=self.full_latent_dim,
                out_features=128,
                bias=False)
        '''
        self.out_conv_1 = nn.ConvTranspose1d(
                in_channels=self.full_latent_dim,
                out_channels=self.full_latent_dim//2,
                kernel_size=8,
                stride=4,
                bias=True
        )
        self.out_conv_2 = nn.ConvTranspose1d(
                in_channels=self.full_latent_dim//2,
                out_channels=self.full_latent_dim//4,
                kernel_size=8,
                stride=4,
                bias=True,
                padding=2
        )
        self.out_conv_3 = nn.ConvTranspose1d(
                in_channels=self.full_latent_dim//4,
                out_channels=1,
                kernel_size=8,
                stride=4,
                bias=False,
                padding=2
        )
        '''
        #self.out_dense_2 = nn.Linear(64, 128, bias=False)

    def forward(self, z, x, return_input=False):
        x_pred = self.encoder(x)
        x_pred = self.activation(x_pred)

        z = self.z_latent_dense(z)
        z = self.activation(z)

        x_pred = torch.cat((x_pred, z), dim=-1)
        x_pred = self.out_dense_1(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = self.out_dense_2(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = self.out_dense_3(x_pred)

        '''
        x_pred = self.activation(x_pred)
        x_pred = x_pred.view(x_pred.size(0), self.full_latent_dim, 1)
        x_pred = self.out_conv_1(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = self.out_conv_2(x_pred)
        x_pred = self.activation(x_pred)
        x_pred = self.out_conv_3(x_pred)
        '''
        x_pred = x_pred.view(x_pred.size(0), 128, 1)
        x_pred = x_pred.transpose(1, 2)
        x_pred = x_pred.unsqueeze(1)
        x_pred = x_pred.transpose(-2, -1)
        #x_pred += x_pred[:, :, :, -1:]

        if return_input:
            return torch.cat((x, x_pred), dim=-1)
        else:
            return x_pred



class Critic(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=4, encoder=None, bilinear=False):
        super(Critic, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.activation = nn.Sigmoid()

        batch_norm = False

        self.hidden_channels = [hidden_channels*2**i for i in range(5)]

        '''
        self.inc = DoubleConv(in_channels, self.hidden_channels[0], batch_norm=batch_norm)
        self.down1 = Down(self.hidden_channels[0], self.hidden_channels[1], batch_norm=batch_norm)
        self.down2 = Down(self.hidden_channels[1], self.hidden_channels[2], batch_norm=batch_norm)
        self.down3 = Down(self.hidden_channels[2], self.hidden_channels[3], batch_norm=batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.hidden_channels[3],
                          self.hidden_channels[4] // factor,
                          batch_norm=batch_norm)
        '''

        self.encoder = encoder

        self.dense1 = nn.Linear(16,#(self.hidden_channels[4] // 1)* 8 * 2,
                                self.hidden_channels[4])
        self.dense2 = nn.Linear(self.hidden_channels[4],#(self.hidden_channels[4] // 1)* 8 * 2,
                                self.hidden_channels[3])
        self.dense3 = nn.Linear(self.hidden_channels[3],#(self.hidden_channels[4] // 1)* 8 * 2,
                                self.hidden_channels[2])
        self.dense4 = nn.Linear(self.hidden_channels[2], 1, bias=False)

    def forward(self, x):
        #x = self.inc(x)
        #x = self.down1(x)
        #x = self.down2(x)
        #x = self.down3(x)
        #x = self.down4(x)
        x = self.encoder(x)

        #x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.activation(x)
        x = self.dense4(x)
        return x


class ConditionalCritic(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=[]):
        super(ConditionalCritic, self).__init__()

        self.hidden_channels = [in_channels] + hidden_channels
        self.activation = nn.LeakyReLU()
        self.kernel_size = 5
        self.stride = 2

        self.conv_layers = nn.ModuleList()
        for i in range(len(self.hidden_channels) - 1):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=self.hidden_channels[i],
                    out_channels=self.hidden_channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=self.stride
                )
            )

        self.dense_layer = nn.Linear(in_features=self.hidden_channels[-1] * 5,
                                     out_features=self.hidden_channels[-1])
        self.out_layer = nn.Linear(in_features=self.hidden_channels[-1],
                                   out_features=1,
                                   bias=False)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        for i in range(len(self.hidden_channels) - 1):
            x = self.activation(x)
            x = self.conv_layers[i](x)
        x = x.view(x.size(0), -1)
        x = self.dense_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)

        return x


if __name__ == '__main__':


    def out_size(in_size, stride, padding, kernel_size, out_pad):
        return (in_size-1)*stride-2*padding+1*(kernel_size-1)+out_pad+1

    stride = [2, 2, 2, 2]
    padding = [0, 0, 0, 0]
    out_pad = [0, 0, 1, 0]
    kernel_size = [3, 3, 3, 2]

    in_size = 1
    for i in range(len(stride)):
        in_size = out_size(in_size, stride[i], padding[i], kernel_size[i],
                           out_pad[i])
        print(in_size)

