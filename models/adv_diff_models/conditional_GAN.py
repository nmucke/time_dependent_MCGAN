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


class UNetGenerator(nn.Module):
    def __init__(self, latent_dim=32, latent_channels=8, in_channels=1, hidden_channels=4, bilinear=True):
        super(UNetGenerator, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.bilinear = bilinear
        self.activation = nn.LeakyReLU()
        #self.sigmoid = nn.Sigmoid()

        batch_norm = True

        self.hidden_channels = [hidden_channels*2**i for i in range(5)]

        self.latent_dense = nn.Linear(latent_dim, latent_channels * 8*2)

        self.inc = DoubleConv(in_channels, self.hidden_channels[0], batch_norm=batch_norm)
        self.down1 = Down(self.hidden_channels[0], self.hidden_channels[1], batch_norm=batch_norm)
        self.down2 = Down(self.hidden_channels[1], self.hidden_channels[2], batch_norm=batch_norm)
        self.down3 = Down(self.hidden_channels[2], self.hidden_channels[3], batch_norm=batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.hidden_channels[3],
                          self.hidden_channels[4] // factor-latent_channels,
                          batch_norm=batch_norm)


        self.up1 = Up(self.hidden_channels[4],
                      self.hidden_channels[3]//factor, bilinear)
        self.up2 = Up(self.hidden_channels[3], self.hidden_channels[2] // factor, bilinear)
        self.up3 = Up(self.hidden_channels[2], self.hidden_channels[1] // factor, bilinear)
        self.up4 = Up(self.hidden_channels[1], in_channels, bilinear)
        #self.out_conv = OutConv(in_channels, in_channels)


    def forward(self, z, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        z = self.latent_dense(z)
        z = self.activation(z)
        z = z.view(z.size(0), self.latent_channels, 8, 2)

        x5 = torch.cat([x5, z], dim=1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #x = self.activation(x)
        #x = self.out_conv(x)
        return x


class UNetCritic(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=4, bilinear=False):
        super(UNetCritic, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.activation = nn.Tanh()

        batch_norm = False

        self.hidden_channels = [hidden_channels*2**i for i in range(5)]

        self.inc = DoubleConv(in_channels, self.hidden_channels[0], batch_norm=batch_norm)
        self.down1 = Down(self.hidden_channels[0], self.hidden_channels[1], batch_norm=batch_norm)
        self.down2 = Down(self.hidden_channels[1], self.hidden_channels[2], batch_norm=batch_norm)
        self.down3 = Down(self.hidden_channels[2], self.hidden_channels[3], batch_norm=batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.hidden_channels[3],
                          self.hidden_channels[4] // factor,
                          batch_norm=batch_norm)

        self.dense1 = nn.Linear((self.hidden_channels[4] // factor)* 8 * 4,
                                self.hidden_channels[4])
        self.dense2 = nn.Linear(self.hidden_channels[4], 1, bias=False)

    def forward(self, x, c):
        x = torch.cat([c, x], dim=-1)
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x


'''
class ConditionalGenerator(nn.Module):
    def __init__(self,
                 out_channels=1,
                 latent_dim=8,
                 hidden_channels=[8, 8],
                 conditional_hidden_neurons=[8, 8],
                 ):

        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels + [out_channels]
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conditional_hidden_neurons = [out_channels] + conditional_hidden_neurons

        stride_space = [2, 2, 2, 2]
        padding_space = [0, 1, 1, 0]
        out_pad_space = [0, 0, 0, 0]
        kernel_size_space = [5, 5, 5, 4]

        stride_time = [2, 2, 2, 2]
        padding_time = [0, 0, 0, 0]
        out_pad_time = [0, 0, 1, 0]
        kernel_size_time = [3, 3, 3, 2]

        self.in_layer = nn.Linear(in_features=self.latent_dim + self.latent_dim,
                                  out_features=self.hidden_channels[0] * 6)

        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        for i in range(len(self.hidden_channels)-1):
            self.conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.hidden_channels[i],
                    out_channels=self.hidden_channels[i + 1],
                    kernel_size=(kernel_size_space[i], kernel_size_time[i]),
                    stride=(stride_space[i], stride_time[i]),
                    padding=(padding_space[i], padding_time[i]),
                    output_padding=(out_pad_space[i], out_pad_time[i])
                )
            )

        for i in range(len(self.hidden_channels) - 1):
            self.batch_norm_layers.append(
                nn.BatchNorm2d(self.hidden_channels[i])
            )

        self.conditional_encoder_layers = nn.ModuleList()
        self.batch_norm_encoder_layers = nn.ModuleList()
        for i in range(len(self.conditional_hidden_neurons)-1):
            self.conditional_encoder_layers.append(
                    nn.Conv2d(
                        in_channels=self.conditional_hidden_neurons[i],
                        out_channels=self.conditional_hidden_neurons[i+1],
                        kernel_size=(kernel_size_space[i], kernel_size_time[i]),
                        stride=(stride_space[i], stride_time[i]),
                        padding=(padding_space[i], padding_time[i])
                    )
            )
            self.batch_norm_encoder_layers.append(
                nn.BatchNorm2d(self.conditional_hidden_neurons[i+1])
            )
        self.encoder_dense_layer = nn.Linear(
                in_features=self.conditional_hidden_neurons[-1] * 6,
                out_features=self.latent_dim
        )


    def forward(self, x, c):

        for i in range(len(self.conditional_hidden_neurons) - 1):
            c = self.conditional_encoder_layers[i](c)
            c = self.activation(c)
            c = self.batch_norm_encoder_layers[i](c)
        c = c.view(c.size(0), -1)
        c = self.encoder_dense_layer(c)

        x = torch.cat((x, c), dim=1)
        x = self.in_layer(x)
        x = x.view(x.size(0), self.hidden_channels[0], 6, 1)
        for i in range(len(self.hidden_channels) - 1):
            x = self.activation(x)
            x = self.batch_norm_layers[i](x)
            x = self.conv_layers[i](x)

        return x
'''
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

