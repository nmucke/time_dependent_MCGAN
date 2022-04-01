import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math
import torch.nn.init as init

class DilatedCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        kernel_size=4,
        dilation_factor=2
    ):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             dilation=dilation_factor,
                                             padding=0)
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1)
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.dilated_causal_conv(x)
        x1 = self.leaky_relu(x1)
        x2 = x[:, :, self.dilation_factor:]
        x2 = self.skip_connection(x2)
        return x1 + x2

class CCNN(nn.Module):
    def __init__(
        self,
        num_channels=2,
        hidden_channels=[5, 5, 5],
        kernel_size=5
    ):
        super(CCNN, self).__init__()
        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factors = [2 ** i for i in range(len(hidden_channels))]

        self.in_channels = [num_channels] + [channels for channels in hidden_channels]


        self.dilated_causal_convs = nn.ModuleList(
            [DilatedCausalConv1d(
                in_channels=self.in_channels[i],
                out_channels=hidden_channels[i],
                kernel_size=kernel_size,
                dilation_factor=self.dilation_factors[i]
            ) for i in range(len(hidden_channels))]
        )


        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],
                                      out_channels=num_channels,
                                      kernel_size=1,
                                      bias=False)
        #self.output_layer.apply(weights_init)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()

        self.inference = False


    def forward(self, inp):
        x = inp.transpose(0,1).transpose(1,2)
        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        x = x.transpose(1, 2).transpose(0, 1)
        if self.inference:
            return torch.cat([inp[1:], x], dim=0)
        else:
            return x
