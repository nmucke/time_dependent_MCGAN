import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math
import torch.nn.init as init


def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)

class ParGenerator(nn.Module):
    def __init__(self, par_latent_dim=2, par_dim=2, par_hidden_neurons=[8, 8]):
        super(ParGenerator, self).__init__()

        self.par_latent_dim = par_latent_dim
        self.par_hidden_neurons = par_hidden_neurons
        self.par_dim = par_dim
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(in_features=self.par_latent_dim,
                                 out_features=self.par_hidden_neurons[0])
        self.batch_norm1 = nn.BatchNorm1d(num_features=self.par_hidden_neurons[0])
        self.linear2 = nn.Linear(in_features=self.par_hidden_neurons[0],
                                 out_features=self.par_hidden_neurons[1])
        self.batch_norm2 = nn.BatchNorm1d(num_features=self.par_hidden_neurons[1])
        self.linear3 = nn.Linear(in_features=self.par_hidden_neurons[1],
                                 out_features=self.par_hidden_neurons[2])
        self.batch_norm3 = nn.BatchNorm1d(num_features=self.par_hidden_neurons[2])
        self.linear4 = nn.Linear(in_features=self.par_hidden_neurons[2],
                                 out_features=self.par_dim,
                                 bias=False)

        self.apply(init_weights)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.batch_norm1(x)
        x = self.activation(self.linear2(x))
        x = self.batch_norm2(x)
        x = self.activation(self.linear3(x))
        x = self.batch_norm3(x)
        x = self.linear4(x)
        return x#self.sigmoid(x)


class ParCritic(nn.Module):
    def __init__(self, par_dim=2, par_hidden_neurons=[8, 8]):
        super(ParCritic, self).__init__()

        self.par_hidden_neurons = par_hidden_neurons
        self.par_dim = par_dim
        self.activation = nn.Tanh()

        self.linear1 = nn.Linear(in_features=self.par_dim,
                                 out_features=self.par_hidden_neurons[0])
        self.linear2 = nn.Linear(in_features=self.par_hidden_neurons[0],
                                 out_features=self.par_hidden_neurons[1])
        self.linear3 = nn.Linear(in_features=self.par_hidden_neurons[0],
                                 out_features=self.par_hidden_neurons[1])
        self.linear4 = nn.Linear(in_features=self.par_hidden_neurons[1],
                                 out_features=1,
                                 bias=False)

        self.apply(init_weights)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim=8,
                 par_dim=1,
                 hidden_channels=[],
                 par_latent_dim=2,
                 par_hidden_neurons=[8, 8]):

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.par_dim = par_dim
        self.hidden_channels = hidden_channels
        self.activation = nn.Tanh()
        self.kernel_size = 5
        self.stride = 2
        self.sigmoid = nn.Sigmoid()

        self.par_latent_dim = par_latent_dim
        self.par_hidden_neurons = par_hidden_neurons

        self.par_generator = ParGenerator(par_latent_dim=self.par_latent_dim,
                                          par_dim=self.par_dim,
                                          par_hidden_neurons=self.par_hidden_neurons)

        self.par_layer = nn.Linear(in_features=self.par_latent_dim,
                                   out_features=16)
        self.state_layer = nn.Linear(in_features=self.latent_dim-self.par_latent_dim,
                                   out_features=self.hidden_channels[0])

        self.in_layer = nn.Linear(in_features=self.hidden_channels[0]+16,
                                  out_features=self.hidden_channels[0] * 4)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_channels[0])
        self.TransposedConv1 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[0],
                out_channels=self.hidden_channels[1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_channels[1])
        self.TransposedConv2 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[1],
                out_channels=self.hidden_channels[2],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden_channels[2])
        self.TransposedConv3 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[2],
                out_channels=self.hidden_channels[3],
                kernel_size=4,
                stride=2,
                bias=False)

    def forward(self, x, output_pars=False):
        #pars = self.par_generator(x[:, -self.par_latent_dim:])
        pars_in = x[:, -self.par_latent_dim:]
        pars = self.activation(self.par_layer(pars_in))
        x = self.activation(self.state_layer(x[:, :-self.par_latent_dim]))
        x = torch.cat((x, pars), dim=1)
        x = self.activation(self.in_layer(x))
        x = x.view(x.size(0), self.hidden_channels[0], 4)
        x = self.batch_norm1(x)
        x = self.activation(self.TransposedConv1(x))
        x = self.batch_norm2(x)
        x = self.activation(self.TransposedConv2(x))
        x = self.batch_norm3(x)
        x = self.TransposedConv3(x)
        x = self.sigmoid(x)

        if output_pars:
            return x, pars_in
        else:
            return x

class Critic(nn.Module):
    def __init__(self, par_dim, hidden_channels=[]):
        super(Critic, self).__init__()

        self.par_dim = par_dim
        self.hidden_channels = hidden_channels
        self.activation = nn.Tanh()
        self.kernel_size = 5
        self.stride = 2

        self.par_layer = nn.Linear(in_features=self.par_dim,
                                   out_features=16)

        self.Conv1 = nn.Conv1d(
                in_channels=self.hidden_channels[0],
                out_channels=self.hidden_channels[1],
                kernel_size=4,
                stride=2,
                bias=True)
        self.Conv2 = nn.Conv1d(
                in_channels=self.hidden_channels[1],
                out_channels=self.hidden_channels[2],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.Conv3 = nn.Conv1d(
                in_channels=self.hidden_channels[2],
                out_channels=self.hidden_channels[3],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)

        self.dense_layer = nn.Linear(in_features=self.hidden_channels[-1] * 4,
                                     out_features=self.hidden_channels[-1])

        self.combined_layer = nn.Linear(in_features= self.hidden_channels[-1] + 16,
                                        out_features=self.hidden_channels[0])

        self.out_layer = nn.Linear(in_features=self.hidden_channels[0],
                                   out_features=1,
                                   bias=False)

    def forward(self, x, pars):
        x = self.activation(self.Conv1(x))
        x = self.activation(self.Conv2(x))
        x = self.activation(self.Conv3(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.dense_layer(x))
        pars = self.activation(self.par_layer(pars))
        x = torch.cat((x, pars), dim=1)
        x = self.activation(self.combined_layer(x))
        x = self.out_layer(x)
        return x