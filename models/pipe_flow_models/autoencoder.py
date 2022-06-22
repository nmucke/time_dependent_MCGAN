import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time

class Encoder1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 8)
        self.lin5 = nn.Linear(8, 4)

    def forward(self, x):


        x = self.lin1(x)
        x = self.activation(x)

        x = self.lin2(x)
        x = self.activation(x)

        x = self.lin3(x)
        x = self.activation(x)

        x = self.lin4(x)
        x = self.activation(x)

        return self.lin5(x)

class Decoder1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.lin1 = nn.Linear(4, 8)
        self.lin2 = nn.Linear(8, 16)
        self.lin3 = nn.Linear(16, 32)
        self.lin4 = nn.Linear(32, 64)
        self.lin5 = nn.Linear(64, 128)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)

        x = self.lin2(x)
        x = self.activation(x)

        x = self.lin3(x)
        x = self.activation(x)

        x = self.lin4(x)
        x = self.activation(x)

        return self.lin5(x)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class Koopman(nn.Module):
    def __init__(self, latent_dim, par_dim, hidden_neurons, num_diags=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.par_dim = par_dim
        self.hidden_neurons = hidden_neurons

        self.diag_ids = torch.arange(latent_dim)

        self.pars_nn_diagonal = nn.Sequential(
            nn.Linear(par_dim, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, latent_dim),
        )

        off_diag_ids_x = []
        off_diag_ids_y = []
        for i in range(1, num_diags):
            off_diag_ids_y.append(np.arange(i, self.latent_dim))
            off_diag_ids_x.append(np.arange(0, self.latent_dim-i))
        self.off_diag_ids_x = torch.LongTensor(np.concatenate(off_diag_ids_x))
        self.off_diag_ids_y = torch.LongTensor(np.concatenate(off_diag_ids_y))

        self.pars_nn_upper_diagonal = nn.Sequential(
            nn.Linear(par_dim, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, self.off_diag_ids_x.size(0)),
        )
        self.pars_nn_lower_diagonal = nn.Sequential(
            nn.Linear(par_dim, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, self.off_diag_ids_x.size(0)),
        )


    def forward(self, x, pars):

        pars_diag = self.pars_nn_diagonal(pars)
        pars_upper_diag = self.pars_nn_upper_diagonal(pars)
        pars_lower_diag = self.pars_nn_lower_diagonal(pars)

        koopman_matrix = torch.autograd.Variable(torch.zeros(
            x.shape[0], self.latent_dim, self.latent_dim
        )).to(x.device)
        koopman_matrix[:, self.diag_ids, self.diag_ids] = pars_diag
        koopman_matrix[:, self.off_diag_ids_x, self.off_diag_ids_y] = \
            pars_upper_diag
        koopman_matrix[:, self.off_diag_ids_y, self.off_diag_ids_x] = \
            pars_lower_diag

        out = torch.bmm(koopman_matrix, x.unsqueeze(-1))

        return out.squeeze(-1), koopman_matrix


class Encoder(nn.Module):
    def __init__(self, latent_dim=32, input_dim=128, hidden_channels=[64, 32, 16]):
        super().__init__()

        self.activation = nn.LeakyReLU()
        #dense_neurons = [input_dim] + hidden_neurons
        conv_channels = [2] + hidden_channels

        '''
        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=dense_neurons[i],
                        out_features=dense_neurons[i+1]
                ) for i in range(len(dense_neurons)-1)]
        )
        '''

        self.conv_layers = nn.ModuleList(
                [nn.Conv1d(
                        in_channels=conv_channels[i],
                        out_channels=conv_channels[i+1],
                        kernel_size=7,
                        stride=2
                ) for i in range(len(conv_channels)-1)]
        )

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(conv_channels[i+1])
                 for i in range(len(conv_channels) - 1)]
        )

        #self.dense_out = nn.Linear(in_features=dense_neurons[-1],
        #                           out_features=latent_dim,
        #                           bias=False
        #                           )
        self.dense_out1 = nn.Linear(in_features=3*conv_channels[-1],
                                   out_features=conv_channels[-1],
                                   bias=True
                                   )
        self.dense_out2 = nn.Linear(in_features=conv_channels[-1],
                                   out_features=latent_dim,
                                   bias=False
                                   )
        self.normalize = nn.LayerNorm(latent_dim)

    def forward(self, x):
        for conv_layer, batch_norm in zip(self.conv_layers,
                                           self.batch_norm_layers):
            x = conv_layer(x)
            x = self.activation(x)
            x = batch_norm(x)

        x = x.view(x.size(0), -1)
        x = self.dense_out1(x)
        x = self.activation(x)
        x = self.dense_out2(x)
        x = self.normalize(x)

        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, input_dim=128, hidden_channels=[]):
        super().__init__()

        self.activation = nn.LeakyReLU()
        self.hidden_channels = hidden_channels

        self.conv_channels = hidden_channels + [2]

        out_pad = [0, 1, 1, 0, 0]
        bias = [True, True, True, True, False]

        self.dense_in1 = nn.Linear(in_features=latent_dim,
                              out_features=self.conv_channels[0])
        self.dense_in2 = nn.Linear(in_features=self.conv_channels[0],
                              out_features=3*self.conv_channels[0])

        self.conv_layers = nn.ModuleList(
                [nn.ConvTranspose1d(
                        in_channels=self.conv_channels[i],
                        out_channels=self.conv_channels[i + 1],
                        kernel_size=7,
                        stride=2,
                        output_padding=out_pad[i],
                        bias=bias[i]
                ) for i in range(len(self.conv_channels) - 1)]
        )
        '''
        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=dense_neurons[i],
                        out_features=dense_neurons[i + 1]
                ) for i in range(len(dense_neurons) - 1)]
        )
        self.dense_out = nn.Linear(in_features=dense_neurons[-1],
                                   out_features=input_dim,
                                   bias=False
                                   )
        '''

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.conv_channels[i])
                 for i in range(len(self.conv_channels) - 1)]
        )

    def forward(self, x):

        x = self.dense_in1(x)
        x = self.activation(x)
        x = self.dense_in2(x)
        x = x.view(x.size(0), self.hidden_channels[0], 3)

        for conv_layer, batch_norm in zip(self.conv_layers,
                                           self.batch_norm_layers):
            x = self.activation(x)
            x = batch_norm(x)
            x = conv_layer(x)
        #x = self.dense_out(x)
        return x[:, :, 0:256]

class Critic(nn.Module):
    def __init__(self, latent_dim=32, hidden_neurons=[]):
        super().__init__()

        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        dense_neurons = [latent_dim] + hidden_neurons

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=dense_neurons[i],
                        out_features=dense_neurons[i + 1]
                ) for i in range(len(dense_neurons) - 1)]
        )
        self.dense_out = nn.Linear(in_features=dense_neurons[-1],
                                   out_features=1,
                                   bias=False
                                   )

        #self.batch_norm_layers = nn.ModuleList(
        #        [nn.BatchNorm1d(dense_neurons[i + 1])
        #         for i in range(len(dense_neurons) - 1)]
        #)

    def forward(self, x):

        #for dense_layer, batch_norm in zip(self.dense_layers,
        #                                   self.batch_norm_layers):
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.activation(x)
            #x = batch_norm(x)

        x = self.dense_out(x)
        return self.sigmoid(x)

class AutoEncoder():
    def __init__(self, latent_dim=32, input_dim=128,
                 encoder_params={}, decoder_params={}):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim,
                               input_dim=input_dim,
                               encoder_params=encoder_params)
        self.decoder = Decoder(latent_dim=latent_dim,
                               input_dim=input_dim,
                               decoder_params=decoder_params)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

if __name__ == '__main__':


    def out_size(in_size, stride, padding, kernel_size, out_pad):
        return (in_size-1)*stride-2*padding+1*(kernel_size-1)+out_pad+1

    stride = [2, 2, 2, 2, 2]
    padding = [0, 0, 0, 0, 0, 0]
    out_pad = [0, 1, 1, 0, 0]
    kernel_size = [7,7,7,7, 7]

    in_size = 3
    for i in range(len(stride)):
        in_size = out_size(in_size, stride[i], padding[i], kernel_size[i],
                           out_pad[i])
        print(in_size)

