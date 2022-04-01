import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TransAm(nn.Module):
    def __init__(
            self,
            feature_size=40,
            num_layers=1,
            dropout=0.1,
            num_channels=2,
            dense_neurons=32,
            nhead=2,
            num_pars=None
    ):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.activation = nn.LeakyReLU()
        self.num_channels = num_channels
        self.num_pars = num_pars

        self.src_mask = None
        self.initial_embedding1 = nn.Conv1d(in_channels=num_channels,
                                           out_channels=feature_size,
                                           kernel_size=1)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,
                                                        nhead=nhead,
                                                        dim_feedforward=dense_neurons,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_layers)

        if num_pars is not None:
            out_features = num_channels + num_pars
        else:
            out_features = num_channels

        self.decoder1 = nn.Linear(in_features=feature_size,
                                 out_features=out_features,
                                 bias=False)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder1.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.initial_embedding1(src.transpose(0,1).transpose(1,2))
        src = self.activation(src)
        src.transpose_(1, 2).transpose_(0, 1)
        #src = self.pos_encoder(src)
        output = self.transformer_encoder(src,
                                          mask=self.src_mask)  # , self.src_mask)

        output = self.decoder1(output)
        #state = output[-1, :, 0:self.num_channels]
        #pars = output[-1, :, -self.num_pars:]

        return output#, pars

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask
