import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math
from torch import nn, Tensor

def generate_square_subsequent_mask(dim1, dim2):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, sequence length
    Return:
        A Tensor of shape [dim1, dim2, dim3]
    """
    return torch.triu(torch.ones(dim1,  dim2) * float('-inf'), diagonal=1)


class PositionalEncoder(nn.Module):
    """
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
    """

    def __init__(
            self,
            dropout: float = 0.1,
            max_seq_len: int = 5000,
            d_model: int = 512
    ):
        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values
        # dependent on position and i
        position = torch.arange(max_seq_len).unsqueeze(1)

        exp_input = torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)

        div_term = torch.exp(exp_input)
        # Returns a new tensor with the exponential of the
        # elements of exp_input

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # torch.Size([target_seq_len, dim_val])

        pe = pe.unsqueeze(0).transpose(0,1)  # torch.Size([target_seq_len, input_size, dim_val])

        # register that pe is not a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        add = self.pe[:x.size(1), :].squeeze(1)
        x = x + add
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 16,
            embed_dim: int = 512,
            dropout_pos_enc: float = 0.1,
            max_seq_len: int = 5000,
            n_heads: int = 8,
            num_encoder_layers: int = 6,
    ):
        super().__init__()

        #self.encoder_input_layer = nn.Linear(
        #        in_features=latent_dim,
        #        out_features=embed_dim
        #)
        self.encoder_input_layer = nn.Conv1d(
                in_channels=latent_dim,
                out_channels=embed_dim,
                kernel_size=1)

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
                d_model=embed_dim,
                dropout=dropout_pos_enc,
                max_seq_len=max_seq_len
        )

        # Create an encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
        )

        # Stack the encoder layer n times in nn.TransformerDecoder
        self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers,
                norm=None
        )

    def forward(self, src):
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        x = src.transpose(0, 1)
        x = x.transpose(1, 2)
        # Apply the encoder input layer
        x = self.encoder_input_layer(x)

        x = x.transpose(1, 2)
        # Apply the positional encoder
        x = self.positional_encoding_layer(x)
        x = x.transpose(0, 1)

        # Apply the encoder
        x = self.encoder(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 16,
            pars_dim: int = 2,
            embed_dim: int = 512,
            n_heads: int = 8,
            num_decoder_layers: int = 6,
            out_seq_len: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim


        #self.decoder_input_layer = nn.Linear(
        #        in_features=latent_dim + pars_dim,
        #        out_features=embed_dim
        #)
        self.decoder_input_layer = nn.Conv1d(
                in_channels=latent_dim+pars_dim,
                out_channels=embed_dim,
                kernel_size=1)

        # Create the decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
        )

        # Stack the decoder layer n times
        self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
                norm=None
        )

        self.decoder_output_layer = nn.Conv1d(
                in_channels=embed_dim,
                out_channels=latent_dim,
                kernel_size=1,
                bias=False
        )

    def forward(self, src, tgt, pars, inference=False):
        """
        Args:
            x: Tensor, shape [batch_size, dec_seq_len, dim_val]
        """
        pars = pars.repeat(tgt.size(0), 1, 1)
        tgt = torch.cat((tgt, pars), dim=2)
        tgt = tgt.transpose(0, 1)
        tgt = tgt.transpose(1, 2)

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt)

        decoder_output = decoder_output.transpose(1, 2)
        decoder_output = decoder_output.transpose(0, 1)

        if inference:
            tgt_mask = None
        else:
            tgt_mask = generate_square_subsequent_mask(
                    decoder_output.size(0),
                    decoder_output.size(0)
            ).to(tgt.device)

        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
        )

        decoder_output = decoder_output.transpose(0, 1)
        decoder_output = decoder_output.transpose(1, 2)
        # Pass through the linear mapping layer

        decoder_output = self.decoder_output_layer(decoder_output)

        decoder_output = decoder_output.transpose(1, 2)
        decoder_output = decoder_output.transpose(0, 1)

        tgt = tgt.transpose(1, 2)
        tgt = tgt.transpose(0, 1)
        decoder_output = decoder_output + tgt[:, :, 0:self.latent_dim]

        return decoder_output




class Transformer(nn.Module):
    def __init__(
            self,
            latent_dim: int = 16,
            pars_dim: int = 2,
            embed_dim: int = 512,
            dropout_pos_enc: float = 0.1,
            max_seq_len: int = 5000,
            out_seq_len: int = 32,
            n_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
    ):
        super(Transformer, self).__init__()
        self.latent_dim = latent_dim
        self.out_seq_len = out_seq_len

        self.encoder = TransformerEncoder(
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                dropout_pos_enc=dropout_pos_enc,
                max_seq_len=max_seq_len,
                n_heads=n_heads,
                num_encoder_layers=num_encoder_layers,
        )

        self.decoder = TransformerDecoder(
                latent_dim=latent_dim,
                pars_dim=pars_dim,
                embed_dim=embed_dim,
                n_heads=n_heads,
                num_decoder_layers=num_decoder_layers,
                out_seq_len=out_seq_len,
        )

    def forward(self, src, tgt, pars, num_steps=None):
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        # Apply the encoder
        src = self.encoder(src)

        if num_steps is not None:
            output = tgt
            for i in range(1, num_steps+1):
                out = self.decoder(src=src, tgt=output[-self.out_seq_len:],
                                   pars=pars, inference=True)[-1:]
                output = torch.cat((output, out), dim=0)
            return output[-num_steps:]
        else:
            # Apply the decoder
            output = self.decoder(src=src, tgt=tgt, pars=pars)
            return output
















































'''

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
'''