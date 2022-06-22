import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math
from torch import nn, Tensor


# Positional encodings
def get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates


def positional_encoding(D, position=20, dim=3, device='cpu'):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(D)[np.newaxis, :],
                            D)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
    return torch.tensor(pos_encoding, device=device)

# function that implement the look_ahead mask for masking future time steps.
def create_look_ahead_mask(size, device='cpu'):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=2)
    return mask  # (size, size)

def gelu_fast(x):
    """ Faster approximate form of GELU activation function
    """
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.H = num_heads  # number of heads
        self.D = embed_dim  # dimension

        self.wq = nn.Linear(self.D, self.D * self.H)
        self.wk = nn.Linear(self.D, self.D * self.H)
        self.wv = nn.Linear(self.D, self.D * self.H)

        self.dense = nn.Linear(self.D * self.H, self.D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += mask

        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention)  # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.activation = gelu_fast
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_mlp_dim, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.D = embed_dim
        self.H = num_heads
        self.dropout_rate = dropout_rate
        self.mlp_hidden = MLP(input_dim=self.D, hidden_dim=hidden_mlp_dim, output_dim=self.D)
        self.mlp_out = MLP(input_dim=self.D, hidden_dim=hidden_mlp_dim, output_dim=self.D)
        self.layernorm1 = nn.LayerNorm(self.D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(self.D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.mha = MultiHeadAttention(self.D, self.H)

    def forward(self, x, look_ahead_mask):
        x = self.layernorm1(x)
        attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
        attn = self.dropout1(attn)  # (B,S,D)
        x = x + attn  # (B, S, D)
        m = self.layernorm2(x)  # (B, S, D)
        m = self.mlp_hidden(m)  # (B, S, D)

        output = x + m

        #mlp_act = torch.relu(self.mlp_hidden(attn))
        #mlp_act = self.mlp_out(mlp_act)
        #mlp_act = self.dropout2(mlp_act)

        #output = self.layernorm2(mlp_act + attn)  # (B, S, D)

        return output, attn_weights


class Transformer(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''

    def __init__(
            self,
            latent_dim,
            pars_dim,
            num_layers,
            embed_dim,
            num_heads,
            hidden_mlp_dim,
            out_features,
            dropout_rate,
            max_seq_len=100,
            device='cpu'
    ):
        super(Transformer, self).__init__()
        self.D = embed_dim
        self.H = num_heads
        self.latent_dim = latent_dim
        self.pars_dim = pars_dim

        self.sqrt_D = torch.tensor(math.sqrt(self.D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(latent_dim, self.D)  # multivariate input
        self.output_projection = nn.Linear(self.D,
                                          out_features)  # multivariate output
        self.pos_encoding = positional_encoding(self.D, position=max_seq_len+1, device=device)
        self.dec_layers = nn.ModuleList([TransformerLayer(self.D, self.H, hidden_mlp_dim,
                                                          dropout_rate=dropout_rate
                                                          ) for _ in
                                         range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        self.pars_net = nn.Sequential(
            nn.Linear(self.pars_dim, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D),
        )

    def forward(self, x, pars, num_steps=None):
        B, S, D = x.shape

        pars = self.pars_net(pars)

        if num_steps is None:
            attention_weights = {}
            x = self.input_projection(x)
            x *= self.sqrt_D

            S = S + 1
            x = torch.cat([pars.unsqueeze(1), x], dim=1)

            pos_encoding = self.pos_encoding[:, :S, :].repeat(B, 1, 1)
            x += pos_encoding

            x = self.dropout(x)

            mask = create_look_ahead_mask(S, device=x.device)
            mask *= -1e9
            for i in range(self.num_layers):
                x, block = self.dec_layers[i](x=x,
                                              look_ahead_mask=mask)
                attention_weights['decoder_layer{}'.format(i + 1)] = block

            x = self.output_projection(x)
            out = x[:, 1:, :]

        else:
            input_seq_len = x.size(1)
            out = x
            S = S + 1
            for i in range(num_steps):
                attention_weights = {}
                x = self.input_projection(out[:, -input_seq_len:])
                x *= self.sqrt_D

                x = torch.cat([pars.unsqueeze(1), x], dim=1)

                pos_encoding = self.pos_encoding[:, :S, :].repeat(B, 1, 1)
                x += pos_encoding

                x = self.dropout(x)

                for i in range(self.num_layers):
                    x, block = self.dec_layers[i](x=x,
                                                  look_ahead_mask=None)
                    attention_weights['decoder_layer{}'.format(i + 1)] = block

                x = self.output_projection(x)
                x = x[:, -1:, :]

                out = torch.cat([out, x], dim=1)
            out = out[:, input_seq_len:, :]


        return out, attention_weights  # (B,S,S)