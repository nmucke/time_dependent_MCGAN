import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means
        there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch,
        input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states
        in the sequence;
        :                              hidden gives the hidden state and cell
        state for the last
        :                              element in the sequence
        '''

        x_input = x_input.view(x_input.shape[0], x_input.shape[1], self.input_size)
        lstm_out, self.hidden = self.lstm(x_input)

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, output_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means
        there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size,
        input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden
        states in the sequence;
        :                                   hidden gives the hidden state and
        cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0),
                                          encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            par_size,
            temporal_latent_dim=8,
    ):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.par_size = par_size
        self.num_layers = num_layers
        self.temporal_latent_dim = temporal_latent_dim

        self.encoder = lstm_encoder(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers)
        self.decoder = lstm_decoder(input_size=input_size + hidden_size//2,
                                    output_size=output_size,
                                    hidden_size=hidden_size,# + temporal_latent_dim,
                                    num_layers=num_layers)

        self.par_dense = nn.Sequential(
            nn.Linear(par_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size//2, hidden_size//2),
        )
        '''
        self.temporal_latent_dense = nn.Sequential(
            nn.Linear(temporal_latent_dim, temporal_latent_dim),
            nn.LeakyReLU(),
            nn.Linear(temporal_latent_dim, temporal_latent_dim),
        )
        '''

    def forward(self, input_tensor, target_len, pars, teacher_forcing=None):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch
        tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values;
        prediction done recursively
        '''
        batch_size = input_tensor.shape[1]

        # encode input_tensor
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(
                target_len,
                batch_size,
                input_tensor.shape[2],
                device=input_tensor.device
        )

        # decode input_tensor
        decoder_input = input_tensor[-1]
        decoder_input = torch.cat((decoder_input, self.par_dense(pars[-1])), dim=-1)
        #decoder_input = self.par_dense(pars[-1])

        #z = torch.randn(self.num_layers, batch_size, self.temporal_latent_dim, device=input_tensor.device)
        decoder_hidden = encoder_hidden
        #decoder_hidden = (torch.cat((decoder_hidden[0], z), dim=-1),
        #                  torch.cat((decoder_hidden[1], z), dim=-1))

        last_state = input_tensor[-1]

        if teacher_forcing is not None:
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                outputs[t] = decoder_output.squeeze(0)
                decoder_input = teacher_forcing[t]
                decoder_input = torch.cat((decoder_input, self.par_dense(pars[-1])), dim=-1)

        else:
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                outputs[t] = decoder_output.squeeze(0)
                decoder_input = decoder_output
                decoder_input = torch.cat((decoder_input, self.par_dense(pars[-1])), dim=-1)


        return outputs

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()

        self.encoder = lstm_encoder(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers)

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, input_tensor):
        score = self.encoder(input_tensor)
        score = self.dense(score[0][-1])
        return self.sigmoid(score)


'''
class Generator(nn.Module):
    def __init__(self, n_input, n_latent, n_hidden, n_output):
        super(Generator, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_latent = n_latent

        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(self.n_input + self.n_latent, self.n_hidden)
        self.fc2 = nn.Linear(self.n_features, self.n_hidden)
        self.fc3 = nn.Linear(self.n_features, self.n_hidden)
        self.fc4 = nn.Linear(self.n_hidden, self.n_output, bias=False)

    def forward(self, x, z):
        x = torch.cat((x, z), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Discriminator, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(self.n_input, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, self.n_output)
        self.fc4 = nn.Linear(self.n_hidden, 1, bias=False)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x
'''