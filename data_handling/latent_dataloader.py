import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt

def create_multistep_data(state, in_seq_len, out_seq_len, num_steps, latent_dim):

    in_state = torch.zeros(
            (num_steps - (in_seq_len + out_seq_len),
             in_seq_len,
             latent_dim),
    )
    state_out = torch.zeros(
            (num_steps - (in_seq_len + out_seq_len),
             out_seq_len,
             latent_dim),
    )
    for i in range(num_steps - in_seq_len - out_seq_len):
        in_state[i] = state[i:i + in_seq_len, :]
        state_out[i] = state[i + in_seq_len:i + in_seq_len + out_seq_len, :]

    return in_state, state_out

def create_onestep_data(state, in_seq_len, num_steps, latent_dim):

    in_state = torch.zeros(
            (num_steps - in_seq_len - 1,
            in_seq_len,
            latent_dim),
    )
    out_state = torch.zeros(
            (num_steps - in_seq_len - 1,
            in_seq_len,
            latent_dim),
    )

    for i in range(0, num_steps - in_seq_len - 1):
        in_ = state[i:i + in_seq_len]
        out_ = state[i + 1:i + in_seq_len + 1]

        in_state[i] = in_
        out_state[i] = out_

    return in_state, out_state




class LatentDatasetTransformersCustom(torch.utils.data.Dataset):
    def __init__(self,
                 data_state,
                 data_pars,
                 num_states_pr_sample=128,
                 sample_size = (8, 512),
                 window_size=(32, 1),
                 num_samples=250,
                 transformer_state=None,
                 transformer_pars=None,
                 multi_step=False,
                 ):

        self.data_state = data_state[:num_samples]
        self.data_pars = data_pars[0:num_samples, 0:1]
        self.num_samples = self.data_pars.shape[0]
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars
        self.num_states_pr_sample = num_states_pr_sample
        self.num_latent = sample_size[0]
        self.num_t = sample_size[1]
        self.num_pars = self.data_pars.shape[-1]
        self.multi_step = multi_step

        self.input_window_size = window_size[0]
        self.output_window_size = window_size[1]

        self.state_IDs = [i for i in range(len(self.data_pars))]

        if self.transformer_state is not None:
            self.transformer_state = transformer_state

        if self.transformer_pars is not None:
            self.transformer_pars = transformer_pars

    def transform_state(self, data):
        return self.transformer_state.min_max_transform(data)

    def inverse_transform_state(self, data):
        return self.transformer_state.min_max_inverse_transform(data)

    def transform_pars(self, data):
        return self.transformer_pars.min_max_transform(data)

    def inverse_transform_pars(self, data):
        return self.transformer_pars.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
                                      dtype=int, endpoint=False)

        state = self.data_state[idx]
        state = state[sample_time_ids]

        pars = self.data_pars[idx]

        if self.multi_step:
            in_state, out_state = create_multistep_data(
                    state=state,
                    in_seq_len=self.input_window_size,
                    out_seq_len=self.output_window_size,
                    num_steps=self.num_states_pr_sample,
                    latent_dim=self.num_latent
            )
        else:
            in_state, out_state = create_onestep_data(
                    state=state,
                    in_seq_len=self.input_window_size,
                    num_steps=self.num_states_pr_sample,
                    latent_dim=self.num_latent
            )

        return in_state, out_state, pars




def get_src_trg(
    sequence: torch.Tensor,
    enc_seq_len: int,
    target_seq_len: int,
    dec_seq_len: int = None,
    ):

    """
    Generate the src (encoder input), trg (decoder input) and trg_y (the target)
    sequences from a sequence.
    Args:
        sequence: tensor, a 1D tensor of length n where
                n = encoder input length + target sequence length
        enc_seq_len: int, the desired length of the input to the transformer encoder
        target_seq_len: int, the desired length of the target sequence (the
                        one against which the model output is compared)
    Return:
        src: tensor, 1D, used as input to the transformer model
        trg: tensor, 1D, used as input to the transformer model
        trg_y: tensor, 1D, the target sequence against which the model output
            is compared when computing loss.
    """
    assert len(sequence) == enc_seq_len + target_seq_len, \
        "Sequence length does not equal (input length + target length)"

    # encoder input
    src = sequence[:enc_seq_len]

    # decoder input. As per the paper, it must have the same dimension as the
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[enc_seq_len-1:len(sequence)-1]

    assert len(trg) == target_seq_len, \
        "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:]

    assert len(trg_y) == target_seq_len, \
        "Length of trg_y does not match target sequence length"

    return src, trg, trg_y # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]


class LatentDatasetTransformers(torch.utils.data.Dataset):
    def __init__(self,
                 data_state,
                 data_pars,
                 num_states_pr_sample=128,
                 sample_size = (8, 512),
                 window_size=(32, 1),
                 transformer_state=None,
                 transformer_pars=None,
                 ):

        self.data_state = data_state
        self.data_pars = data_pars[:,0:1]
        self.num_samples = self.data_pars.shape[0]
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars
        self.num_states_pr_sample = num_states_pr_sample
        self.num_latent = sample_size[0]
        self.num_t = sample_size[1]
        self.num_pars = self.data_pars.shape[-1]

        self.input_window_size = window_size[0]
        self.output_window_size = window_size[1]

        self.state_IDs = [i for i in range(len(self.data_pars))]

        if self.transformer_state is not None:
            self.transformer_state = transformer_state

        if self.transformer_pars is not None:
            self.transformer_pars = transformer_pars

    def transform_state(self, data):
        return self.transformer_state.min_max_transform(data)

    def inverse_transform_state(self, data):
        return self.transformer_state.min_max_inverse_transform(data)

    def transform_pars(self, data):
        return self.transformer_pars.min_max_transform(data)

    def inverse_transform_pars(self, data):
        return self.transformer_pars.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
                                      dtype=int, endpoint=False)

        state = self.data_state[idx]
        state = state[sample_time_ids]

        pars = self.data_pars[idx]

        src_data = torch.zeros(
                self.num_states_pr_sample - self.input_window_size - self.output_window_size,
                self.input_window_size,
                self.num_latent
        )
        tgt_data = torch.zeros(
                self.num_states_pr_sample - self.input_window_size - self.output_window_size,
                self.output_window_size,
                self.num_latent
        )
        tgt_y_data = torch.zeros(
                self.num_states_pr_sample - self.input_window_size - self.output_window_size,
                self.output_window_size,
                self.num_latent
        )

        for i in range(self.input_window_size, self.num_states_pr_sample-self.output_window_size):
            src, tgt, tgt_y = get_src_trg(
                sequence=state[i-self.input_window_size:i+self.output_window_size],
                enc_seq_len=self.input_window_size,
                target_seq_len=self.output_window_size
            )
            src_data[i-self.input_window_size] = src
            tgt_data[i-self.input_window_size] = tgt
            tgt_y_data[i-self.input_window_size] = tgt_y

        return src_data, tgt_data, tgt_y_data, pars





class LatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_state,
                 data_pars,
                 num_states_pr_sample=128,
                 sample_size = (8, 512),
                 window_size=(32, 1),
                 transformer_state=None,
                 transformer_pars=None,
                 ):

        self.data_state = data_state
        self.data_pars = data_pars
        self.num_samples = self.data_pars.shape[0]
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars
        self.num_states_pr_sample = num_states_pr_sample
        self.num_latent = sample_size[0]
        self.num_t = sample_size[1]
        self.num_pars = self.data_pars.shape[-1]

        self.window_size = window_size[0]
        self.window_size_pred = window_size[1]

        self.state_IDs = [i for i in range(len(self.data_pars))]

        if self.transformer_state is not None:
            self.transformer_state = transformer_state

        if self.transformer_pars is not None:
            self.transformer_pars = transformer_pars

    def transform_state(self, data):
        return self.transformer_state.min_max_transform(data)

    def inverse_transform_state(self, data):
        return self.transformer_state.min_max_inverse_transform(data)

    def transform_pars(self, data):
        return self.transformer_pars.min_max_transform(data)

    def inverse_transform_pars(self, data):
        return self.transformer_pars.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
                                      dtype=int, endpoint=False)

        state = self.data_state[idx]
        state = state[sample_time_ids]

        pars = self.data_pars[idx]
        pars = pars[sample_time_ids]

        state_conditions = np.zeros(
                (self.num_states_pr_sample-(self.window_size+self.window_size_pred),
                 self.window_size,
                 self.num_latent)
        )
        state_pred = np.zeros(
                (self.num_states_pr_sample-(self.window_size+self.window_size_pred),
                 self.window_size_pred,
                 self.num_latent)
        )

        pars_conditions = np.zeros(
                (self.num_states_pr_sample-(self.window_size+self.window_size_pred),
                 self.window_size,
                 self.num_pars)
        )
        pars_pred = np.zeros(
                (self.num_states_pr_sample-(self.window_size+self.window_size_pred),
                 self.window_size_pred,
                 self.num_pars)
        )
        for i in range(self.num_states_pr_sample-self.window_size-self.window_size_pred):
            state_conditions[i] = state[i:i+self.window_size, :]
            state_pred[i] = state[i+self.window_size:i+self.window_size+self.window_size_pred, :]

            pars_conditions[i] = pars[i:i+self.window_size, :]
            pars_pred[i] = pars[i+self.window_size:i+self.window_size+self.window_size_pred, :]


        state_out = {
            'state_conditions': torch.tensor(state_conditions, dtype=torch.get_default_dtype()),
            'state_pred': torch.tensor(state_pred, dtype=torch.get_default_dtype())
        }
        pars_out = {
            'pars_conditions': torch.tensor(pars_conditions, dtype=torch.get_default_dtype()),
            'pars_pred': torch.tensor(pars_pred, dtype=torch.get_default_dtype())
        }
        return state_out, pars_out