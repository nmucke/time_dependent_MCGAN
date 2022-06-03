import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt

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