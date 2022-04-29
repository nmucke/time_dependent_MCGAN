import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt

class AdvDiffDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 num_files=10,
                 num_states_pr_sample=10,
                 sample_size = (128, 512),
                 window_size=32,
                 transformer_state=None,
                 transformer_pars=None,
                 ):

        self.data_path_state = data_path
        self.num_files = num_files
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars
        self.num_states_pr_sample = num_states_pr_sample
        self.num_x = sample_size[0]
        self.num_t = sample_size[1]
        self.window_size = window_size

        self.state_IDs = [i for i in range(self.num_files)]

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
        return self.num_files

    def __getitem__(self, idx):

        #sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
        #                              dtype=int, endpoint=False)
        sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
                                      dtype=int, endpoint=False)

        data = np.load(f"{self.data_path_state}_{idx}.npy", allow_pickle=True)
        data = data.item()
        state = data['sol']
        state = state[:, sample_time_ids]

        state_conditions = np.zeros((self.num_states_pr_sample-2*self.window_size,
                                    self.num_x,
                                    self.window_size))
        state_pred = np.zeros((self.num_states_pr_sample-2*self.window_size,
                                    self.num_x,
                                    self.window_size))

        for i in range(self.window_size, state.shape[-1]-self.window_size):
            state_conditions[i-self.window_size, :, :] = state[:, i-self.window_size:i]
            state_pred[i-self.window_size, :, :] = state[:, i:i+self.window_size]

        return torch.tensor(state_conditions, dtype=torch.get_default_dtype()), \
            torch.tensor(state_pred, dtype=torch.get_default_dtype())
