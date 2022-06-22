import pdb
import numpy as np
import torch

class TransformState():
    def __init__(self,
                 mode='minmax',
                 min=0,
                 max=1,
                 num_channels=2
                 ):

        self.min = min
        self.max = max
        self.num_channels = num_channels
        self.channel_min = [1e12, 1e12]
        self.channel_max = [-1e12, -1e12]

    def partial_fit(self, data):
        """
        data: (batch_size, num_channels, num_x)
        """
        min = []
        max = []
        for i in range(self.num_channels):
            min.append(np.min(data[:, i, :]))
            max.append(np.max(data[:, i, :]))
        for i in range(self.num_channels):
            if self.channel_min[i] > min[i]:
                self.channel_min[i] = min[i]
            if self.channel_max[i] < max[i]:
                self.channel_max[i] = max[i]

    def transform(self, data):
        for i in range(self.num_channels):
            data[:, i, :] = (data[:, i, :] - self.channel_min[i]) / \
                            (self.channel_max[i]- self.channel_min[i])
            data[:, i, :] = data[:, i, :] * (self.max - self.min) + self.min

        return data

class TransformPars():
    def __init__(self,
                 mode='minmax',
                 min=0,
                 max=1,
                 num_pars=2
                 ):

        self.min = min
        self.max = max
        self.num_pars = num_pars
        self.par_min = [1e12, 1e12]
        self.par_max = [-1e12, -1e12]

    def partial_fit(self, data):
        """
        data: (batch_size, num_pars)
        """
        min = []
        max = []
        for i in range(self.num_pars):
            min.append(np.min(data[:, i]))
            max.append(np.max(data[:, i]))

        for i in range(self.num_pars):
            if self.par_min[i] > min[i]:
                self.par_min[i] = min[i]
            if self.par_max[i] < max[i]:
                self.par_max[i] = max[i]

    def transform(self, data):
        for i in range(self.num_pars):
            data[:, i] = (data[:, i] - self.par_min[i] ) / \
                            (self.par_max[i] - self.par_min[i])
            data[:, i] = data[:, i] * (self.max - self.min) + self.min

        return data


class PipeFlowDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 num_files=10,
                 num_states_pr_sample=10,
                 sample_size=(2000, 256),
                 transformer_state=None,
                 transformer_pars=None,
                 pars=False,
                 with_koopman_training=False
                 ):

        self.data_path_state = data_path
        self.num_files = num_files
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars
        self.num_states_pr_sample = num_states_pr_sample
        self.num_t = sample_size[0]
        self.num_x = sample_size[1]
        self.pars = pars
        self.with_koopman_training = with_koopman_training

        self.state_IDs = [i for i in range(self.num_files)]

        if self.transformer_state is not None:
            self.transformer_state = transformer_state.transform

        if self.transformer_pars is not None:
            self.transformer_pars = transformer_pars.transform

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # sample_time_ids = np.random.randint(0,self.num_t,
        # self.num_states_pr_sample)
        sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
                                      dtype=int, endpoint=False)

        data = np.load(f"{self.data_path_state}_{idx}.npy", allow_pickle=True)
        data = data.item()
        u = data['u']
        pressure = data['pressure']

        state = np.stack((u, pressure), axis=1)
        state = state[sample_time_ids, :]
        if self.transformer_state is not None:
            state = self.transformer_state(state)
        if self.pars:
            pars = data['params']
            pars = np.array([[pars['friction'], pars['inflow_freq']]])

            if self.transformer_pars is not None:
                pars = self.transformer_pars(pars)

            pars = torch.tensor(pars[0], dtype=torch.get_default_dtype())

            return torch.tensor(state, dtype=torch.get_default_dtype()), pars
        else:
            return torch.tensor(state, dtype=torch.get_default_dtype())




