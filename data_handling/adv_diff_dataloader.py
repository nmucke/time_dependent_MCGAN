import pdb
import numpy as np
import torch

class AdvDiffDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 num_files=10,
                 num_states_pr_sample=10,
                 sample_size = (128, 512),
                 transformer_state=None,
                 transformer_pars=None,
                 pars=False
                 ):

        self.data_path_state = data_path
        self.num_files = num_files
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars
        self.num_states_pr_sample = num_states_pr_sample
        self.num_x = sample_size[0]
        self.num_t = sample_size[1]
        self.pars = pars

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
        #sample_time_ids = np.random.randint(0,self.num_t, self.num_states_pr_sample)
        sample_time_ids = np.linspace(0, self.num_t, self.num_states_pr_sample,
                                      dtype=int, endpoint=False)

        data = np.load(f"{self.data_path_state}_{idx}.npy", allow_pickle=True)
        data = data.item()
        state = data['sol']
        state = state[:, sample_time_ids].transpose()

        if self.pars:
            pars = data['PDE_params']
            pars = torch.tensor([pars['velocity'], pars['diffusion']],
                                dtype=torch.get_default_dtype())

            return torch.tensor(state, dtype=torch.get_default_dtype()), pars
        else:
            return torch.tensor(state, dtype=torch.get_default_dtype())

def get_dataloader(data_path,
                    num_files=100000,
                    transformer_state=None,
                    transformer_pars=None,
                    batch_size=512,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True,
                    num_states_pr_sample=10,
                    sample_size = (128, 512),
                    pars=False
                    ):

    dataset = AdvDiffDataset(
            data_path=data_path,
            num_files=num_files,
            transformer_state=transformer_state,
            transformer_pars=transformer_pars,
            num_states_pr_sample=num_states_pr_sample,
            sample_size=(128, 512),
            pars=pars
    )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
    )

    return dataloader