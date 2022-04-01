import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset
import data_handling.data_handling_utils as utils

def batchify(data, batch_size):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        batch_size: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """

    num_features = data.size(-1)
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    return data.view(batch_size, seq_len, num_features).transpose(0,1)

def get_batch(features, targets, time_step, batch_size):

    input = features[:, time_step:time_step+batch_size]
    input = input.reshape(-1, features.shape[-2], features.shape[-1])
    input = torch.swapaxes(input, 0, 1)

    target = targets[:, time_step:time_step+batch_size]
    target = target.reshape(-1, features.shape[-2], features.shape[-1])
    target = torch.swapaxes(target, 0, 1)
    return input, target


def create_inout_sequences(input_data, memory, output_window=1):
    inout_features = []
    inout_targets = []
    L = input_data.shape[0]
    for i in range(L - memory):
        train_seq = input_data[i:i + memory]
        train_label = input_data[i + output_window:i + memory + output_window]
        inout_features.append(train_seq)
        inout_targets.append(train_label)
    return torch.stack(inout_features), torch.stack(inout_targets)



def prepare_batch(data, time_step, memory):
    out_features = torch.zeros([memory, data.shape[0], data.shape[2]])
    out_targets = torch.zeros([memory, data.shape[0], data.shape[2]])
    for i in range(data.shape[0]):
        #batchified_data = utils.batchify(data[i], batch_size=1)
        batchified_data = data[i].unsqueeze(1)#utils.batchify(data[i], batch_size=1)
        batch_features, batch_targets = utils.get_batch(
                batchified_data,
                i=time_step,
                batch_size=memory)
        out_features[:, i:i+1, :] = batch_features
        out_targets[:, i:i+1, :] = batch_targets

    return out_features, out_targets

class TimeSeriesDataset(Dataset):
    def __init__(self, data=None, memory=50):
        if type(data) == str:
            self.data = np.load(data)
        else:
            self.data = data

        self.data = torch.tensor(self.data, dtype=torch.get_default_dtype())

        self.memory = memory
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features, targets = utils.create_inout_sequences(
                self.data[idx],
                memory=self.memory
                )
        return features, targets