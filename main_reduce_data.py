import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.adv_diff_dataloader import get_dataloader
import models.adv_diff_models.adversarial_AE as models
from utils.seed_everything import seed_everything
from training.train_adversarial_AE import TrainAdversarialAE
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint

torch.set_default_dtype(torch.float32)

if __name__ == '__main__':

    seed_everything()

    dataloader_params = {
        'num_files': 100000,
        'transformer_state': None,
        'transformer_pars': None,
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 8,
        'drop_last': True,
        'num_states_pr_sample': 64,
        'sample_size': (128, 512),
        'pars': False
    }
    data_path = 'data/advection_diffusion/train_data/adv_diff'
    dataloader = get_dataloader(data_path, **dataloader_params)

    latent_dim = 8
    input_dim = 128
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [64, 32],
    }

    latent_dim = 8
    encoder = models.Encoder(**encoder_params).to('cuda')
    load_checkpoint(
        checkpoint_path='model_weights/AdvAE',
        encoder=encoder,
        )

    reduced_data = torch.zeros(dataloader_params['num_files'], 512, latent_dim)
    for i, data in enumerate(dataloader):

        batch_size = data.shape[0]

        data = data.reshape(-1, data.shape[2])
        data = data.to('cuda')

        z = encoder(data)

        z = z.view(batch_size, 512, latent_dim).cpu().detach()
        reduced_data[i*dataloader_params['batch_size']:i*dataloader_params['batch_size']+batch_size] = z

        print(f'{i} of {len(dataloader)}')

    np.save('reduced_data', reduced_data.numpy())

    '''
    plt.figure()
    plt.plot(reduced_data[0,:,0])
    plt.plot(reduced_data[0,:,1])
    plt.plot(reduced_data[0,:,2])
    plt.plot(reduced_data[0,:,3])
    plt.plot(reduced_data[0,:,4])
    plt.plot(reduced_data[0,:,5])
    plt.plot(reduced_data[0,:,6])
    plt.plot(reduced_data[0,:,7])
    plt.show()
    '''

