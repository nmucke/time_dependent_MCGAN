import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.adv_diff_dataloader import get_dataloader
import models.pipe_flow_models.autoencoder as models
from utils.seed_everything import seed_everything
from training.train_adversarial_AE import TrainAdversarialAE
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from torch.utils.data import DataLoader
from data_handling.pipe_flow_dataloader import PipeFlowDataset, TransformState, TransformPars

torch.set_default_dtype(torch.float32)

if __name__ == '__main__':

    seed_everything()


    with_koopman_training = False
    with_adversarial_training = True

    num_time_steps = 1000
    dataset_params = {
        'num_files': 2000,
        'num_states_pr_sample': 1000,
        'sample_size': (1000, 256),
        'pars': True,
        'with_koopman_training': with_koopman_training,
    }
    batch_size = 4
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 1,
        'drop_last': True,
    }

    data_path = 'pipe_flow/data/pipe_flow'
    dataset = PipeFlowDataset(data_path, **dataset_params)
    dataloader = DataLoader(dataset, **dataloader_params)

    transformer_state = TransformState()
    transformer_pars = TransformPars()

    for i, (state, pars) in enumerate(dataloader):
        transformer_state.partial_fit(state.numpy().reshape(batch_size*num_time_steps, 2, 256))
        transformer_pars.partial_fit(pars.numpy())

    dataset = PipeFlowDataset(
            data_path,
            **dataset_params,
            transformer_state=transformer_state,
            transformer_pars=transformer_pars
    )
    dataloader = DataLoader(dataset, **dataloader_params)

    latent_dim = 16
    input_dim = 128
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_channels': [16, 32, 64, 128, 256],
    }

    encoder = models.Encoder(**encoder_params).to('cuda')

    load_string = 'AE_pipe_flow_large_' + str(latent_dim)
    if with_koopman_training and with_adversarial_training:
        load_string += '_koopman_adversarial'
    elif with_adversarial_training:
        load_string += '_adversarial'
    elif with_koopman_training:
        load_string += '_koopman'

    checkpoint_path = 'model_weights/' + load_string
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])


    reduced_data = torch.zeros(dataset_params['num_files'], num_time_steps, latent_dim)
    reduced_data_pars = torch.zeros(dataset_params['num_files'], num_time_steps, 2)
    for i, (data, pars) in enumerate(dataloader):

        batch_size = data.shape[0]

        data = data.reshape(-1, data.shape[2], data.shape[3])
        data = data.to('cuda')

        pars = pars.reshape(-1, 1, pars.shape[1])
        pars = pars.repeat(1, num_time_steps, 1)
        reduced_data_pars[i * dataloader_params['batch_size']:i * dataloader_params[
            'batch_size'] + batch_size] = pars

        z = encoder(data)

        z = z.view(batch_size, num_time_steps, latent_dim).cpu().detach()
        reduced_data[i*dataloader_params['batch_size']:i*dataloader_params['batch_size']+batch_size] = z

        print(f'{i} of {len(dataloader)}')

    np.save('reduced_data_pipe_flow_' + str(latent_dim), reduced_data.numpy())
    np.save('reduced_data_pipe_flow_pars_' + str(latent_dim), reduced_data_pars.numpy())

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

