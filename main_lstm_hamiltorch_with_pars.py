import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.latent_dataloader import LatentDataset
import models.adv_diff_models.adversarial_AE as models
from utils.seed_everything import seed_everything
from training.train_adversarial_AE import TrainAdversarialAE
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from torch.utils.data import DataLoader
import models.adv_diff_models.latent_time_gan as time_models
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import hamiltorch
import sys
import os
from inference.data_assimilation import DataAssimilation, PrepareData, PrepareModels, space_obs_operator


torch.set_default_dtype(torch.float32)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':

    seed_everything()
    cuda = True
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    ##### Load encoder/decoder model#####
    input_dim = 128
    latent_dim = 3
    input_window_size = 16
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [4, 8, 16, 32],
    }

    decoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [32, 16, 8, 4],
    }

    forecast_model_params = {
        'input_size': latent_dim,
        'output_size': latent_dim,
        'hidden_size': 16,
        'num_layers': 2,
        'par_size': 2
    }

    prepare_models = PrepareModels(
        AE_checkpoint_path='model_weights/AdvAE',
        forecast_model_checkpoint_path='model_weights/seq2seq_model',
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        forecast_model_params=forecast_model_params,
        device=device
    )

    decoder = prepare_models.get_decoder()
    encoder = prepare_models.get_encoder()
    forecast_model = prepare_models.get_forecast_model()

    ##### Prepare data #####

    num_states_pr_sample = 256
    total_steps = 1024
    sample_time_ids = np.linspace(0, total_steps, num_states_pr_sample,
                                  dtype=int, endpoint=False)
    data = PrepareData(
        data_path='reduced_data',
        device=device,
        sample_time_ids=sample_time_ids,
        total_time_steps=total_steps
    )

    true_z_state, true_pars = data.get_data(state_case=110, par_case=110)
    hf_state = data.get_high_fidelity_state(true_z_state, decoder)

    space_obs_ids = torch.arange(0, 128, 15, device=device)
    time_obs_ids = torch.arange(0, 256, 30, device=device)
    time_obs_mask = torch.zeros(num_states_pr_sample, dtype=torch.bool, device=device)
    time_obs_mask[time_obs_ids] = 1

    space_observation_operator = lambda x: space_obs_operator(x, space_obs_ids)
    observations = space_observation_operator(hf_state)

    std_obs = 0.01
    observations += torch.randn(observations.shape, device=device) * std_obs

    ##### Prepare posterior #####

    posterior_params = {
        'decoder': decoder,
        'forecast_model': forecast_model,
        'device': device,
        'obs_operator': space_observation_operator,
        'std_obs': std_obs,
        'forecast_std': 2.,
        'latent_dim': latent_dim,
        'num_obs': len(space_obs_ids),
        'with_pars': True
    }

    HMC_params = {
        'num_samples': 1000,
        'step_size': 1.,
        'num_steps_per_sample': 5,
        'burn': 750,
        'integrator': hamiltorch.Integrator.IMPLICIT,
        'sampler': hamiltorch.Sampler.HMC_NUTS,
        'desired_accept_rate': 0.3
    }

    data_assimlation = DataAssimilation(
            observation_times=time_obs_ids,
            posterior_params=posterior_params
    )
    data_assimlation.HMC_params = HMC_params


    z_state, pars = data.get_data(state_case=0, par_case=0)

    z_history_init = z_state[:input_window_size, :]
    z_history_init = z_history_init.unsqueeze(1)

    pars_history_init = pars[:input_window_size, :]
    pars_history_init = pars_history_init.unsqueeze(1)

    ##### Run assimilation #####
    num_assimilation_steps = 200
    total_steps = num_assimilation_steps + input_window_size
    pbar = tqdm(range(input_window_size, total_steps), total=num_assimilation_steps)
    z_sol = z_history_init.clone()
    pars_sol = pars_history_init.clone()
    count_since_last_obs = 0
    for i in pbar:
        if time_obs_mask[i]:

            data_assimlation.set_posterior_information(
                observations=observations[i:i+1, :],
                pars=pars_sol[-1:],
                target_len=1,
                z_history=z_sol[-input_window_size:].detach(),
            )

            z = z_sol[-1].detach().requires_grad_()
            #z = torch.randn((1, latent_dim), device=device, requires_grad=True)
            pars = pars_sol[-1:].detach().requires_grad_()
            z_MAP, pars_MAP = data_assimlation.compute_z_MAP_with_observations(
                    z=z,
                    pars=pars,
                    num_iterations=15,
                    print_progress=True
            )
            z_sol = torch.cat((z_sol, z_MAP.unsqueeze(0)), dim=0)
            pars_sol = torch.cat((pars_sol, pars_MAP), dim=0)

            z_history_backward_correction = \
                z_sol[-(input_window_size+count_since_last_obs):-count_since_last_obs].detach().requires_grad_()
            left_BC = z_sol[-count_since_last_obs].detach().requires_grad_()
            right_BC = z_sol[-1].detach().requires_grad_()
            pars_backward_correction = pars[-1:].detach().requires_grad_()

            z_history, z_new, pars = data_assimlation.backward_correction(
                    z_history=z_history_backward_correction,
                    pars=pars_backward_correction,
                    target_len=count_since_last_obs,
                    left_BC=left_BC,
                    right_BC=right_BC,
                    num_iterations=10
            )
            z_sol[-(input_window_size+count_since_last_obs):-count_since_last_obs] = z_history
            z_sol[-count_since_last_obs:] = z_new
            pars_sol[-1:] = pars

            count_since_last_obs = 0

        else:
            count_since_last_obs += 1

            z = data_assimlation.compute_step_without_observations(
                    z_history=z_sol[-input_window_size:],
                    pars=pars_sol[-1:],
                    target_len=1
            )
            z_sol = torch.cat((z_sol, z), dim=0)
            pars_sol = torch.cat((pars_sol, pars_sol[-1:]), dim=0)

    pred_hf_state = data.get_high_fidelity_state(z_sol[:,0], decoder)
    pred_hf_state = pred_hf_state.detach().cpu().numpy()

    z_sol = z_sol.detach().cpu().numpy()
    pars_sol = pars_sol.detach().cpu().numpy()


    target_len = num_assimilation_steps
    z_pred_no_assimilation = data_assimlation.compute_step_without_observations(
            z_history=z_history_init,
            pars=pars_history_init,
            target_len=target_len
    )
    z_pred_no_assimilation = torch.cat((z_history_init, z_pred_no_assimilation), dim=0)

    pred_hf_no_assimilation = data.get_high_fidelity_state(
            z_pred_no_assimilation[:, 0],
            decoder
    )
    pred_hf_no_assimilation = pred_hf_no_assimilation.detach().cpu().numpy()
    z_pred_no_assimilation = z_pred_no_assimilation.detach().cpu().numpy()

    hf_state = hf_state.detach().cpu().numpy()
    true_z_state = true_z_state.detach().cpu().numpy()
    true_pars = true_pars.detach().cpu().numpy()

    assimilation_error = np.mean(np.abs(pred_hf_state - hf_state[0:total_steps]), axis=1)
    no_assimilation_error = np.mean(np.abs(pred_hf_no_assimilation - hf_state[0:total_steps]), axis=1)

    ##### Plotting #####
    plt.figure(figsize=(10, 15))
    plt.subplot(4,1,1)
    plt.plot(true_z_state[0:total_steps, 0], color='tab:blue', label='True')
    plt.plot(true_z_state[0:total_steps, 1], color='tab:blue')
    plt.plot(true_z_state[0:total_steps, 2], color='tab:blue')
    plt.plot(z_sol[:, 0, 0], color='tab:red', label='Assimilated')
    plt.plot(z_sol[:, 0, 1], color='tab:red')
    plt.plot(z_sol[:, 0, 2], color='tab:red')
    plt.plot(z_pred_no_assimilation[:, 0, 0], color='tab:green', label='No Assimilation')
    plt.plot(z_pred_no_assimilation[:, 0, 1], color='tab:green')
    plt.plot(z_pred_no_assimilation[:, 0, 2], color='tab:green')
    plt.xlabel('Time')
    plt.ylabel('z')
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(hf_state[0, :], color='tab:blue', label='True')
    plt.plot(hf_state[total_steps-1, :], color='tab:blue')
    plt.plot(pred_hf_state[0, :], color='tab:red', label='Assimilated')
    plt.plot(pred_hf_state[total_steps-1, :], color='tab:red')
    plt.plot(pred_hf_no_assimilation[0, :], color='tab:green', label='No Assimilation')
    plt.plot(pred_hf_no_assimilation[total_steps-1, :], color='tab:green')
    plt.text(25, np.max(hf_state[0, :]), 't=0')
    plt.text(50, np.max(hf_state[total_steps-1, :]), 't=120')
    plt.xlabel('Space')
    plt.ylabel('High-Fidelity State')
    plt.legend()

    plt.subplot(4,1,3)
    plt.semilogy(assimilation_error, color='tab:blue', label='Assimilation')
    plt.semilogy(no_assimilation_error, color='tab:green', label='No Assimilation')
    for i in range(time_obs_ids[0:total_steps].shape[0]):
        plt.axvline(x=time_obs_ids[i], color='black', alpha=0.5, linewidth=0.5)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')

    plt.subplot(4,1,4)
    plt.plot(true_pars[:, 0], color='tab:blue', label='True advection')
    plt.plot(true_pars[:, 1], color='tab:red', label='True diffusion')
    plt.plot(pars_sol[:, 0, 0], color='tab:blue', linestyle='--', label='Assimilated advection')
    plt.plot(pars_sol[:, 0, 1], color='tab:red', linestyle='--', label='Assimilated diffusion')
    for i in range(time_obs_ids[0:total_steps].shape[0]):
        plt.axvline(x=time_obs_ids[i], color='black', alpha=0.5, linewidth=0.5)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Parameters')

    plt.show()






