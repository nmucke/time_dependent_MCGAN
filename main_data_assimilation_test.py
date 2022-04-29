import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb
from models.adv_diff_models.transformer import TransAm
from scipy.integrate import solve_ivp
import data_handling.data_handling_utils as utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.adv_diff_models.time_gan import Generator, Critic, ParCritic
from tqdm import tqdm
import hamiltorch
from utils.seed_everything import seed_everything
from inference import posterior, compute_posterior

torch.set_default_tensor_type(torch.DoubleTensor)

def obs_operator(data, obs_idx):
    return data[0, :, obs_idx]


def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def lorenz_data(sigma=10, beta=2.667, rho=28, num_time_steps=500, tmax=5):
    # Lorenz paramters and initial conditions.

    u0, v0, w0 = 0, 1, 1.05


    # Integrate the Lorenz equations.
    soln = solve_ivp(lambda t,x: lorenz(t, x, sigma, beta, rho), (0, tmax), (u0, v0, w0),
                     dense_output=True)
    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, tmax, num_time_steps)
    x, y, z = soln.sol(t)

    return np.stack((x,y,z), axis=1)

if __name__ == '__main__':
    seed_everything(1)

    cuda = False
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    num_time_steps = 500
    tmax = 40
    data = []
    num_train_data = 1500

    data = np.load('lorenz_data.npy')
    pars = np.load('pars_lorenz_data.npy')

    memory = 52

    scaler_state = MinMaxScaler()
    data = scaler_state.fit(data.reshape(num_train_data*num_time_steps, 3))

    scaler_pars = StandardScaler()
    pars = scaler_pars.fit(pars)


    latent_dim = 8

    generator = Generator(
        latent_dim=latent_dim,
        par_dim=3,
        hidden_channels=[64, 32, 16, 3],
        par_latent_dim=3,
        par_hidden_neurons=[4, 4, 4]
    ).to(device)

    checkpoint = torch.load('TimeGAN')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    true_par = np.array([[29, 2.6, 11]])
    data = lorenz_data(
            rho=true_par[0, 0],
            beta=true_par[0, 1],
            sigma=true_par[0, 2],
            tmax=40,
            num_time_steps=1000
    )
    num_time_steps = 500
    data = scaler_state.transform(data)
    data = data.reshape(1, 1000, 3)
    data = data[:, -num_time_steps:, :]
    data = np.swapaxes(data, 1, 2)
    data = torch.from_numpy(data)
    data = data.to(device)

    true_par = scaler_pars.transform(true_par)
    true_par = torch.from_numpy(true_par)
    true_par = true_par.to(device)

    num_skip_steps = 5
    full_obs_idx = range(0, num_time_steps, num_skip_steps)#(0, 1, 5, 10, 15, 20, 25, 40, 50)
    obs_mask = np.zeros(num_time_steps, dtype=bool)
    obs_mask[full_obs_idx] = True



    obs_std = 0.05
    noise_mean = torch.zeros(data.shape, device=device)
    noise_std = obs_std*torch.ones(data.shape, device=device)
    noise = torch.distributions.Normal(noise_mean, noise_std)
    data_obs = data+noise.sample().to(device)


    HMC_params = {'num_samples': 1000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 750,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3
                  }

    posterior_params = {'generator': generator,
                        'prior_mean': torch.zeros(latent_dim,
                                                  device=device),
                        'prior_std': torch.ones(latent_dim, device=device),
                        }

    pred_list = []
    std_list = []
    par_list = []
    obs_list = []
    par_MAP_list = []
    state_MAP_list = []

    data_noise = {
        'mean': torch.zeros(data_obs[:, :, 0:memory].shape, device=device),
        'std': obs_std*torch.ones(data_obs[:, :, 0:memory].shape, device=device)
    }
    latent_posterior = posterior.LatentPosterior(
            generator=generator,
            data_noise=data_noise,
            device=device,
            obs_operator=obs_operator,
    )

    jump = 5
    progress_bar = tqdm(range(memory, num_time_steps, jump))
    z_MAP = torch.randn(1, latent_dim, requires_grad=True, device=device)
    last_state = None
    for i in progress_bar:
        #if i > memory:
        #    last_state = {}
        #    last_state['mean'] = gen_mean[:, -memory//2:]
        #    last_state['std'] = gen_std[:, -memory//2:]

        #    last_state_obs_mask = np.zeros(memory, dtype=bool)
        #    last_state_obs_mask[0:memory//2] = True
        #    last_state['obs_operator'] = lambda x: obs_operator(x, last_state_obs_mask)

        window_data_obs = data_obs[:, :, i-memory+1:i+1]

        window_obs_mask = obs_mask[i-memory+1:i+1]
        time_obs_operator = lambda data: obs_operator(data, window_obs_mask).to(device)
        observations = time_obs_operator(window_data_obs)
        latent_posterior.obs_operator = time_obs_operator

        noise_mean = torch.zeros(observations.shape, device=device)
        noise_std = obs_std * torch.ones(observations.shape, device=device)

        latent_posterior.noise_mean = noise_mean
        latent_posterior.noise_std = noise_std
        z_MAP = compute_posterior.MAP_estimate(
                observations=observations,
                posterior=latent_posterior,
                num_iter=2000,
                print_progress=False,
                last_state=last_state,
        )
        z_samples = compute_posterior.hamiltonian_monte_carlo(
                z_init=z_MAP[0],
                observations=observations,
                posterior=latent_posterior,
                HMC_params=HMC_params,
                print_progress=False,
                last_state=last_state,
        )

        gen_samples, gen_pars = generator(z_samples, output_pars=True)
        gen_mean = gen_samples.mean(dim=0)
        gen_std = gen_samples.std(dim=0)

        par_list.append(gen_pars.detach().cpu().numpy())
        pred_list.append(gen_mean.detach().cpu().numpy())
        std_list.append(gen_std.detach().cpu().numpy())

        state_MAP, par_MAP = generator(z_MAP, output_pars=True)
        state_MAP_list.append(state_MAP[0].detach().cpu().numpy())
        par_MAP_list.append(par_MAP[0].detach().cpu().numpy())

        if i == memory:
            pred = gen_mean.detach().cpu().numpy()
            std = gen_std.detach().cpu().numpy()
        else:
            pred = np.concatenate((pred, gen_mean.detach().cpu().numpy()[:,-jump:]), axis=1)
            std = np.concatenate((std, gen_std.detach().cpu().numpy()[:,-jump:]), axis=1)

    plt.figure(figsize=(15, 10))
    plt.plot(range(0,num_time_steps, num_skip_steps), data_obs[0, 0, obs_mask],
             '.', color='tab:orange', markersize=10, label='Observations')
    plt.plot(range(0,num_time_steps), data[0, 0, :num_time_steps], 'black',
             linewidth=2, label='True')
    plt.plot(pred[0, :], '--', color='tab:blue', linewidth=3)
    plt.fill_between(range(pred.shape[1]),
                     pred[0, :] - 2 * std[0, :],
                     pred[0, :] + 2 * std[0, :],
                     alpha=0.25,
                     color='tab:blue')
    plt.show()
    pdb.set_trace()

    plot_par = np.asarray(par_list)
    std_list = np.asarray(std_list)
    plot_pred = np.asarray(pred_list)
    plot_state_MAP = np.asarray(state_MAP_list)
    plot_par_MAP = np.asarray(par_MAP_list)

    plt.figure(figsize=(15, 10))

    plt.plot(range(0,num_time_steps, num_skip_steps), data_obs[0, 0, obs_mask],
             '.', color='tab:orange', markersize=10, label='Observations')
    plt.plot(range(0,num_time_steps), data[0, 0, :num_time_steps], 'black',
             linewidth=2, label='True')

    plt.plot(range(0, memory), plot_pred[0, 0, :], '--',
             color='tab:blue', linewidth=3, label='Prediction')
    plt.fill_between(range(0,memory),
                     plot_pred[0, 0, :] - 2*std_list[0, 0, :],
                     plot_pred[0, 0, :] + 2*std_list[0, 0, :],
                     alpha=0.25,
                     color='tab:blue')

    plt.plot(range(0, memory), plot_state_MAP[0, 0, :], '--',
             color='tab:green', linewidth=3, label='MAP')

    for i in range(1, plot_pred.shape[0]):
        plt.plot(range(i*memory//2,i*memory//2+memory), plot_pred[i, 0, :], '--',
                 color='tab:blue', linewidth=3)
        plt.fill_between(range(i*memory//2,i*memory//2+memory),
                         plot_pred[i, 0, :] - 2*std_list[i, 0, :],
                         plot_pred[i, 0, :] + 2*std_list[i, 0, :],
                         alpha=0.25,
                         color='tab:blue')

        plt.plot(range(i*memory//2,i*memory//2+memory), plot_state_MAP[i, 0, :], '--',
                 color='tab:green', linewidth=3)
    for i in range(plot_pred.shape[0], 2):
        plt.axvline(x=i*memory//2, color='black', linestyle='-', linewidth=1)
    plt.grid()
    plt.legend()

    plt.savefig(f'lorenz_state_estimation_{num_skip_steps}_{obs_std}.png')
    plt.show()

    plot_par = plot_par.reshape(plot_par.shape[0]*plot_par.shape[1], 3)
    plt.figure(figsize=(15, 7.5))
    plt.subplot(1, 3, 1)
    plt.hist(plot_par[:, 0], bins=50, density=True, label='rho')
    plt.axvline(x=true_par[0, 0], color='black', linestyle='-', linewidth=3)
    plt.axvline(x=np.mean(plot_par[:, 0], axis=0), color='tab:orange', linestyle='-', linewidth=3)
    plt.axvline(x=np.mean(plot_par_MAP[:, 0], axis=0), color='tab:green', linestyle='-', linewidth=3)
    plt.subplot(1, 3, 2)
    plt.hist(plot_par[:, 1], bins=50, density=True, label='beta')
    plt.axvline(x=true_par[0, 1], color='black', linestyle='-', linewidth=3)
    plt.axvline(x=np.mean(plot_par[:, 1], axis=0), color='tab:orange', linestyle='-', linewidth=3)
    plt.axvline(x=np.mean(plot_par_MAP[:, 1], axis=0), color='tab:green', linestyle='-', linewidth=3)
    plt.subplot(1, 3, 3)
    plt.hist(plot_par[:, 2], bins=50, density=True, label='phi')
    plt.axvline(x=true_par[0, 2], color='black', linestyle='-', linewidth=3)
    plt.axvline(x=np.mean(plot_par[:, 2], axis=0), color='tab:orange', linestyle='-', linewidth=3)
    plt.axvline(x=np.mean(plot_par_MAP[:, 2], axis=0), color='tab:green', linestyle='-', linewidth=3)
    plt.savefig(f'lorenz_parameter_estimation_{num_skip_steps}_{obs_std}.png')
    plt.show()