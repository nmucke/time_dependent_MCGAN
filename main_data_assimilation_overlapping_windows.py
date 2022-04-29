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
from training.train_transformer import TrainTransformer
from training.train_time_gan import TrainTimeGAN
from utils.time_series_utils import multistep_pred
from models.adv_diff_models.time_gan import Generator, Critic, ParCritic
from tqdm import tqdm
import hamiltorch
from utils.seed_everything import seed_everything

torch.set_default_tensor_type(torch.DoubleTensor)

torch.manual_seed(0)
np.random.seed(0)

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def obs_operator(data, obs_idx):
    return data[0, :, obs_idx]

class LatentPosterior():
    def __init__(self,
                 generator,
                 prior,
                 obs_noise,
                 print_progress=False
                 ):

        self.print_progress = print_progress
        self.latent_dim = generator.latent_dim

        self.generator = generator

        # Define prior distribution
        self.prior = prior
        self.prior_dist = torch.distributions.Normal(
                prior['mean'],
                prior['std']
        )

        # Define likelihood distribution
        self.obs_noise = obs_noise

    def prior_log_probability(self, z):
        # Compute prior
        return self.prior_dist.log_prob(z).sum()

    def likelihood_log_probability(self, gen_state, observations, obs_operator):
        # Compute likelihood
        gen_obs = obs_operator(gen_state)
        obs_error = observations - gen_obs

        likelihood = torch.distributions.Normal(
                torch.zeros(observations.shape),
                self.obs_noise['std']*torch.ones(observations.shape)
        ).log_prob(obs_error).sum()
        return likelihood

    def previous_latent_log_probability(
            self,
            gen_state,
            previous_latent_state,
    ):
        previous_latent_gen = previous_latent_state['obs_operator'](gen_state)
        previous_latent_score = torch.distributions.Normal(
                previous_latent_state['mean'],
                torch.tensor([1])#previous_latent_state['std']+1e-8
        ).log_prob(previous_latent_gen).sum()


        return previous_latent_state['weight']*previous_latent_score

    def posterior_log_probability(
            self,
            z,
            observations,
            obs_operator,
            previous_latent_state=None,
    ):

        # Compute prior
        z_prior_score = self.prior_log_probability(z)

        # Compute likelihood
        gen_state = self.generator(z.view(1, self.latent_dim))
        likelihood_score = self.likelihood_log_probability(
                gen_state,
                observations,
                obs_operator
        )

        if previous_latent_state is not None:
            previous_latent_score = self.previous_latent_log_probability(
                    gen_state,
                    previous_latent_state=previous_latent_state
            )
            return z_prior_score + likelihood_score + previous_latent_score
            #return previous_latent_score
        else:
            return z_prior_score + likelihood_score

    def compute_MAP(
            self,
            z,
            observations,
            obs_operator,
            previous_latent_state=None,
            num_iter=1000,
    ):
        optim = torch.optim.Adam([z], lr=.1)

        if self.print_progress:
            pbar = tqdm(
                    range(num_iter),
                    total=num_iter,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        else:
            pbar = range(num_iter)

        for _ in pbar:
            optim.zero_grad()

            negative_log_prob = -self.posterior_log_probability(
                    z,
                    observations,
                    obs_operator,
                    previous_latent_state=previous_latent_state
            )
            negative_log_prob.backward()
            optim.step()

            #if self.print_progress:
            #    pbar.set_postfix({"negative_log_prob": negative_log_prob.item()})

        return z

    def sample_latent_posterior(
            self,
            z,
            observations,
            obs_operator,
            previous_latent_state=None,
            HMC_params=None,
    ):

        posterior = lambda z: self.posterior_log_probability(
            z,
            observations,
            obs_operator,
            previous_latent_state=previous_latent_state
        )

        if z.shape[0] == 1:
            z = z.squeeze(0)

        if self.print_progress:
            z_samples = hamiltorch.sample(
                    log_prob_func=posterior,
                    params_init=z,
                    **HMC_params
            )
        else:
            with HiddenPrints():
                z_samples = hamiltorch.sample(
                        log_prob_func=posterior,
                        params_init=z,
                        **HMC_params
                )
        return torch.stack(z_samples)

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
    scaler_state.fit(data.reshape(num_train_data*num_time_steps, 3))

    scaler_pars = StandardScaler()
    scaler_pars.fit(pars)

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
    data = scaler_state.transform(data).reshape(1, 1000, 3)
    data = data[:, -num_time_steps:, :]
    data = np.swapaxes(data, 1, 2)
    data = torch.from_numpy(data).to(device)

    true_par = scaler_pars.transform(true_par)
    true_par = torch.from_numpy(true_par).to(device)

    num_skip_steps = 5
    full_obs_idx = range(0, num_time_steps, num_skip_steps)#(0, 1, 5, 10, 15, 20, 25, 40, 50)
    obs_mask = np.zeros(num_time_steps, dtype=bool)
    obs_mask[full_obs_idx] = True

    obs_std = 0.05
    noise_mean = torch.zeros(data.shape, device=device)
    noise_std = obs_std*torch.ones(data.shape, device=device)
    noise = torch.distributions.Normal(noise_mean, noise_std)
    data_obs = data+noise.sample().to(device)


    HMC_params = {'num_samples': 5000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 3000,
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
    '''
    progress_bar = tqdm(range(1, memory+1, num_skip_steps))
    for i in progress_bar:
        window_data_obs = data_obs[:, :, 0:memory]

        window_obs_mask = np.zeros(memory, dtype=bool)
        window_obs_mask[0:i] = obs_mask[0:i]
        time_obs_operator = lambda data: obs_operator(data, window_obs_mask).to(device)
        observations = time_obs_operator(window_data_obs)
        z_MAP = MAP_estimate(generator, observations,
                             latent_dim=latent_dim,
                             obs_std=obs_std,
                             obs_operator=time_obs_operator,
                             device=device
                             )

        noise_mean = torch.zeros(observations.shape, device=device)
        noise_std = obs_std * torch.ones(observations.shape, device=device)

        posterior_params['obs_operator'] = time_obs_operator
        posterior_params['observations'] = observations
        posterior_params['noise_mean'] = noise_mean
        posterior_params['noise_std'] = noise_std

        z_samples = hamiltonian_MC(
                z_init=z_MAP[0],
                posterior_params=posterior_params,
                HMC_params=HMC_params
        )

        gen_samples, gen_pars = generator(z_samples, output_pars=True)
        gen_mean = gen_samples.mean(dim=0).detach().cpu().numpy()
        gen_std = gen_samples.std(dim=0).detach().cpu().numpy()

        par_list.append(gen_pars.detach().cpu().numpy())
        pred_list.append(gen_mean)
        std_list.append(gen_std)
    '''

    prior = {'mean': torch.zeros(latent_dim, device=device),
             'std': torch.ones(latent_dim, device=device)}
    obs_noise = {'mean': 0.,
                 'std': obs_std}


    latent_posterior = LatentPosterior(
            generator=generator,
            prior=prior,
            obs_noise=obs_noise,
            print_progress=False
    )

    z_MAP = torch.randn(1, latent_dim, requires_grad=True, device=device)
    progress_bar = tqdm(range(memory, num_time_steps, memory))
    for i in progress_bar:
        window_data_obs = data_obs[:, :, i-memory+1:i+1]

        window_obs_mask = obs_mask[i-memory+1:i+1]
        time_obs_operator = lambda data: obs_operator(data, window_obs_mask).to(device)
        observations = time_obs_operator(window_data_obs)

        if i > 10000:
            previous_latent_state_mask = np.zeros(memory, dtype=bool)
            previous_latent_state_mask[0:-num_skip_steps] = True
            previous_latent_state_operator = lambda data: obs_operator(data, previous_latent_state_mask).to(device)
            previous_latent_state_mean = gen_mean[:,num_skip_steps:]
            previous_latent_state_std = gen_std[:,num_skip_steps:]
            previous_latent_state = {
                    'mean': previous_latent_state_mean,
                    'std': previous_latent_state_std,
                    'obs_operator': previous_latent_state_operator,
                    'weight': .1
            }
        else:
            previous_latent_state = None

        z_MAP = latent_posterior.compute_MAP(
                z=z_MAP,
                observations=observations,
                obs_operator=time_obs_operator,
                num_iter=1000,
                previous_latent_state=previous_latent_state
        )

        z_samples = latent_posterior.sample_latent_posterior(
                z=z_MAP,
                observations=observations,
                obs_operator=time_obs_operator,
                HMC_params=HMC_params,
                previous_latent_state=previous_latent_state
        )

        latent_posterior.prior['mean'] = z_samples.mean(dim=0)
        latent_posterior.prior['std'] = z_samples.std(dim=0)

        gen_samples, gen_pars = generator(z_samples, output_pars=True)
        gen_mean = gen_samples.mean(dim=0).detach()
        gen_std = gen_samples.std(dim=0).detach()

        par_list.append(gen_pars.detach().cpu().numpy())
        pred_list.append(gen_mean.cpu().numpy())
        std_list.append(gen_std.cpu().numpy())
        #par_list.append(gen_pars.detach().cpu().numpy())
        #pred_list.append(gen_mean.cpu().numpy()[:,0:num_skip_steps])
        #std_list.append(gen_std.cpu().numpy()[:,0:num_skip_steps])

        state_MAP, par_MAP = generator(z_MAP, output_pars=True)
        state_MAP_list.append(state_MAP[0].detach().cpu().numpy())
        par_MAP_list.append(par_MAP[0].detach().cpu().numpy())


    plot_par = np.asarray(par_list)
    std_list = np.asarray(std_list)
    plot_pred = np.asarray(pred_list)
    plot_state_MAP = np.asarray(state_MAP_list)
    plot_par_MAP = np.asarray(par_MAP_list)

    #lol = np.swapaxes(plot_pred, 1, 2)
    #lol = lol.reshape(5 * 90, 3)
    #plt.figure()
    #plt.plot(lol[:, 0])
    #plt.plot(data[0, 0, :num_time_steps])
    #plt.show()
    #pdb.set_trace()


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
        plt.plot(range(i*memory,i*memory+memory), plot_pred[i, 0, :], '--',
                 color='tab:blue', linewidth=3)
        plt.fill_between(range(i*memory,i*memory+memory),
                         plot_pred[i, 0, :] - 2*std_list[i, 0, :],
                         plot_pred[i, 0, :] + 2*std_list[i, 0, :],
                         alpha=0.25,
                         color='tab:blue')

        plt.plot(range(i*memory,i*memory+memory), plot_state_MAP[i, 0, :], '--',
                 color='tab:green', linewidth=3)
    #for i in range(plot_pred.shape[0]):
        #plt.axvline(x=i*memory, color='black', linestyle='-', linewidth=1)
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