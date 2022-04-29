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

torch.set_default_tensor_type(torch.DoubleTensor)

torch.manual_seed(0)
np.random.seed(0)


def obs_operator(data, obs_idx):
    return data[0, :, obs_idx]


def MAP_estimate(generator, data, latent_dim, obs_idx, obs_std, device):
    z = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optim = torch.optim.Adam([z], lr=.1)

    true_obs = data#obs_operator(data, obs_idx).to(device)
    num_iter = 2500

    progress_bar = tqdm(
            range(num_iter),
            total=num_iter,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for _ in progress_bar:
        optim.zero_grad()
        gen_data = generator(z)
        gen_obs = obs_operator(gen_data, obs_idx)
        error = 1/obs_std/obs_std*torch.pow(torch.linalg.norm(gen_obs - true_obs), 2) \
                + torch.pow(torch.linalg.norm(z), 2)
        error.backward()
        optim.step()

        progress_bar.set_postfix({"Error": error.item()})

    return z

def latent_posterior(z, generator,
                     obs_operator, observations,
                     previous_states, previous_state_obs_operator, previous_state_std,
                     prior_mean, prior_std,
                     noise_mean, noise_std):
    z_prior_score = torch.distributions.Normal(prior_mean,
                                               prior_std).log_prob(z).sum()

    gen_state = generator(z.view(1, len(z)))
    gen_state = obs_operator(gen_state)
    error = observations - gen_state

    reconstruction_score = torch.distributions.Normal(noise_mean,
                                      noise_std).log_prob(error).sum()
    return z_prior_score + reconstruction_score

def hamiltonian_MC(z_init,posterior_params, HMC_params):
    posterior = lambda z: latent_posterior(z, **posterior_params)
    z_samples = hamiltorch.sample(log_prob_func=posterior,
                                  params_init=z_init,
                                  **HMC_params)
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

    generat_new_data = False
    continue_training = True
    train = True
    cuda = True
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    num_time_steps = 500
    tmax = 40
    data = []
    num_train_data = 1500

    if generat_new_data:
        #rho_list = np.linspace(20, 30, num_train_data)
        rho_list = np.random.normal(28, 1.5, num_train_data)
        beta_list = np.random.normal(2.667, 0.05, num_train_data)
        sigma = np.random.normal(10, 1, num_train_data)
        for i in range(num_train_data):
            data.append(lorenz_data(
                    rho=rho_list[i],
                    beta=beta_list[i],
                    sigma=sigma[i],
                    tmax=tmax,
                    num_time_steps=num_time_steps*2
                    )
            )
        data = np.asarray(data)
        data = data[:, -num_time_steps:, :]
        pars = np.stack((rho_list, beta_list, sigma), axis=1)

        np.save('lorenz_data.npy', data)
        np.save('pars_lorenz_data.npy', pars)
    else:
        data = np.load('lorenz_data.npy')
        pars = np.load('pars_lorenz_data.npy')

    memory = 52

    scaler_state = MinMaxScaler()
    data = scaler_state.fit_transform(data.reshape(num_train_data*num_time_steps, 3))
    data = data.reshape(num_train_data, num_time_steps, 3)

    scaler_pars = StandardScaler()
    pars = scaler_pars.fit_transform(pars)


    train_data = []
    train_pars = []
    for j in range(num_train_data):
        for i in range(data.shape[1]-memory):
            train_data.append(data[j,i:i+memory,:])
            train_pars.append([pars[j]])
    train_data = np.asarray(train_data)
    train_data = train_data.reshape(train_data.shape[0], memory, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    train_data = torch.from_numpy(train_data)

    train_pars = np.asarray(train_pars)
    train_pars = train_pars.reshape(train_pars.shape[0], 3)
    train_pars = torch.from_numpy(train_pars)

    dataset = TensorDataset(train_data, train_pars)
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            num_workers=4,
                            shuffle=True)

    pretrain_dataset = TensorDataset(torch.tensor(pars))
    pretrain_dataloader = DataLoader(pretrain_dataset,
                                     batch_size=512,
                                     num_workers=2,
                                     shuffle=True,
                                     drop_last=True)

    latent_dim = 8

    generator = Generator(
        latent_dim=latent_dim,
        par_dim=3,
        hidden_channels=[64, 32, 16, 3],
        par_latent_dim=3,
        par_hidden_neurons=[4, 4, 4]
    ).to(device)

    critic = Critic(
        par_dim=3,
        hidden_channels=[3, 16, 32, 64],
    ).to(device)
    par_critic = ParCritic(
        par_dim=3,
        par_hidden_neurons=[4, 4, 4],
    ).to(device)

    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=1e-4)

    par_generator_optimizer = torch.optim.RMSprop(generator.par_generator.parameters(), lr=1e-4)
    par_critic_optimizer = torch.optim.RMSprop(par_critic.parameters(), lr=1e-4)

    if continue_training:
        checkpoint = torch.load('TimeGAN')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    if train:
        trainer = TrainTimeGAN(
                generator=generator,
                critic=critic,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                par_generator=generator.par_generator,
                par_critic=par_critic,
                par_generator_optimizer=par_generator_optimizer,
                par_critic_optimizer=par_critic_optimizer,
                n_critic=3,
                gamma=10,
                save_string='TimeGAN',
                n_epochs=5000,
                device=device)
        trainer.train(dataloader=dataloader, pretrain_dataloader=None)#pretrain_dataloader)
        generator.eval()
    else:
        checkpoint = torch.load('TimeGAN')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

    '''
    z_samples = torch.randn(1000, 10).to(device)
    _, par_gen_samples = generator(z_samples, output_pars=True)
    par_gen_samples = par_gen_samples.cpu().detach().numpy()
    plt.figure()
    plt.hist(rho_list, bins=50)
    plt.hist(par_gen_samples, bins=50, alpha=0.5)
    plt.show()
    pdb.set_trace()
    '''
    true_par = np.array([[29, 2.6, 11]])
    data = lorenz_data(
            rho=true_par[0, 0],
            beta=true_par[0, 1],
            sigma=true_par[0, 2],
            tmax=40,
            num_time_steps=1000
    )

    data = scaler_state.transform(data)
    data = data.reshape(1, 1000, 3)
    data = np.swapaxes(data, 1, 2)
    data = data[:, :, 600:600+memory]
    data = torch.from_numpy(data)
    data = data.to(device)

    true_par = scaler_pars.transform(true_par)
    true_par = torch.from_numpy(true_par)
    true_par = true_par.to(device)

    obs_idx = range(0, 52, 1)#(0, 1, 5, 10, 15, 20, 25, 40, 50)

    obs_std = 0.05
    observations = obs_operator(data, obs_idx).to(device)
    noise_mean = torch.zeros(observations.shape, device=device)
    noise_std = obs_std*torch.ones(observations.shape, device=device)
    noise = torch.distributions.Normal(noise_mean, noise_std)
    observations += noise.sample().to(device)


    z_MAP = MAP_estimate(generator, observations,
                         latent_dim=latent_dim,
                         obs_idx=obs_idx,
                         obs_std=obs_std,
                         device=device
                         )

    posterior_params = {'generator': generator,
                        'obs_operator': lambda x: obs_operator(x, obs_idx),
                        'observations': observations,
                        'prior_mean': torch.zeros(latent_dim, device=device),
                        'prior_std': torch.ones(latent_dim, device=device),
                        'noise_mean': noise_mean,
                        'noise_std': noise_std
                        }
    HMC_params = {'num_samples': 5000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 3500,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3
                  }


    z_samples = hamiltonian_MC(
            z_init=z_MAP[0],
            posterior_params=posterior_params,
            HMC_params=HMC_params
        )

    gen_samples, gen_pars = generator(z_samples, output_pars=True)
    gen_mean = gen_samples.mean(dim=0).detach().cpu().numpy()
    gen_std = gen_samples.std(dim=0).detach().cpu().numpy()

    gen_pars = gen_pars.detach().cpu().numpy()

    gen_data = generator(z_MAP)
    plt.figure(figsize=(20,12))
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.plot(gen_mean[i,:], label='Mean', color='red', linewidth=2)
        plt.fill_between(range(memory),
                         gen_mean[i,:] - 2*gen_std[i,:],
                         gen_mean[i,:] + 2*gen_std[i,:],
                         alpha=0.2,
                         color='red')
        plt.plot(list(obs_idx), observations[i].detach().cpu().numpy(),
                 '.b', markersize=10)
        plt.plot(data[0,i,:].numpy(), label='True', color='black', linewidth=2)
        plt.legend()
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.hist(gen_pars[:,i], bins=50)
        plt.axvline(true_par[0,i], color='black', linewidth=2)

    plt.show()

