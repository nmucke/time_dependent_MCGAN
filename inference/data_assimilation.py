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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import hamiltorch
import sys
import os

torch.set_default_dtype(torch.float32)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def space_obs_operator(state, ids, space_dim=1):
    return torch.index_select(state, space_dim, ids)

class PosteriorDistribution():
    def __init__(
            self,
            decoder,
            forecast_model,
            device,
            obs_operator,
            std_obs,
            forecast_std,
            latent_dim,
            num_obs,
            with_pars=False
    ):
        self.decoder = decoder
        self.device = device
        self.forecast_model = forecast_model
        self.obs_operator = obs_operator
        self.latent_dim = latent_dim
        self.num_obs = num_obs
        self.with_pars = with_pars


        self.zeros_latent_size = torch.zeros((latent_dim, ), device=device)
        self.ones_latent_size = torch.ones((latent_dim, ), device=device)

        self.zeros_obs_size = torch.zeros((self.num_obs, ), device=device)
        self.ones_obs_size = torch.ones((self.num_obs, ), device=device)

        self.zeros_pars_shape = torch.zeros((1, 1, 2), device=device)
        self.ones_pars_shape = torch.ones((1, 1, 2), device=device)

        self.forecast_std = forecast_std
        self.std_obs = std_obs


        self.prior_distribution = torch.distributions.Normal(
            self.zeros_latent_size,
            self.ones_latent_size
        )

        self.forecast_likelihood_distrubution = torch.distributions.Normal(
                self.zeros_latent_size,
                self.forecast_std * self.ones_latent_size
        )

        self.reconstruction_likelihood_distribution = torch.distributions.Normal(
                self.zeros_obs_size,
                self.std_obs * self.ones_obs_size
        )



        self.HMC_params = {
            'num_samples': 1000,
            'step_size': 1.,
            'num_steps_per_sample': 5,
            'burn': 750,
            'integrator': hamiltorch.Integrator.IMPLICIT,
            'sampler': hamiltorch.Sampler.HMC_NUTS,
            'desired_accept_rate': 0.3
            }

    def log_prior(self, z):
        log_prior = self.prior_distribution.log_prob(z).sum()
        return log_prior

    def forecast_log_likelihood(self, z, z_history, pars, target_len=1):
        z_pred = self.forecast_model(
                input_tensor=z_history,
                target_len=target_len,
                pars=pars
        )
        residual = z_pred - z
        forecast_log_likelihood = \
            self.forecast_likelihood_distrubution.log_prob(residual).sum()

        return forecast_log_likelihood

    def reconstruction_log_likelihood(self, z, observations):
        generated_obs = self.decoder(z.view(1, z.shape[-1]))
        generated_obs = self.obs_operator(generated_obs)

        residual = generated_obs - observations

        reconstruction_log_likelihood = \
            self.reconstruction_likelihood_distribution.log_prob(residual).sum()

        return reconstruction_log_likelihood

    def pars_log_prior(self, pars):
        pars_log_prior = torch.distributions.Normal(
                self.zeros_pars_shape,
                5*self.ones_pars_shape
            ).log_prob(pars).sum()

        #if pars[0, 0, 0] < 0 or pars[0, 0, 0] > 1 or pars[0, 0, 1] < 0 or pars[0, 0, 1] > 1:
        #    pars_log_prior = -torch.tensor(1e8, device=self.device)
        #else:
        #    pars_log_prior = torch.tensor(0., device=self.device)
        return pars_log_prior

    def latent_log_posterior(self, z, pars=None):
        log_prior = self.log_prior(z)
        if pars is not None:
            forecast_log_likelihood = self.forecast_log_likelihood(
                    z=z,
                    z_history=self.z_history,
                    pars=pars,
                    target_len=self.target_len
            )
        else:
            forecast_log_likelihood = self.forecast_log_likelihood(
                    z=z,
                    z_history=self.z_history,
                    pars=self.pars,
                    target_len=self.target_len
            )
        reconstruction_log_likelihood = self.reconstruction_log_likelihood(
                z=z,
                observations=self.observations
        )

        if self.with_pars:
            pars_log_prior = self.pars_log_prior(pars)
            return log_prior + forecast_log_likelihood \
                   + reconstruction_log_likelihood + pars_log_prior
        else:
            return log_prior + forecast_log_likelihood + reconstruction_log_likelihood

    def set_posterior_information(self, observations, z_history, pars, target_len=1):
        self.z_history = z_history
        self.pars = pars
        self.target_len = target_len
        self.observations = observations

    def compute_maximum_a_posteriori(self, z, pars=None, num_iterations=100, print_progress=False):
        """Compute MAP estimate of z given observations and history."""
        if self.with_pars:
            optimizer = torch.optim.LBFGS(
                    [z, pars],
                    history_size=5,
                    max_iter=5,
                    line_search_fn="strong_wolfe",
            )
        else:
            optimizer = torch.optim.LBFGS(
                    [z],
                    history_size=5,
                    max_iter=5,
                    line_search_fn="strong_wolfe",
            )

        if self.device.type == 'cuda':
            self.forecast_model.train()
        for i in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                log_posterior = -self.latent_log_posterior(z, pars)
                log_posterior.backward(retain_graph=True)
                return log_posterior
            #optimizer.zero_grad()
            #log_posterior = -self.latent_log_posterior(z)
            #log_posterior.backward(retain_graph=True)
            optimizer.step(closure)

        return z, pars

    def MCMC_sample_from_latent_posterior(self, z_init, print_progress=True):

        if print_progress:
            z_samples = hamiltorch.sample(
                    log_prob_func=self.latent_log_posterior,
                    params_init=z_init,
                    **self.HMC_params
            )
        else:
            with HiddenPrints():
                z_samples = hamiltorch.sample(
                        log_prob_func=self.latent_log_posterior,
                        params_init=z_init,
                        **self.HMC_params
                )
        return z_samples

class DataAssimilation(PosteriorDistribution):
    def __init__(
            self,
            observation_times,
            posterior_params,
            ):
        super().__init__(**posterior_params)
        self.observation_times = observation_times

    def compute_latent_initial_condition(self, z_init):
        z_init = self.compute_maximum_a_posteriori(z_init)
        return z_init

    def latent_to_high_fidelity(self, z):
        return self.decoder(z.view(-1, z.shape[-1]))

    def compute_step_without_observations(self, z_history, pars, target_len=1):
        z_new = self.forecast_model(
                input_tensor=z_history,
                target_len=target_len,
                pars=pars
        )
        return z_new

    def compute_z_MAP_with_observations(self, z, pars=None, num_iterations=100, print_progress=False):
        return self.compute_maximum_a_posteriori(z, pars, num_iterations, print_progress)

    def compute_z_samples_with_observations(self, z_init):
        z_samples = self.MCMC_sample_from_latent_posterior(
                z_init=z_init,
                print_progress=False
        )
        return z_samples

    def backward_correction(
            self, z_history, pars, target_len, left_BC, right_BC, num_iterations
    ):

        self.backward_correction_loss_function = nn.MSELoss()

        if self.device.type == 'cuda':
            self.forecast_model.train()
        z_opt = z_history.detach().requires_grad_()
        if self.with_pars:
            optimizer = torch.optim.LBFGS(
                    [z_opt, pars],
                    history_size=5,
                    max_iter=5,
                    line_search_fn="strong_wolfe",
            )
        else:
            optimizer = torch.optim.LBFGS(
                    [z_opt],
                    history_size=5,
                    max_iter=5,
                    line_search_fn="strong_wolfe",
            )

        for i in range(num_iterations):
            def closure():
                optimizer.zero_grad()

                loss = self.BC_residual(
                        z_history=z_opt,
                        pars=pars,
                        target_len=target_len,
                        left_BC=left_BC,
                        right_BC=right_BC
                )
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)

        z_new = self.compute_step_without_observations(
                z_history=z_opt,
                pars=pars,
                target_len=target_len
        )

        return z_history, z_new, pars

    def BC_residual(self, z_history, pars, target_len, left_BC, right_BC):

        z_new = self.compute_step_without_observations(
                z_history=z_history,
                pars=pars,
                target_len=target_len
        )
        return self.backward_correction_loss_function(left_BC, z_history[-1]) \
               + self.backward_correction_loss_function(right_BC, z_new[-1])


class PrepareData():
    def __init__(self, data_path, sample_time_ids, total_time_steps, device):
        self.data_path = data_path
        self.device = device
        self.sample_time_ids = sample_time_ids
        self.total_time_steps = total_time_steps

        self.data = np.load(f'{data_path}.npy')
        self.data = torch.tensor(self.data, dtype=torch.get_default_dtype())
        self.data = self.data[:, self.sample_time_ids]

        self.data_pars = np.load(f'{data_path}_pars.npy')
        self.data_pars = self.data_pars.reshape(-1, 2)
        self.data_pars = StandardScaler().fit_transform(self.data_pars)
        self.data_pars = self.data_pars.reshape(10000, total_time_steps, 2)
        self.data_pars = torch.tensor(self.data_pars, dtype=torch.get_default_dtype())
        self.data_pars = self.data_pars[:, self.sample_time_ids]

    def get_data(self, state_case, par_case):
        state = self.data[state_case, :, :]

        pars = self.data_pars[par_case]
        return state.to(self.device), pars.to(self.device)

    def get_high_fidelity_state(self, state, decoder):
        num_time_steps = state.shape[1]

        state = decoder(state)

        return state

class PrepareModels():
    def __init__(self,
                 AE_checkpoint_path,
                 forecast_model_checkpoint_path,
                 encoder_params,
                 decoder_params,
                 forecast_model_params,
                 device
                 ):
        self.AE_checkpoint_path = AE_checkpoint_path
        self.forecast_model_checkpoint_path = forecast_model_checkpoint_path

        self.AE_checkpoint = torch.load(self.AE_checkpoint_path)
        self.forecast_model_checkpoint = torch.load(self.forecast_model_checkpoint_path)

        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.forecast_model_params = forecast_model_params

        self.device = device

    def get_encoder(self):
        encoder = models.Encoder(**self.encoder_params)
        encoder.load_state_dict(self.AE_checkpoint['encoder_state_dict'])
        encoder.to(self.device)
        encoder.eval()
        return encoder

    def get_decoder(self):
        decoder = models.Decoder(**self.decoder_params)
        decoder.load_state_dict(self.AE_checkpoint['decoder_state_dict'])
        decoder.to(self.device)
        decoder.eval()
        return decoder

    def get_forecast_model(self):
        forecast_model = time_models.lstm_seq2seq(**self.forecast_model_params)
        forecast_model.load_state_dict(
                self.forecast_model_checkpoint['model_state_dict'])
        forecast_model.to(self.device)
        forecast_model.eval()
        return forecast_model
