import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import hamiltorch

def cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def init_condition_gaussian(coef_vec, x_vec):
    mu = coef_vec[0]
    sigma = coef_vec[1]
    return np.exp(-np.power(x_vec - mu, 2.) / (2 * np.power(sigma, 2.)))


class LogLikelihood(nn.Module):
    def __init__(self, decoder, obs_operator, noise_distribution):
        super(LogLikelihood, self).__init__()

        self.decoder = decoder
        self.obs_operator = obs_operator
        self.noise_distribution = noise_distribution


    def forward(self, pred, obs):
        """
        pred: (batch_size, latent_dim)
        obs: (1, obs_dim)
        """
        pred = self.decoder(pred)
        pred = self.obs_operator(pred)

        discrepancy = pred - obs.repeat(pred.shape[0], 1)

        log_likelihood = self.noise_distribution.log_prob(discrepancy).sum(dim=1)

        return log_likelihood


class ParticleFilter():
    def __init__(
            self,
            forecast_model,
            num_particles,
            log_likelihood,
            weight_update='bootstrap',
            x_vec=np.linspace(-1, 1, 128),
            model_forecast_std=0.1,
            par_std=0.1,
            device='cpu'
    ):
        self.forecast_model = forecast_model
        self.num_particles = num_particles
        self.log_likelihood = log_likelihood
        self.weight_update = weight_update
        self.x_vec = x_vec
        self.device = device
        self.model_forecast_std = model_forecast_std
        self.par_std = par_std
        self.latent_dim = self.forecast_model.latent_dim
        self.pars_dim = self.forecast_model.pars_dim
        self.smoothing_factor = torch.tensor(.1)

        self.t_idx = 0.

        self.model_forecast_error_distribution = torch.distributions.Normal(
            torch.zeros(self.latent_dim).to(device),
            model_forecast_std*torch.ones(self.latent_dim).to(device)
        )
        self.par_error_distribution = torch.distributions.Normal(
            torch.zeros(self.pars_dim).to(device),
            par_std*torch.ones(self.pars_dim).to(device)
        )

        if self.weight_update == 'bootstrap':
            self.weight_function = self.log_likelihood

        self.weights = self.num_particles/self.num_particles * torch.ones(self.num_particles)
        self.weights = self.weights.to(self.device)

        self.ESS = 1 / self.weights.sum()
        self.ESS_threshold = self.num_particles / 2


    def generate_initial_particles(self, init_means, init_stds):
        particles = np.zeros((self.num_particles, len(self.x_vec)))
        for i, (mu, std) in enumerate(zip(init_means, init_stds)):
            coef_vec = np.array([mu, std])
            particles[i] = init_condition_gaussian(coef_vec, self.x_vec)

        particles = torch.tensor(
                particles,
                dtype=torch.get_default_dtype(),
                device=self.device
        ).unsqueeze(1)

        return particles

    def pars_log_prob(self, state, pars, obs, num_steps):
        state_pred, _ = self.forecast_model(
                x=state,
                pars=pars.unsqueeze(0),
                num_steps=num_steps
        )
        log_prob = self.log_likelihood(state_pred[0], obs).sum()

        log_prob += self.par_error_distribution.log_prob(pars).sum()

        return log_prob

    def update_pars(self, state, pars, obs, num_steps):
        """
        pars: (num_particles, par_dim)
        """

        #V = cov(pars)
        #par_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        #        loc=pars,
        #        covariance_matrix=self.smoothing_factor**2*V
        #)
        #pars = par_distribution.sample(sample_shape=(1,))[0]

        #pars = pars + self.par_error_distribution.sample((pars.shape[0],))

        pars_new = pars.mean(dim=0).unsqueeze(0)
        pars_new.requires_grad = True
        optimizer = optim.Adam([pars_new], lr=.1)
        for i in range(250):
            optimizer.zero_grad()
            state_pred, _ = self.forecast_model(
                    x=state.mean(dim=0).unsqueeze(0),
                    pars=pars_new,
                    num_steps=num_steps
            )
            log_like = -self.log_likelihood(state_pred[:, -1], obs).sum()
            log_like.backward()
            optimizer.step()
        return pars_new.detach().repeat(self.num_particles, 1)



    def generate_particles(self, state, pars, num_steps, batch_size=512):
        """
        state: (batch_size, state_dim)
        """
        self.t_idx += 0.1
        #pars = torch.sqrt(1-self.smoothing_factor**2)*pars \
        #    + (1-torch.sqrt(1-self.smoothing_factor**2))*torch.mean(pars, dim=0)
        pars = pars + self.par_error_distribution.sample((pars.shape[0],))

        particles = torch.zeros(
                self.num_particles,
                num_steps,
                self.latent_dim,
                device=self.device
        )
        for i in range(0, self.num_particles, batch_size):
            particles[i:i+batch_size], _ = self.forecast_model(
                    x=state[i:i+batch_size],
                    pars=pars[i:i+batch_size],
                    num_steps=num_steps
            )
        forecast_error = self.model_forecast_error_distribution.sample(
                sample_shape=(particles.shape[0:2])
        )

        particles = particles + forecast_error



        return particles.detach(), pars.detach()

    def compute_log_likelihood(self, preds, obs, batch_size=512):
        """
        preds: (num_particles, state_dim)
        obs: (obs_dim)
        """
        log_likelihood = torch.zeros(preds.shape[0], device=self.device)
        for i in range(0, preds.shape[0], batch_size):
            log_likelihood[i:i+batch_size] = self.log_likelihood(preds[i:i+batch_size], obs)

        return log_likelihood.detach()

    def update_weights(self, particles, obs, lol=False):
        """
        particles: (num_particles, state_dim)
        obs: (obs_dim)
        """

        #weight_multiplier = self.weight_function(particles, obs)
        weight_multiplier = self.compute_log_likelihood(particles, obs)
        weight_multiplier = torch.exp(weight_multiplier)

        if lol:
            self.weights = weight_multiplier
        else:
            self.weights = self.weights * weight_multiplier
        self.weights = self.weights / self.weights.sum()

        self.ESS = 1 / torch.pow(self.weights, 2).sum()

    def get_resample_particle_ids(self):
        """
        """

        indices = torch.multinomial(
                input=self.weights,
                num_samples=self.num_particles,
                replacement=True
        )

        self.weights = 1/self.num_particles * torch.ones(self.num_particles)
        self.weights = self.weights.to(self.device)

        #self.weights = self.weights[indices]

        return indices

        #self.weights = 1/self.num_particles * torch.ones(self.num_particles)



























