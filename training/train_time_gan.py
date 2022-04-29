import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


class TrainTimeGAN():
    def __init__(self, generator, critic,
                 generator_optimizer, critic_optimizer,
                 par_generator, par_critic,
                 par_generator_optimizer, par_critic_optimizer,
                 n_critic=5, gamma=10, save_string='TimeGAN',
                 n_epochs=100, device='cpu'):

        self.device = device
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

        self.par_generator = par_generator
        self.par_critic = par_critic
        self.par_generator_optimizer = par_generator_optimizer
        self.par_critic_optimizer = par_critic_optimizer

        self.eps = 1e-15

        self.n_epochs = n_epochs
        self.save_string = save_string

        self.generator.train(mode=True)
        self.critic.train(mode=True)
        self.par_generator.train(mode=True)
        self.par_critic.train(mode=True)

        self.latent_dim = self.generator.latent_dim
        self.par_latent_dim = self.generator.par_latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

    def train(self, dataloader, pretrain_dataloader=None):
        """Train generator and critic"""

        generator_loss_list = []
        critic_loss_list = []

        if pretrain_dataloader is not None:
            print('Pretraining starting')
            progress_bar = tqdm(range(0, 1000))
            for epoch in progress_bar:
                for bidx, real_pars in enumerate(pretrain_dataloader):

                    #real_pars = real_pars[0].to(self.device)
                    real_pars = torch.randn(248, 3).to(self.device)

                    #for _ in range(3):
                    par_critic_loss, par_gp = self.par_critic_train_step(real_pars)

                    # Train generator
                    if bidx % self.n_critic == 0:
                        par_generator_loss = self.par_generator_train_step(real_pars)

                progress_bar.set_postfix({"generator": par_generator_loss,
                                          "critic": par_critic_loss,
                                          "gp": par_gp})

            lol = self.sample_pars(10000)
            plt.figure()
            plt.hist(lol.detach().cpu().numpy(), bins=50)
            #plt.hist(real_pars.detach().cpu().numpy(), bins=50)
            plt.show()
            pdb.set_trace()

            print(f'Pretraining complete, ', end='')
            print(f'par_generator_loss: {par_generator_loss:0.4f}', end=' ')
            print(f'par_critic_loss: {par_critic_loss:0.4f}', end=' ')
            print(f'par_gp: {par_gp:0.4f}')
        for epoch in range(1, self.n_epochs + 1):

            # Train one step
            generator_loss, critic_loss, gp, \
            par_generator_loss, par_critic_loss, par_gp \
                = self.train_epoch(dataloader)

            print(f'Epoch: {epoch}, generator_loss: {generator_loss:.5f}, ', end=' ')
            print(f'critic_loss: {critic_loss:.5f}, ', end=' ')
            print(f'gp_loss: {gp:.5f}, ', end=' ')

            print(f'par_generator_loss: {par_generator_loss:.5f}, ', end=' ')
            print(f'par_critic_loss: {par_critic_loss:.5f}, ', end=' ')
            print(f'par_gp_loss: {par_gp:.5f}, ')

            # Save loss
            generator_loss_list.append(generator_loss)
            critic_loss_list.append(critic_loss)

            # Save generator and critic weights
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, self.save_string)

        self.generator.train(mode=False)
        self.critic.train(mode=False)

        return generator_loss_list, critic_loss_list

    def train_epoch(self, dataloader):
        """Train generator and critic for one epoch"""

        for bidx, (real_data, real_pars) in tqdm(enumerate(dataloader),
                 total=int(len(dataloader.dataset)/dataloader.batch_size)):

            real_data = real_data.to(self.device)
            real_pars = real_pars.to(self.device)
            batch_size = real_data.size(0)

            # Train critic
            #for _ in range(self.n_critic):

            #par_critic_loss, par_gp = self.par_critic_train_step(real_pars)

            critic_loss, gp = self.critic_train_step(real_data,
                                                     real_pars,
                                                     batch_size)

            # Train generator
            if bidx % self.n_critic == 0:
                #par_generator_loss = self.par_generator_train_step(real_pars)
                generator_loss = self.generator_train_step(batch_size)

        return generator_loss, critic_loss, gp, 1,1,1#\
               #par_generator_loss, par_critic_loss, par_gp

    def par_critic_train_step(self, real_pars):
        self.par_generator.eval()
        self.par_critic_optimizer.zero_grad()
        generated_pars = self.sample_pars(real_pars.size(0))
        par_grad_penalty = self.gradient_penalty(real_pars=real_pars,
                                             generated_pars=generated_pars,
                                             par_critic=True)
        par_cri_loss = self.par_critic(generated_pars).mean() \
                        - self.par_critic(real_pars).mean() + par_grad_penalty

        par_cri_loss.backward()
        self.par_critic_optimizer.step()
        self.par_generator.train()

        return par_cri_loss.detach().item(), par_grad_penalty.detach().item()


    def critic_train_step(self, real_data, real_pars, batch_size):
        """Train critic one step"""

        self.generator.eval()
        self.critic_optimizer.zero_grad()

        generated_data, generated_pars = self.sample(batch_size, output_pars=True)

        grad_penalty = self.gradient_penalty(real_data, generated_data, real_pars)
        cri_loss = self.critic(generated_data, generated_pars).mean() \
                 - self.critic(real_data, real_pars).mean() + grad_penalty

        cri_loss.backward()
        self.critic_optimizer.step()

        self.generator.train(mode=True)

        return cri_loss.detach().item(),  grad_penalty.detach().item()


    def par_generator_train_step(self, real_pars):
        self.par_critic.eval()
        self.par_generator_optimizer.zero_grad()
        generated_pars = self.sample_pars(real_pars.size(0))
        par_gen_loss = -self.par_critic(generated_pars).mean()

        par_gen_loss.backward()
        self.par_generator_optimizer.step()
        self.par_critic.train()

        return par_gen_loss.detach().item()

    def generator_train_step(self, batch_size):
        self.critic.eval()
        self.generator.par_generator.eval()
        self.generator_optimizer.zero_grad()

        generated_data, generated_pars = self.sample(batch_size, output_pars=True)

        generator_loss = -self.critic(generated_data, generated_pars).mean()
        generator_loss.backward()
        self.generator_optimizer.step()

        self.generator.par_generator.train()
        self.critic.train(mode=True)

        return generator_loss.detach().item()

    def gradient_penalty(self, data=None, generated_data=None,
                         real_pars=None, generated_pars=None, par_critic=False):
        """Compute gradient penalty"""

        if par_critic:
            batch_size = real_pars.size(0)
            epsilon = torch.rand(batch_size, real_pars.size(1), device=self.device)
            epsilon = epsilon.expand_as(real_pars)

            interpolation = epsilon * real_pars.data + (1 - epsilon) * generated_pars
            interpolation = torch.autograd.Variable(interpolation,
                                                    requires_grad=True)

            interpolation_critic_score = self.par_critic(interpolation)

        else:
            batch_size = data.size(0)
            epsilon = torch.rand(batch_size, data.size(1), 1, device=self.device)
            epsilon = epsilon.expand_as(data)

            interpolation = epsilon * data.data + (1 - epsilon) * generated_data
            interpolation = torch.autograd.Variable(interpolation,
                                                    requires_grad=True)
            interpolation_critic_score = self.critic(interpolation, real_pars)

        grad_outputs = torch.ones(interpolation_critic_score.size(),
                                  device=self.device)

        gradients = torch.autograd.grad(outputs=interpolation_critic_score,
                                        inputs=interpolation,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]
        gradients_norm = torch.sqrt(
            torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def sample_pars(self, n_samples):
        z = torch.randn(n_samples, self.par_latent_dim).to(self.device)
        return self.par_generator(z)

    def sample(self, n_samples, output_pars=False):
        """Generate n_samples fake samples"""
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        return self.generator(z, output_pars=output_pars)
