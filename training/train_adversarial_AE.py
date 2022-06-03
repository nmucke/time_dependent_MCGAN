import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


class TrainAdversarialAE():
    def __init__(self, encoder, decoder, critic, encoder_optimizer,
                 decoder_optimizer, critic_optimizer, encoder_reg_optimizer,
                 latent_dim=32, n_critic=5, gamma=10, save_string='AdvAE',
                 n_epochs=100, device='cpu'):

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.enc_opt = encoder_optimizer
        self.enc_reg_opt = encoder_reg_optimizer
        self.dec_opt = decoder_optimizer
        self.cri_opt = critic_optimizer

        self.eps = 1e-15

        self.n_epochs = n_epochs
        self.save_string = save_string

        self.encoder.train(mode=True)
        self.decoder.train(mode=True)
        self.critic.train(mode=True)


        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

        self.reconstruction_loss_function = nn.MSELoss()

        self.critic_loss_function = nn.BCELoss()

    def train(self, dataloader):
        """Train generator and critic"""

        recon_loss_list = []
        critic_loss_list = []
        enc_loss_list = []
        for epoch in range(1, self.n_epochs + 1):

            # Train one step
            recon_loss, critic_loss, enc_loss, gp = self.train_epoch(dataloader)

            print(f'Epoch: {epoch}, recon_loss: {recon_loss:.5f}, ', end=' ')
            print(f'critic_loss: {critic_loss:.5f}, ', end=' ')
            print(f'enc_loss: {enc_loss:.5f}, ', end=' ')
            print(f'gp_loss: {gp:.5f}, ')


            # Save loss
            recon_loss_list.append(recon_loss)
            critic_loss_list.append(critic_loss)
            enc_loss_list.append(enc_loss)

            # Save generator and critic weights
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
                'encoder_reg_optimizer_state_dict': self.enc_reg_opt.state_dict(),
                'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
                'critic_optimizer_state_dict': self.cri_opt.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
            'encoder_reg_optimizer_state_dict': self.enc_reg_opt.state_dict(),
            'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
            'critic_optimizer_state_dict': self.cri_opt.state_dict(),
            }, self.save_string)

        self.encoder.train(mode=False)
        self.decoder.train(mode=False)
        self.critic.train(mode=False)

        return recon_loss_list, critic_loss_list, enc_loss_list

    def train_epoch(self, dataloader):
        """Train generator and critic for one epoch"""

        #for bidx, (real_data, real_pars) in tqdm(enumerate(dataloader),
        #         total=int(len(dataloader.dataset)/dataloader.batch_size)):
        for bidx, real_data in tqdm(enumerate(dataloader),
                 total=int(len(dataloader.dataset)/dataloader.batch_size)):

            real_data_shape = real_data.shape
            #real_data = real_data.reshape(-1, 1, real_data_shape[2])
            real_data = real_data.reshape(-1, real_data_shape[2])
            shuffle_ids = torch.randperm(real_data.shape[0])
            real_data = real_data[shuffle_ids]
            real_data = real_data.to(self.device)

            #pdb.set_trace()
            #real_pars = real_pars[shuffle_ids]
            #real_pars = real_pars.to(self.device)

            self.encoder.eval()
            critic_loss, gp = self.critic_train_step(real_data)
            self.encoder.train()

            if bidx % self.n_critic == 0:
                recon_loss = self.reconstruction_train_step(real_data)
                #enc_loss = self.regularization_train_step(real_data)


        return recon_loss, critic_loss, 1,1#enc_loss, gp

    def critic_train_step(self, data):
        """Train critic one step"""

        batch_size = data.size(0)
        self.encoder.eval()

        self.cri_opt.zero_grad()

        generated_latent_data = self.encoder(data)
        true_latent_data = self.sample(batch_size)

        critic_real = self.critic(true_latent_data)
        critic_generated = self.critic(generated_latent_data)

        target_real = torch.ones_like(critic_real)
        target_generated = torch.zeros_like(critic_generated)

        #grad_penalty = self.gradient_penalty(true_latent_data, generated_latent_data)
        #cri_loss = self.critic(generated_latent_data).mean() \
        #         - self.critic(true_latent_data).mean() + grad_penalty

        #cri_loss = -torch.mean(torch.log(critic_real+self.eps) \
        #                       + torch.log(1 - critic_generated+self.eps))

        cri_loss = 0.5 * self.critic_loss_function(critic_real, target_real) \
                   + 0.5 * self.critic_loss_function(critic_generated, target_generated)

        cri_loss.backward()
        self.cri_opt.step()

        self.encoder.train()

        '''
        a = list(self.critic.parameters())[0].clone()
        self.cri_opt.step()
        b = list(self.critic.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
        pdb.set_trace()
        '''

        return cri_loss.detach().item(),  1#grad_penalty.detach().item()

    def regularization_train_step(self, data):
        self.enc_reg_opt.zero_grad()

        generated_latent_data = self.encoder(data)
        #critic_generated = self.critic(generated_latent_data)
        #enc_loss = -torch.mean(torch.log(critic_generated+self.eps))

        enc_loss = -self.critic(generated_latent_data).mean()
        enc_loss.backward()
        self.enc_reg_opt.step()

        return enc_loss.detach().item()

    def reconstruction_train_step(self, real_data):
        """Train generator one step"""

        self.critic.eval()

        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()

        latent_data = self.encoder(real_data)
        reconstruction = self.decoder(latent_data)

        critic_generated = self.critic(latent_data)
        target_generated = torch.ones_like(critic_generated)

        reconstruction_loss = 0.999 * self.reconstruction_loss_function(reconstruction,
                                                                real_data)
        reconstruction_loss += 0.001 * self.critic_loss_function(critic_generated, target_generated)

        reconstruction_loss.backward()
        self.enc_opt.step()
        self.dec_opt.step()

        self.critic.train()

        '''
        a = list(self.decoder.parameters())[0].clone()
        #print(a)
        self.enc_opt.step()
        self.dec_opt.step()
        b = list(self.decoder.parameters())[0].clone()
        #print(b)
        print(torch.equal(a.data, b.data))
        #pdb.set_trace()
        '''

        return reconstruction_loss.detach().item()

    def gradient_penalty(self, data, generated_data):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device)
        epsilon = epsilon.expand_as(data)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data
        interpolation = torch.autograd.Variable(interpolation,
                                                requires_grad=True)

        interpolation_critic_score = self.critic(interpolation)

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

    def sample(self, n_samples):
        """Generate n_samples fake samples"""
        return torch.randn(n_samples, self.latent_dim).to(self.device)
