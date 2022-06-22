import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


class TrainAdversarialAE():
    def __init__(self, encoder, decoder, critic, koopman,
                 encoder_optimizer, decoder_optimizer,
                 critic_optimizer,
                 koopman_optimizer,
                 with_koopman_training,
                 with_adversarial_training,
                 latent_dim=32, n_critic=5, gamma=10, save_string='AdvAE',
                 n_epochs=100, device='cpu'):

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.koopman = koopman
        self.enc_opt = encoder_optimizer
        self.dec_opt = decoder_optimizer
        self.cri_opt = critic_optimizer
        self.koopman_opt = koopman_optimizer
        scheduler_step_size = 5
        scheduler_gamma = 0.95

        self.enc_opt_scheduler = optim.lr_scheduler.StepLR(
                self.enc_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )
        self.dec_opt_scheduler = optim.lr_scheduler.StepLR(
                self.dec_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )
        self.cri_opt_scheduler = optim.lr_scheduler.StepLR(
                self.cri_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )
        self.koopman_opt_scheduler = optim.lr_scheduler.StepLR(
                self.koopman_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )

        self.with_koopman_training = with_koopman_training
        self.with_adversarial_training = with_adversarial_training
        self.n_epochs = n_epochs
        self.save_string = save_string

        self.encoder.train()
        self.decoder.train()
        self.critic.train()
        self.koopman.train()

        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

        self.reconstruction_loss_function = nn.MSELoss()
        self.koopman_loss_function = nn.MSELoss()
        self.critic_loss_function = nn.BCELoss()

    def train(self, dataloader):
        """Train generator and critic"""

        #self.pre_train_AE(dataloader, num_epochs=10)
        #self.encoder.eval()
        #self.decoder.eval()
        #self.pre_train_koopman(dataloader, num_epochs=1)
        #self.encoder.train()
        #self.decoder.train()

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'koopman_state_dict': self.koopman.state_dict(),
            'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
            'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
            'critic_optimizer_state_dict': self.cri_opt.state_dict(),
            'koopman_optimizer_state_dict': self.koopman_opt.state_dict(),
            }, self.save_string)

        recon_loss_list = []
        critic_loss_list = []
        enc_loss_list = []
        self.teacher_forcing_rate = 1.
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch

            # Train one step
            recon_loss, critic_loss, enc_loss, gp = self.train_epoch(dataloader)
            self.teacher_forcing_rate = self.teacher_forcing_rate * 0.98

            # Save loss
            recon_loss_list.append(recon_loss)
            critic_loss_list.append(critic_loss)
            enc_loss_list.append(enc_loss)

            # Save generator and critic weights
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'koopman_state_dict': self.koopman.state_dict(),
                'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
                'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
                'critic_optimizer_state_dict': self.cri_opt.state_dict(),
                'koopman_optimizer_state_dict': self.koopman_opt.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'koopman_state_dict': self.koopman.state_dict(),
            'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
            'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
            'critic_optimizer_state_dict': self.cri_opt.state_dict(),
            'koopman_optimizer_state_dict': self.koopman_opt.state_dict(),
            }, self.save_string)

        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()
        self.koopman.eval()

        return recon_loss_list, critic_loss_list, enc_loss_list

    def train_epoch(self, dataloader):
        """Train generator and critic for one epoch"""

        pbar = tqdm(
                enumerate(dataloader),
                total=int(len(dataloader.dataset)/dataloader.batch_size)
        )
        recon_loss = 0

        for bidx, (real_state, real_pars) in pbar:

            self.batch_size, self.num_steps, self.num_channels, self.num_x = real_state.shape

            self.time_vec = torch.linspace(0, 50, self.num_steps).float().to(self.device)
            self.time_vec = self.time_vec.repeat(self.batch_size, 1)
            self.time_vec = self.time_vec.reshape(self.batch_size*self.num_steps, 1)

            real_state = real_state.to(self.device)
            real_pars = real_pars.to(self.device)

            if self.with_adversarial_training:
                self.encoder.eval()
                critic_loss, gp = self.critic_train_step(
                        state=real_state.reshape(-1, self.num_channels, self.num_x),
                        pars=real_pars
                )
                self.encoder.train()
            else:
                critic_loss = 0

            #if bidx % self.n_critic == 0:
            self.critic.eval()
            recon_loss += self.train_step(
                    real_state=real_state,
                    real_pars=real_pars
            )
            self.critic.train()

            pbar.set_postfix({
                    'recon_loss': recon_loss/(bidx+1),
                    'critic_loss': critic_loss/(bidx+1),
                    'epoch': self.epoch,
                    }
            )

        self.enc_opt_scheduler.step()
        self.dec_opt_scheduler.step()
        if self.with_adversarial_training:
            self.cri_opt_scheduler.step()
        if self.with_koopman_training:
            self.koopman_opt_scheduler.step()

        return recon_loss/(bidx+1), 1, 1,1#enc_loss, gp

    def critic_train_step(self, state, pars):
        """Train critic one step"""

        batch_size = state.size(0)

        self.cri_opt.zero_grad()

        generated_latent_data = self.encoder(state)
        true_latent_data = self.sample(batch_size)

        generated_latent_data = torch.cat([generated_latent_data, self.time_vec], dim=1)
        true_latent_data = torch.cat([true_latent_data, self.time_vec], dim=1)

        critic_real = self.critic(true_latent_data)
        critic_generated = self.critic(generated_latent_data)

        target_real = torch.ones_like(critic_real)
        target_generated = torch.zeros_like(critic_generated)

        cri_loss = 0.5 * self.critic_loss_function(critic_real, target_real) \
                   + 0.5 * self.critic_loss_function(critic_generated, target_generated)

        cri_loss.backward()
        self.cri_opt.step()

        '''
        a = list(self.critic.parameters())[0].clone()
        self.cri_opt.step()
        b = list(self.critic.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
        pdb.set_trace()
        '''

        return cri_loss.detach().item(),  1#grad_penalty.detach().item()

    #def train_step(self, real_data, real_state_init, real_state_preds, real_pars):
    def train_step(self, real_state, real_pars):

        # Reshape state to be (batch_size*time_steps, n_features)
        real_state = real_state.reshape(
                self.batch_size*self.num_steps,
                self.num_channels,
                self.num_x
        )

        # Encode state
        real_latent = self.encoder(real_state)

        real_latent = torch.cat([real_latent, self.time_vec], dim=1)

        if self.with_adversarial_training:
            # Compute critic loss
            loss_critic = self.critic_loss_function(
                    self.critic(real_latent),
                    torch.ones_like(self.critic(real_latent))
            )

        # Decode state
        state_recon = self.decoder(real_latent[:, :-1])

        # Compute reconstruction loss
        loss_recon = self.reconstruction_loss_function(state_recon, real_state)
        # Reshape latent state to be (batch_size, time_steps, latent_dim)

        loss = 0
        if self.with_koopman_training:
            real_latent = real_latent.reshape(
                    self.batch_size,
                    self.num_steps,
                    self.latent_dim
            )

            # Preallocate tensor for koopman prediction
            z_preds = torch.zeros((self.batch_size,
                                   self.num_steps,
                                   self.latent_dim),
                                  device=self.device)
            # Time step in latent space using koopman
            reg_loss = 0
            z_preds[:, 0, :] = real_latent[:, 0, :]
            z_old = real_latent[:, 0, :]
            for i in range(0, self.num_steps-1):
                if self.teacher_forcing_rate > torch.rand(1):
                    z_new, koopman_mat = self.koopman(real_latent[:, i], real_pars)
                else:
                    z_new, koopman_mat = self.koopman(z_old, real_pars)
                z_preds[:, i+1, :] = z_new
                z_old = z_new

                reg_loss += torch.sum(torch.pow(koopman_mat, 2))

            # Compute koopman loss in latent space
            loss_koopman_latent = self.reconstruction_loss_function(
                    z_preds[:, 1:, :],
                    real_latent[:, 1:, :]
            )

            # Reshape latent koopman preds to be (batch_size*time_steps, latent_dim)
            z_preds = z_preds.reshape(-1, self.latent_dim)

            # Decode koopman preds
            recon_preds = self.decoder(z_preds)

            # Reshape decoded koopman preds to (batch_size, time_steps-1, n_features)
            recon_preds = recon_preds.reshape(
                    self.batch_size,
                    self.num_steps,
                    self.num_x
            )

            # Reshape real state to (batch_size, time_steps, n_features)
            real_state = real_state.reshape(
                    self.batch_size,
                    self.num_steps,
                    self.num_x
            )

            # Compute koopman loss in state space
            loss_koopman = self.reconstruction_loss_function(recon_preds[:, 1:, :], real_state[:, 1:, :])

            # Compute total loss
            loss = loss + 1e-1*loss_koopman + 1e-1*loss_koopman_latent + 1e-2*reg_loss

        if self.with_adversarial_training:
            loss = loss + 1e-3*loss_critic

        loss = loss + loss_recon

        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        if self.with_koopman_training:
            self.koopman_opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.koopman.parameters(), 0.1)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)

        self.enc_opt.step()
        self.dec_opt.step()
        if self.with_koopman_training:
            self.koopman_opt.step()

        return loss_recon.detach().item()

    def pre_train_AE(self, dataloader, num_epochs=20):

        pbar = tqdm(range(num_epochs), total=num_epochs)
        for epoch in pbar:
            for bidx, (real_state, _) in enumerate(dataloader):
                self.batch_size, self.num_steps, self.num_x = real_state.shape

                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()

                self.real_state_shape = real_state.shape
                real_state = real_state.reshape(
                        self.batch_size*self.num_steps,
                        self.num_x
                )
                real_state = real_state.to(self.device)
                z = self.encoder(real_state)
                state_recon = self.decoder(z)
                loss = self.reconstruction_loss_function(
                        state_recon,
                        real_state
                )
                loss.backward()
                self.enc_opt.step()
                self.dec_opt.step()

    def pre_train_koopman(self, dataloader, num_epochs=5):

        pbar = tqdm(range(num_epochs), total=num_epochs)
        for epoch in pbar:
            for bidx, (real_state, real_pars) in enumerate(dataloader):

                self.batch_size, self.num_steps, self.num_x = real_state.shape

                # Reshape state to be (batch_size*time_steps, n_features)
                real_state = real_state.reshape(-1, self.num_x)
                real_state = real_state.to(self.device)
                real_pars = real_pars.to(self.device)

                # Encode state
                real_latent = self.encoder(real_state)

                # Reshape latent state to be (batch_size, time_steps, latent_dim)
                real_latent = real_latent.reshape(
                        self.batch_size,
                        self.num_steps,
                        self.latent_dim
                )

                # Preallocate tensor for koopman prediction
                z_preds = torch.zeros(
                    (self.batch_size, self.num_steps - 1, self.latent_dim),
                    device=self.device
                )
                reg_loss = 0
                # Time step in latent space using koopman
                for i in range(0, self.num_steps - 1):
                    z_preds[:, i, :], koopman_mat = self.koopman(
                            real_latent[:, i],
                            real_pars
                    )

                    reg_loss += torch.sum(torch.pow(koopman_mat, 2))

                # Compute koopman loss in latent space
                loss_koopman_latent = self.reconstruction_loss_function(
                    z_preds,
                    real_latent[:, 1:, :]
                )
                # Reshape latent koopman preds to be (batch_size*time_steps, latent_dim)
                z_preds = z_preds.reshape(-1, self.latent_dim)

                # Decode koopman preds
                recon_preds = self.decoder(z_preds)

                # Reshape decoded koopman preds to (batch_size, time_steps-1, n_features)
                recon_preds = recon_preds.reshape(
                    self.batch_size,
                    self.num_steps - 1,
                    self.num_x
                )

                # Reshape real state to (batch_size, time_steps, n_features)
                real_state = real_state.reshape(
                        self.batch_size,
                        self.num_steps,
                        self.num_x
                )

                # Compute koopman loss in state space
                loss_koopman = self.reconstruction_loss_function(
                    recon_preds,
                    real_state[:, 1:, :]
                )

                # Compute total loss
                loss = 1e-2 * loss_koopman + 1e-2 * loss_koopman_latent + 1e-2 * reg_loss

                self.koopman_opt.zero_grad()
                loss.backward()
                self.koopman_opt.step()

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
