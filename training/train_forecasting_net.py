import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import imageio

class TrainForecastingNet():
    def __init__(self,
                 model,
                 model_optimizer,
                 scheduler=None,
                 save_string='forecasting_net',
                 n_epochs=100,
                 device='cpu'):

        self.device = device
        self.model = model
        self.optimizer = model_optimizer

        self.scheduler = scheduler


        self.n_epochs = n_epochs
        self.save_string = 'model_weights/' + save_string

        self.model.train(mode=True)


        self.loss_function = nn.MSELoss()
        #self.loss_function = nn.L1Loss()


    def train(self, dataloader, temporal_batch_size=32):
        """Train generator and critic"""

        loss_list = []
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch

            # Train one step
            loss = self.train_epoch(dataloader=dataloader,
                                    temporal_batch_size=temporal_batch_size)

            if self.scheduler is not None:
                self.scheduler.step()

            # Save loss
            loss_list.append(loss)

            # Save generator and critic weights
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        self.model.eval()

        return loss_list

    def train_epoch(self, dataloader, temporal_batch_size):
        """Train generator and critic for one epoch"""

        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader.dataset) // dataloader.batch_size,
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for bidx, (features, targets) in progress_bar:
            loss = 0
            shuffled_ids = np.arange(0,features.shape[1])
            np.random.shuffle(shuffled_ids)
            features[:, shuffled_ids] = features[:, shuffled_ids]
            targets[:, shuffled_ids] = targets[:, shuffled_ids]

            features = features.view(-1, 1, features.shape[2], features.shape[3])
            targets = targets.view(-1, 1, targets.shape[2], targets.shape[3])

            features = features.to(self.device)
            targets = targets.to(self.device)
            for batch, i in enumerate(range(0, features.shape[0]-1, temporal_batch_size)):
                batch_features = features[i:i+temporal_batch_size]
                batch_targets = targets[i:i+temporal_batch_size]
                loss += self.train_step(batch_features, batch_targets)

            progress_bar.set_postfix({"Loss": loss/batch,
                                      'Epoch': self.epoch,
                                      'of': self.n_epochs})
            total_loss += loss/batch
        return total_loss

    def train_step(self, features, targets):
        self.optimizer.zero_grad()
        output = self.model(features)
        loss = self.loss_function(output, targets)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
        self.optimizer.step()

        return loss.detach().item()



class TrainForecastingGAN():
    def __init__(self,
                 generator,
                 critic,
                 generator_optimizer,
                 critic_optimizer,
                 scheduler=None,
                 save_string='forecasting_net',
                 n_epochs=100,
                 device='cpu'):

        self.device = device
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

        self.scheduler = scheduler

        self.latent_dim = generator.z_latent_dim

        self.n_critic = 3
        self.gamma = 30

        self.n_epochs = n_epochs
        self.save_string = 'model_weights/' + save_string

        self.generator.train(mode=True)
        self.critic.train(mode=True)


    def train(self, dataloader, temporal_batch_size=32):
        """Train generator and critic"""

        loss_list = []
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch

            # Train one step
            cri_loss, grad_penalty, gen_loss = self.train_epoch(dataloader=dataloader,
                                    temporal_batch_size=temporal_batch_size)

            if self.scheduler is not None:
                self.scheduler.step()

            # Save loss
            loss_list.append(cri_loss)

            # Save generator and critic weights
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        self.generator.eval()
        self.critic.eval()

        return loss_list

    def train_epoch(self, dataloader, temporal_batch_size):
        """Train generator and critic for one epoch"""

        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader.dataset) // dataloader.batch_size,
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for bidx, (conditions, targets) in progress_bar:
            loss = 0
            shuffled_ids = np.arange(0,conditions.shape[1])
            np.random.shuffle(shuffled_ids)
            conditions[:, shuffled_ids] = conditions[:, shuffled_ids]
            targets[:, shuffled_ids] = targets[:, shuffled_ids]

            conditions = conditions.view(-1, 1, conditions.shape[2], conditions.shape[3])
            targets = targets.view(-1, 1, targets.shape[2], targets.shape[3])

            real_data = torch.cat((conditions, targets), dim=-1)
            real_data = real_data[:, :, :, -32:]

            conditions = conditions.to(self.device)
            real_data = real_data.to(self.device)

            cri_loss = 0
            grad_penalty = 0
            gen_loss = 0
            for batch, i in enumerate(range(0, conditions.shape[0]-1, temporal_batch_size)):
                batch_conditions = conditions[i:i+temporal_batch_size]
                batch_real_data = real_data[i:i+temporal_batch_size]

                cri_loss_, grad_penalty_ = self.critic_train_step(
                        batch_real_data,
                        batch_conditions,
                        batch_conditions.shape[0]
                )
                cri_loss += cri_loss_
                grad_penalty += grad_penalty_

                if batch % self.n_critic ==0:
                    gen_loss_ = self.generator_train_step(
                            batch_conditions,
                            batch_conditions.shape[0],
                            real_data=batch_real_data
                    )
                    gen_loss += gen_loss_

            progress_bar.set_postfix({"cri_loss": cri_loss/batch,
                                      "grad_penalty": grad_penalty/batch,
                                      "gen_loss": gen_loss/batch,
                                      'Epoch': self.epoch,
                                      'of': self.n_epochs})

        return cri_loss, grad_penalty, gen_loss

    def critic_train_step(self, real_data, conditions, batch_size):
        """Train critic one step"""

        self.generator.eval()
        self.critic_optimizer.zero_grad()

        generated_data = self.sample(batch_size, conditions)

        grad_penalty = self.gradient_penalty(real_data, generated_data[:, :, :, -32:])

        cri_loss = self.critic(generated_data[:, :, :, -32:]).mean() \
                   - self.critic(real_data).mean()# + grad_penalty

        cri_loss.backward()

        self.critic_optimizer.step()

        self.generator.train(mode=True)

        return cri_loss.detach().item(), grad_penalty.detach().item()

    def generator_train_step(self, conditions, batch_size, real_data):
        self.critic.eval()
        self.generator_optimizer.zero_grad()

        generated_data = self.sample(batch_size, conditions)

        generator_loss = -self.critic(generated_data[:, :, :, -32:]).mean()
        #generator_loss = nn.L1Loss()(generated_data[:, :, :, -32:], real_data)
        generator_loss.backward()
        #self.generator_optimizer.step()

        '''
        a = list(self.generator.parameters())[-1].clone()
        self.generator_optimizer.step()
        b = list(self.generator.parameters())[-1].clone()
        print(torch.equal(a.data, b.data))
        '''


        self.critic.train(mode=True)

        return generator_loss.detach().item()

    def gradient_penalty(self, data=None, generated_data=None):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, data.size(1), 1, 1,
                             device=self.device)
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

    def sample(self, n_samples, conditions):
        """Generate n_samples fake samples"""
        z = torch.randn(n_samples, self.latent_dim, requires_grad=True).to(
            self.device)
        return self.generator(z, conditions, return_input=True)

    def save_generator_image(self, image, path):
        """Save image"""
        save_image(image, path)