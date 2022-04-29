import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import imageio


class TrainConditionalGAN():
    def __init__(self, generator, critic,
                 generator_optimizer, critic_optimizer,
                 n_critic=5, gamma=10, save_string='TimeGAN',
                 n_epochs=100, device='cpu'):

        self.device = device
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

        self.eps = 1e-15

        self.n_epochs = n_epochs
        self.save_string = save_string

        self.generator.train(mode=True)
        self.critic.train(mode=True)

        self.latent_dim = self.generator.latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

        self.fixed_z = torch.randn(64, self.latent_dim).to(self.device)

    def train(self, dataloader):
        """Train generator and critic"""

        generator_loss_list = []
        critic_loss_list = []
        images = []

        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch

            # Train one step
            generator_loss, critic_loss, gp, \
                = self.train_epoch(dataloader)

            #print(f'Epoch: {epoch}, generator_loss: {generator_loss:.5f}, ', end=' ')
            #print(f'critic_loss: {critic_loss:.5f}, ', end=' ')
            #print(f'gp_loss: {gp:.5f}, ', end=' ')

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

            self.fixed_conditions = dataloader.dataset[np.random.randint(0, len(dataloader.dataset))][0]
            ids = np.linspace(0, len(self.fixed_conditions) - 1, num=64,
                              dtype=int)
            self.fixed_conditions = self.fixed_conditions[ids].unsqueeze(1)
            self.fixed_conditions = self.fixed_conditions.to(self.device)
            # Save generated images
            generated_img = self.generator(self.fixed_z, self.fixed_conditions)
            generated_img = generated_img.to('cpu').detach()

            generated_img = make_grid(generated_img[:, 0:1])
            images.append(generated_img)
            self.save_generator_image(generated_img,
                                      f"outputs_GAN/gen_img{epoch}.png")

        # save the generated images as GIF file
        imgs = [np.array(self.to_pil_image(img)) for img in images]
        imageio.mimsave('outputs_GAN/generator_images.gif', imgs)

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

        progress_bar = tqdm(enumerate(dataloader),
                 total=int(len(dataloader.dataset)/dataloader.batch_size))
        for bidx, (conditions, real_data) in progress_bar:

            real_data = real_data.view(
                    real_data.shape[0]*real_data.shape[1], 1,
                    real_data.shape[2],
                    real_data.shape[3],
            ).to(self.device)
            conditions = conditions.view(
                    conditions.shape[0]*conditions.shape[1], 1,
                    conditions.shape[2],
                    conditions.shape[3],
            ).to(self.device)
            batch_size = real_data.size(0)

            critic_loss, gp = self.critic_train_step(real_data,
                                                     conditions,
                                                     batch_size)

            # Train generator
            if bidx % self.n_critic == 0:
                generator_loss = self.generator_train_step(real_data, conditions, batch_size)


            progress_bar.set_postfix({"Gen":generator_loss,
                                      'Critic': critic_loss,
                                      'GP': gp,
                                      'epoch': self.epoch})
        return generator_loss, critic_loss, gp

    def critic_train_step(self, real_data, conditions, batch_size):
        """Train critic one step"""

        self.generator.eval()
        self.critic_optimizer.zero_grad()

        generated_data = self.sample(batch_size, conditions)

        grad_penalty = self.gradient_penalty(real_data, generated_data, conditions)

        cri_loss = self.critic(generated_data, conditions).mean() \
                 - self.critic(real_data, conditions).mean() + grad_penalty

        cri_loss.backward()

        self.critic_optimizer.step()

        self.generator.train(mode=True)

        return cri_loss.detach().item(),  grad_penalty.detach().item()

    def generator_train_step(self, real_data, conditions, batch_size):
        self.critic.eval()
        self.generator_optimizer.zero_grad()

        generated_data = self.sample(batch_size, conditions)

        #generator_loss = -self.critic(generated_data, conditions).mean()
        generator_loss = nn.MSELoss()(generated_data, real_data)
        generator_loss.backward()
        self.generator_optimizer.step()

        '''
        a = list(self.generator.parameters())[0].clone()
        self.generator_optimizer.step()
        b = list(self.generator.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
        '''

        self.critic.train(mode=True)

        return generator_loss.detach().item()

    def gradient_penalty(self, data=None, generated_data=None, conditions=None):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, data.size(1), 1, 1, device=self.device)
        epsilon = epsilon.expand_as(data)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data
        interpolation = torch.autograd.Variable(interpolation,
                                                requires_grad=True)

        conditions_in = torch.autograd.Variable(conditions, requires_grad=True)

        interpolation_critic_score = self.critic(interpolation, conditions_in)

        grad_outputs = torch.ones(interpolation_critic_score.size(),
                                  device=self.device)

        gradients = torch.autograd.grad(outputs=interpolation_critic_score,
                                        inputs=[interpolation,conditions_in],
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]
        gradients_norm = torch.sqrt(
            torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()


    def sample(self, n_samples, conditions):
        """Generate n_samples fake samples"""
        z = torch.randn(n_samples, self.latent_dim, requires_grad=True).to(self.device)
        return self.generator(z, conditions)

    def save_generator_image(self, image, path):
        """Save image"""
        save_image(image, path)