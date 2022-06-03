import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.adv_diff_dataloader import get_dataloader
import models.adv_diff_models.conditional_GAN as models
from utils.seed_everything import seed_everything
from training.train_conditional_gan import TrainConditionalGAN
from data_handling.conditional_gan_adv_diff_dataloader import AdvDiffDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from tqdm import tqdm


torch.set_default_dtype(torch.float32)

if __name__ == '__main__':

    seed_everything()

    continue_training = False
    train = True

    if not train:
        continue_training = True
        cuda = False
    else:
        cuda = True

    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    data_path = 'data/advection_diffusion/train_data/adv_diff'

    dataset_params = {
        'data_path': data_path,
        'num_files': 1000,
        'num_states_pr_sample': 128,
        'sample_size': (128, 512),
        'window_size': 32,
        'transformer_state': None,
        'transformer_pars': None,
    }
    dataloader_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 8,
        'drop_last': True,
    }

    dataset = AdvDiffDataset(**dataset_params)
    dataloader = DataLoader(dataset, **dataloader_params)

    latent_dim = 150
    input_dim = 128

    generator_params = {
        'out_channels': 1,
        'latent_dim': latent_dim,
        'hidden_channels': [4, 8, 16, 32],
        'conditional_hidden_neurons': [4, 8, 16, 32],
    }

    critic_params = {
        'in_channels': 1,
        'hidden_channels': [32, 16, 8, 4],
    }


    #generator = models.ConditionalGenerator(**generator_params).to(device)
    generator = models.UNetGenerator(latent_dim=latent_dim, hidden_channels=2, in_channels=1, bilinear=False).to(device)
    critic = models.UNetCritic(in_channels=1, hidden_channels=2).to(device)

    z = torch.randn(1, latent_dim).to(device)
    c = dataloader.dataset[0][0]
    x = dataloader.dataset[0][1]
    lol = generator(z, c[0:1].unsqueeze(1).to(device))
    lol = critic(lol, c[0:1].unsqueeze(1).to(device))
    '''
    plt.figure()
    plt.imshow(torch.cat([c,x],dim=-1).detach().cpu().numpy()[0, :, :])
    plt.show()
    '''


    learning_rate = 1e-4

    generator_optimizer = torch.optim.RMSprop(
            generator.parameters(),
            lr=learning_rate
    )

    critic_optimizer = torch.optim.RMSprop(
            critic.parameters(),
            lr=learning_rate
    )

    if train:
        training_params = {
            'n_critic': 3,
            'gamma': 1,
            'n_epochs': 1000,
            'save_string': 'model_weights/ConditionalGAN',
            'device': device
        }
        trainer = TrainConditionalGAN(
                generator=generator,
                critic=critic,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                **training_params
        )

        trainer.train(dataloader=dataloader)

    if not train:
        checkpoint = torch.load('model_weights/ConditionalGAN')
        generator.load_state_dict(checkpoint['generator_state_dict'])

        c = dataloader.dataset[0][0]
        x_true = dataloader.dataset[0][1].unsqueeze(1)

        z = torch.randn(1, latent_dim, requires_grad=True).to(device)
        z_optimizer = torch.optim.Adam([z], lr=1e-0)

        x_pred_list = []
        x = c.unsqueeze(1).to(device)
        for i in range(len(x_true)):
            pbar = tqdm(range(100))
            '''
            for j in pbar:
                z_optimizer.zero_grad()
                pred = generator(z, c[i:i+1].unsqueeze(1).to(device))
                loss = torch.mean((pred - x_true[i:i+1])**2)
                loss.backward(retain_graph=True)
                z_optimizer.step()

                pbar.set_postfix({"loss": loss.item(),
                                  "step": i,
                                  "of": len(x_true)})
            '''

            x = generator(z, c[i:i+1].unsqueeze(1).to(device))
            #pred = pred.mean(dim=0).unsqueeze(0)
            x_pred_list.append(x.detach())
        x_pred_list = torch.stack(x_pred_list)
        x_pred_list = x_pred_list.squeeze(1)
        x_pred_list = x_pred_list.squeeze(1)
        x_pred_list = x_pred_list.cpu().detach().numpy()

        x_true = x_true.squeeze(1).cpu().detach().numpy()

        plt.figure()
        plt.plot(x_true[0, :, 10])
        plt.plot(x_pred_list[0, :, 10])
        plt.show()

        plt.figure()
        for i in [1, 10, 30, 40, 50, 60]:
            plt.plot(x_true[i, :, -1], label='True', color='tab:blue')
            plt.plot(x_pred_list[i, :, -1], label='Pred', color='tab:orange')
        plt.show()






