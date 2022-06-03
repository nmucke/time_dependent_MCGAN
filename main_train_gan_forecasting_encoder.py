import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pdb
from models.adv_diff_models.pre_train_time_gan import ForecastingNet, ConditionalGenerator, Critic
from scipy.integrate import solve_ivp
import data_handling.data_handling_utils as utils
from torch.utils.data import DataLoader
from training.train_forecasting_net import TrainForecastingGAN
from utils.time_series_utils import multistep_pred
from data_handling.conditional_gan_adv_diff_dataloader import AdvDiffDataset

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':

    train = False

    cuda = True
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    data_path = 'data/advection_diffusion/train_data/adv_diff'

    dataset_params = {
        'data_path': data_path,
        'num_files': 10000,
        'num_states_pr_sample': 128,
        'sample_size': (128, 512),
        'window_size': (32, 1),
        'transformer_state': None,
        'transformer_pars': None,
    }
    dataloader_params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8,
        'drop_last': True,
    }

    dataset = AdvDiffDataset(**dataset_params)
    dataloader = DataLoader(dataset, **dataloader_params)

    forecasting_model = ForecastingNet(
            latent_dim=16,
            in_channels=1,
            hidden_channels=2,
            bilinear=False
    ).to(device)
    checkpoint = torch.load('model_weights/forecasting_net')
    forecasting_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = forecasting_model.encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    generator = ConditionalGenerator(
            z_latent_dim=16,
            in_channels=1,
            hidden_channels=2,
            encoder=encoder,
            bilinear=False
    ).to(device)
    critic = Critic(
            in_channels=1,
            hidden_channels=2,
            encoder=encoder,
            bilinear=False
    ).to(device)


    if train:
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

        trainer = TrainForecastingGAN(
                generator=generator,
                critic=critic,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                scheduler=None,
                save_string='forecasting_gan',
                n_epochs=100,
                device=device
        )
        trainer.train(dataloader=dataloader,
                      temporal_batch_size=128)
    else:
        checkpoint = torch.load('model_weights/forecasting_gan')
        generator.load_state_dict(checkpoint['generator_state_dict'])

    generator.eval()

    num_steps = 100
    test_case = 0

    pred_list = []
    true = dataset[test_case][1].cpu().detach().numpy()

    init = dataset[test_case][0][0:1].unsqueeze(0).repeat(256, 1, 1, 1).to(device)
    for i in range(num_steps - 5):
        init = dataset[test_case][0][i:i+1].unsqueeze(0).repeat(256, 1, 1, 1).to(device)
        z = torch.randn(256, 16, requires_grad=True).to(device)
        pred = generator(z, init, return_input=True)
        #init = pred[:, :, :, -32:].mean(dim=0).unsqueeze(0).repeat(256, 1, 1, 1)
        pred_list.append(pred[:, 0, :, -1:].mean(dim=0))

    pred_list = torch.stack(pred_list).cpu().detach().numpy()
    plt.figure()
    plt.plot(true[0, :, 0], label='true', color='blue')
    plt.plot(true[50, :, 0], label='true', color='blue')
    plt.plot(true[90, :, 0], label='true', color='blue')
    plt.plot(pred_list[0, :, 0], label='pred', color='red')
    plt.plot(pred_list[50, :, 0], label='pred', color='red')
    plt.plot(pred_list[90, :, 0], label='pred', color='red')
    plt.show()
    pdb.set_trace()


