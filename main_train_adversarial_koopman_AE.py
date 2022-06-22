import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.adv_diff_dataloader import get_dataloader
import models.adv_diff_models.adversarial_AE as models
from utils.seed_everything import seed_everything
from training.train_adversarial_koopman_AE import TrainAdversarialAE
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.set_default_dtype(torch.float32)

if __name__ == '__main__':

    seed_everything()

    with_koopman_training = False
    with_adversarial_training = True
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

    data_pars = np.load('reduced_data_pars.npy')
    data_pars = data_pars.reshape(-1, 2)
    transformer_pars = StandardScaler()
    transformer_pars.fit(data_pars)

    dataloader_params = {
        'num_files': 2000,
        'transformer_state': None,
        'transformer_pars': transformer_pars.transform,
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8,
        'drop_last': True,
        'num_states_pr_sample': 512,
        'sample_size': (128, 512),
        'pars': True,
        'with_koopman_training': with_koopman_training,
    }
    data_path = 'data/advection_diffusion/train_data/adv_diff'
    dataloader = get_dataloader(data_path, **dataloader_params)

    latent_dim = 4
    input_dim = 128
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [4, 8, 16, 32],
    }

    decoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [32, 16, 8, 4],
    }

    critic_params = {
        'latent_dim': latent_dim + 1,
        'hidden_neurons': [8, 8],
    }

    koopman_params = {
        'latent_dim': latent_dim,
        'par_dim': 2,
        'hidden_neurons': 8,
        'num_diags': 5,
    }

    encoder = models.Encoder(**encoder_params).to(device)
    decoder = models.Decoder(**decoder_params).to(device)
    critic = models.Critic(**critic_params).to(device)
    koopman = models.Koopman(**koopman_params).to(device)

    recon_learning_rate = 1e-2
    recon_weight_decay = 1e-6
    koopman_weight_decay = 1e-5

    critic_learning_rate = 1e-2

    encoder_optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=recon_learning_rate,
            #weight_decay=recon_weight_decay
    )
    decoder_optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=recon_learning_rate,
            #weight_decay=recon_weight_decay
    )
    critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=critic_learning_rate
    )
    koopman_optimizer = torch.optim.Adam(
            koopman.parameters(),
            lr=recon_learning_rate,
            #weight_decay=koopman_weight_decay
    )

    if continue_training:

        load_string = 'AE'
        if with_koopman_training and with_adversarial_training:
            load_string += '_koopman_adversarial'
        elif with_adversarial_training:
                load_string += '_adversarial'
        elif with_koopman_training:
                load_string += '_koopman'
        load_checkpoint(
            checkpoint_path=f'model_weights/{load_string}',
            encoder=encoder,
            decoder=decoder,
            critic=critic,
            koopman=koopman,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            critic_optimizer=critic_optimizer,
            koopman_optimizer=koopman_optimizer,
        )


    if train:
        save_string = 'AE'
        if with_koopman_training and with_adversarial_training:
            save_string += '_koopman_adversarial'
        elif with_adversarial_training:
                save_string += '_adversarial'
        elif with_koopman_training:
                save_string += '_koopman'

        training_params = {
            'n_critic': 1,
            'gamma': 10,
            'n_epochs': 500,
            'save_string': 'model_weights/'+save_string,
            'with_koopman_training': with_koopman_training,
            'with_adversarial_training': with_adversarial_training,
            'device': device
        }
        trainer = TrainAdversarialAE(
                encoder=encoder,
                decoder=decoder,
                critic=critic,
                koopman=koopman,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                critic_optimizer=critic_optimizer,
                koopman_optimizer=koopman_optimizer,
                latent_dim=latent_dim,
                **training_params
        )

        trainer.train(dataloader=dataloader)

    else:

        encoder.eval()
        decoder.eval()

        x_list = []
        num_samples = 4
        for i in range(100, 100+num_samples):
            state, pars = dataloader.dataset[i]
            x_list.append(state)
        x = torch.stack(x_list)
        x = x.reshape(-1, x.shape[2])

        z = encoder(x)
        pred = decoder(z)
        pred = pred.view(num_samples, dataloader_params['num_states_pr_sample'],
                         128).detach().numpy()

        z = z.view(num_samples, dataloader_params['num_states_pr_sample'], latent_dim)
        z = z.detach().numpy()

        x = x.detach().numpy()
        x = x.reshape(num_samples, dataloader_params['num_states_pr_sample'], 128)

        t1, t2, t3 = 0, 100, -1
        idx1, idx2, idx3, idx4 = 0, 1, 2, 3

        plt.figure(figsize=(12,16))
        plt.title(load_string)
        plt.subplot(3,2,1)
        plt.plot(x[idx1, t1], color='tab:blue')
        plt.plot(pred[idx1, t1], '--',color='tab:red')
        plt.plot(x[idx1, t2], color='tab:blue')
        plt.plot(pred[idx1, t2], '--', color='tab:red')
        plt.plot(x[idx1, t3], color='tab:blue')
        plt.plot(pred[idx1, t3], '--', color='tab:red')
        plt.grid()

        plt.subplot(3,2,2)
        plt.plot(x[idx2, t1], color='tab:blue')
        plt.plot(pred[idx2, t1], '--', color='tab:red')
        plt.plot(x[idx2, t2], color='tab:blue')
        plt.plot(pred[idx2, t2], '--', color='tab:red')
        plt.plot(x[idx2, t3], color='tab:blue')
        plt.plot(pred[idx2, t3], '--', color='tab:red')
        plt.grid()

        plt.subplot(3,2,3)
        plt.plot(z[idx1, :, 0], z[idx1, :, 1])
        plt.plot(z[idx2, :, 0], z[idx2, :, 1])
        plt.grid()

        plt.subplot(3,2,4)
        for i in range(latent_dim):
            plt.plot(z[idx1, :, i])
        plt.grid()

        zz = torch.randn(10, latent_dim)
        gen_data = decoder(zz).detach().numpy()

        plt.subplot(3,2,5)
        plt.plot(gen_data[idx1])
        plt.plot(gen_data[idx2])
        plt.plot(gen_data[idx3])
        plt.plot(gen_data[idx4])
        plt.grid()

        dataloader_params = {
            'num_files': 100000,
            'transformer_state': None,
            'transformer_pars': None,
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 8,
            'drop_last': True,
            'num_states_pr_sample': 64,
            'sample_size': (128, 512),
            'pars': True
        }
        data_path = 'data/advection_diffusion/train_data/adv_diff'
        dataloader = get_dataloader(data_path, **dataloader_params)


        x_list = []
        par_list = []
        num_samples = 500
        for i in range(num_samples):
            state, pars = dataloader.dataset[i]
            x_list.append(state)
            par_list.append(pars)
        x = torch.stack(x_list)
        x = x.reshape(-1, x.shape[2])

        pars = torch.stack(par_list)
        pars = pars.reshape(-1, pars.shape[1]).detach().numpy()
        pars = pars[:,1:2]
        pars = np.tile(pars, (1,dataloader_params['num_states_pr_sample']))
        pars = pars.reshape(dataloader_params['num_states_pr_sample']*num_samples)
        z = encoder(x).detach().numpy()

        time_vec = np.linspace(0, 1.75, dataloader_params['num_states_pr_sample'])
        time_vec = time_vec.reshape(1, -1)
        time_vec = np.tile(time_vec, (num_samples, 1))
        time_vec = time_vec.reshape(-1)
        plt.subplot(3,2,6)
        plt.scatter(z[:, 0], z[:, 1], c=time_vec, alpha=0.01)
        plt.grid()

        plt.show()







