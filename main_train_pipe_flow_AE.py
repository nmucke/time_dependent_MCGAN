import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.pipe_flow_dataloader import PipeFlowDataset, TransformState, TransformPars
import models.pipe_flow_models.autoencoder as models
from utils.seed_everything import seed_everything
from training.train_adv_AE_pipe_flow import TrainAdversarialAE
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

if __name__ == '__main__':

    seed_everything()

    with_koopman_training = False
    with_adversarial_training = True
    continue_training = False
    train = True

    if not train:
        continue_training = True
        cuda = True
    else:
        cuda = True

    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    transformer_state = TransformState()
    transformer_pars = TransformPars()
    num_states_pr_sample = 512
    dataset_params = {
        'num_files': 1250,
        'num_states_pr_sample': num_states_pr_sample,
        'sample_size': (1000, 256),
        'pars': True,
        'with_koopman_training': with_koopman_training,
    }
    batch_size = 4
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'drop_last': True,
    }
    data_path = 'pipe_flow/data/pipe_flow'
    dataset = PipeFlowDataset(data_path, **dataset_params)
    dataloader = DataLoader(dataset, **dataloader_params)

    for i, (state, pars) in enumerate(dataloader):
        transformer_state.partial_fit(state.numpy().reshape(batch_size*num_states_pr_sample, 2, 256))
        transformer_pars.partial_fit(pars.numpy())

    dataset = PipeFlowDataset(
            data_path,
            **dataset_params,
            transformer_state=transformer_state,
            transformer_pars=transformer_pars
    )
    dataloader = DataLoader(dataset, **dataloader_params)

    latent_dim = 8
    par_dim = 2
    input_dim = 128
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_channels': [8, 16, 32, 64, 128],
    }

    decoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_channels': [128, 64, 32, 16, 8],
    }

    critic_params = {
        'latent_dim': latent_dim + 1,
        'hidden_neurons': [16, 16],
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

    recon_learning_rate = 1e-3
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

        load_string = 'AE_pipe_flow'
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
        save_string = 'AE_pipe_flow_' + str(latent_dim)
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

        encoder = encoder.to(device)
        decoder = decoder.to(device)

        encoder.eval()
        decoder.eval()

        x_list = []
        num_samples = 2
        for i in range(100, 100+num_samples):
            state, pars = dataloader.dataset[i]
            x_list.append(state)
        x = torch.stack(x_list)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        x = x.to(device)

        z = encoder(x)
        pred = decoder(z)
        pred_u = pred[:, 0]
        pred_u = pred_u.view(num_samples, num_states_pr_sample,
                         256).cpu().detach().numpy()
        pred_p = pred[:, 1]
        pred_p = pred_p.view(num_samples, num_states_pr_sample,
                         256).cpu().detach().numpy()

        z = z.view(num_samples, num_states_pr_sample, latent_dim)
        z = z.cpu().detach().numpy()

        x = x.cpu().detach().numpy()
        x = x.reshape(num_samples, num_states_pr_sample, 2, 256)
        u = x[:, :, 0]
        p = x[:, :, 1]

        t1, t2, t3 = 0, 100, -1
        idx1, idx2, idx3, idx4 = 0, 1, 2, 3

        plt.figure(figsize=(15,20))
        plt.title(load_string)
        plt.subplot(4,2,1)
        plt.plot(u[idx1, t1], color='tab:blue')
        plt.plot(pred_u[idx1, t1], '--',color='tab:red')
        plt.plot(u[idx1, t2], color='tab:blue')
        plt.plot(pred_u[idx1, t2], '--', color='tab:red')
        plt.plot(u[idx1, t3], color='tab:blue')
        plt.plot(pred_u[idx1, t3], '--', color='tab:red')
        plt.grid()

        plt.subplot(4,2,2)
        plt.plot(u[idx2, t1], color='tab:blue')
        plt.plot(pred_u[idx2, t1], '--', color='tab:red')
        plt.plot(u[idx2, t2], color='tab:blue')
        plt.plot(pred_u[idx2, t2], '--', color='tab:red')
        plt.plot(u[idx2, t3], color='tab:blue')
        plt.plot(pred_u[idx2, t3], '--', color='tab:red')
        plt.grid()

        plt.subplot(4,2,3)
        plt.plot(p[idx1, t1], color='tab:blue')
        plt.plot(pred_p[idx1, t1], '--',color='tab:red')
        plt.plot(p[idx1, t2], color='tab:blue')
        plt.plot(pred_p[idx1, t2], '--', color='tab:red')
        plt.plot(p[idx1, t3], color='tab:blue')
        plt.plot(pred_p[idx1, t3], '--', color='tab:red')
        plt.grid()

        plt.subplot(4,2,4)
        plt.plot(p[idx2, t1], color='tab:blue')
        plt.plot(pred_p[idx2, t1], '--', color='tab:red')
        plt.plot(p[idx2, t2], color='tab:blue')
        plt.plot(pred_p[idx2, t2], '--', color='tab:red')
        plt.plot(p[idx2, t3], color='tab:blue')
        plt.plot(pred_p[idx2, t3], '--', color='tab:red')
        plt.grid()

        #plt.subplot(4,2,5)
        #plt.plot(z[idx1, :, 0], z[idx1, :, 1])
        #plt.plot(z[idx2, :, 0], z[idx2, :, 1])
        #plt.grid()

        plt.subplot(4,1,3)
        for i in range(3):
            plt.plot(z[idx1, :, i])
        plt.grid()

        zz = torch.randn(10, latent_dim, device=device)
        gen_data = decoder(zz).cpu().detach().numpy()
        gen_data = gen_data[:, 0]

        plt.subplot(4,2,7)
        plt.plot(gen_data[idx1])
        plt.plot(gen_data[idx2])
        plt.plot(gen_data[idx3])
        plt.plot(gen_data[idx4])
        plt.grid()

        dataset = PipeFlowDataset(
                data_path, **dataset_params,
                transformer_state=transformer_state,
                transformer_pars=transformer_pars
        )

        x_list = []
        par_list = []
        num_samples = 1
        for i in range(num_samples):
            state, pars = dataset[i]
            x_list.append(state)
            par_list.append(pars)
        x = torch.stack(x_list)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        x = x.to(device)

        pars = torch.stack(par_list)
        pars = pars.reshape(-1, pars.shape[1]).detach().numpy()
        pars = pars[:,1:2]
        pars = np.tile(pars, (1,num_states_pr_sample))
        pars = pars.reshape(num_states_pr_sample*num_samples)
        z = encoder(x).cpu().detach().numpy()

        time_vec = np.linspace(0, 1.75, num_states_pr_sample)
        time_vec = time_vec.reshape(1, -1)
        time_vec = np.tile(time_vec, (num_samples, 1))
        time_vec = time_vec.reshape(-1)
        plt.subplot(4,2,8)
        plt.scatter(z[:, 0], z[:, 1], c=time_vec, alpha=0.2)
        plt.grid()

        plt.show()







