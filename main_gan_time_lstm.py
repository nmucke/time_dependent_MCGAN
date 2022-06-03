import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.latent_dataloader import LatentDataset
import models.adv_diff_models.adversarial_AE as models
from utils.seed_everything import seed_everything
from training.train_adversarial_AE import TrainAdversarialAE
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from torch.utils.data import DataLoader
import models.adv_diff_models.latent_time_gan as time_models
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

torch.set_default_dtype(torch.float32)


if __name__ == '__main__':

    seed_everything()

    continue_training = False
    train = False

    if not train:
        continue_training = True
        cuda = False
    else:
        cuda = True

    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    input_window_size = 16
    output_window_size = 32
    latent_dim = 4

    ##### Load data #####
    data = np.load('reduced_data.npy')
    data = data.reshape(-1, latent_dim)
    state_transformer = MinMaxScaler(feature_range=(0, 1))
    data = state_transformer.fit_transform(data)
    data = data.reshape(10000, 512, latent_dim)
    data = torch.tensor(data, dtype=torch.get_default_dtype())

    data_pars = np.load('reduced_data_pars.npy')
    data_pars = data_pars.reshape(-1, 2)
    data_pars = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_pars)
    data_pars = data_pars.reshape(10000, 512, 2)
    data_pars = torch.tensor(data_pars, dtype=torch.get_default_dtype())

    dataset_parameters = {
        'num_states_pr_sample': 128,
        'sample_size': (latent_dim, 512),
        'window_size': (input_window_size, output_window_size),
    }
    dataloader_parameters = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 4
    }

    dataset = LatentDataset(
        data,
        data_pars,
        **dataset_parameters
    )
    dataloader = DataLoader(dataset=dataset, **dataloader_parameters)

    ##### Load encoder/decoder model#####
    input_dim = 128
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [8, 16, 32, 64],
    }

    decoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [64, 32, 16, 8],
    }

    encoder = models.Encoder(**encoder_params)
    decoder = models.Decoder(**decoder_params)
    decoder.eval()

    checkpoint_path = 'model_weights/AdvAE'
    checkpoint = torch.load(checkpoint_path)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    ##### Define prediction model #####
    model = time_models.lstm_seq2seq(
            input_size=latent_dim,
            output_size=latent_dim,
            hidden_size=16,
            num_layers=2,
            par_size=2
    ).to(device)

    critic = time_models.Critic(
            input_size=latent_dim+2,
            hidden_size=16,
            num_layers=2,
    ).to(device)


    ##### Define optimizer #####
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=1e-4,
    )

    ##### Train model #####

    teacher_forcing_ratio = 0.40

    if train:
        alpha = 5
        num_epochs = 1000
        for epoch in range(num_epochs):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            gen_loss = 0
            critic_loss = 0
            for i, (state, pars) in pbar:
                optimizer.zero_grad()
                optimizer_critic.zero_grad()


                state_conditions = state['state_conditions']

                batch_size = state_conditions.shape[0]
                num_states = state_conditions.shape[1]

                state_conditions = state_conditions.view(batch_size*num_states, input_window_size, latent_dim)
                state_conditions = state_conditions.transpose(0, 1)
                state_conditions = state_conditions.to(device)

                par_conditions = pars['pars_conditions']
                par_conditions = par_conditions.view(batch_size*num_states, input_window_size, 2)
                par_conditions = par_conditions.transpose(0, 1)
                par_conditions = par_conditions.to(device)

                state_preds = state['state_pred']
                state_preds = state_preds.view(batch_size*num_states, output_window_size, latent_dim)
                state_preds = state_preds.transpose(0, 1)
                state_preds = state_preds.to(device)

                model_output = model(
                        state_conditions,
                        target_len=output_window_size,
                        pars=par_conditions,
                        teacher_forcing=None
                )
                model_output = torch.cat((state_conditions, model_output), dim=0)
                model_output = torch.cat((model_output, par_conditions.repeat((3,1,1))), dim=-1)

                labels_gen = torch.zeros(batch_size*num_states, 1).to(device)
                label_true = torch.ones(batch_size*num_states, 1).to(device)

                true_data = torch.cat((state_conditions, state_preds), dim=0)
                true_data = torch.cat((true_data, par_conditions.repeat((3,1,1))), dim=-1)

                critic_output_true = critic(true_data)
                critic_output_gen = critic(model_output)

                critic_loss = -nn.BCELoss()(critic_output_true, label_true) - nn.BCELoss()(critic_output_gen, labels_gen)

                critic_loss.backward()
                optimizer_critic.step()

                model_output = model(
                        state_conditions,
                        target_len=output_window_size,
                        pars=par_conditions,
                        teacher_forcing=None
                )
                model_output = torch.cat((state_conditions, model_output), dim=0)
                model_output = torch.cat((model_output, par_conditions.repeat((3,1,1))), dim=-1)
                critic_output_gen = critic(model_output)
                gen_loss = nn.BCELoss()(critic_output_gen, label_true)
                gen_loss += alpha*nn.MSELoss()(model_output[-output_window_size:, :, 0:latent_dim], state_preds)
                gen_loss.backward()
                optimizer.step()


                #loss = nn.MSELoss()(model_output, state_preds)
                #loss.backward()
                #optimizer.step()

                gen_loss += gen_loss.item()
                critic_loss += critic_loss.item()

                pbar.set_postfix({
                    "gen loss": gen_loss/(i+1e-12),
                    "critic loss": critic_loss/(i+1e-12),
                    'epoch': epoch
                })

            alpha *= 0.9
            teacher_forcing_ratio *= 0.85

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_weights/seq2seq_model')
    else:
        checkpoint_path = 'model_weights/seq2seq_model'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    i, (state, pars) = next(enumerate(dataloader))
    error = 0
    plt.figure()
    for j in range(3):
        state_conditions = state['state_conditions']
        state_conditions = state_conditions[j, 0:1]
        state_conditions = state_conditions.transpose(0, 1)
        state_conditions = state_conditions.to(device)

        par_conditions = pars['pars_conditions']
        par_conditions = par_conditions[j, 0:1]
        par_conditions = par_conditions.transpose(0, 1)
        par_conditions = par_conditions.to(device)

        state_preds = state['state_pred']
        state_preds = state_preds[j, :, 0]
        state_preds = state_transformer.inverse_transform(state_preds)
        state_preds = torch.tensor(state_preds, dtype=torch.get_default_dtype()).to(device)
        decoded_true = decoder(state_preds)
        decoded_true = decoded_true.detach().cpu().numpy()

        outputs = []
        for i in range(200):
            model_output = model(state_conditions, target_len=decoded_true.shape[0], pars=par_conditions)
            outputs.append(model_output)
        outputs = torch.stack(outputs)
        model_output_mean = outputs.mean(dim=0)
        model_output_std = outputs.std(dim=0)
        model_output_std = model_output_std[:, 0, 0:latent_dim].detach().cpu().numpy()


        model_output_mean = model_output_mean[:, 0, 0:latent_dim]
        model_output_mean = state_transformer.inverse_transform(model_output_mean.detach().cpu().numpy())
        model_output_mean = torch.tensor(model_output_mean, dtype=torch.get_default_dtype()).to(device)
        decoded_pred = decoder(model_output_mean)
        decoded_pred = decoded_pred.detach().cpu().numpy()
        model_output_mean = model_output_mean.detach().cpu().numpy()

        plt.subplot(3, 2, 2*j-1+2)
        plt.plot(state_preds[:, 0], label='target', color='tab:blue')
        plt.plot(state_preds[:, 2], color='tab:blue')
        plt.plot(state_preds[:, 3], color='tab:blue')
        plt.plot(model_output_mean[:, 0], label='prediction', color='tab:orange')
        plt.plot(model_output_mean[:, 2], color='tab:orange')
        plt.plot(model_output_mean[:, 3], color='tab:orange')
        plt.fill_between(range(decoded_true.shape[0]),
                         model_output_mean[:, 0]+100*model_output_std[:, 0],
                         model_output_mean[:, 0]-100*model_output_std[:, 0],
                         alpha=0.2, color='tab:orange')
        plt.fill_between(range(decoded_true.shape[0]),
                         model_output_mean[:, 2]+100*model_output_std[:, 2],
                         model_output_mean[:, 2]-100*model_output_std[:, 2],
                         alpha=0.2, color='tab:orange')
        plt.fill_between(range(decoded_true.shape[0]),
                         model_output_mean[:, 3]+100*model_output_std[:, 3],
                         model_output_mean[:, 3]-100*model_output_std[:, 3],
                         alpha=0.2, color='tab:orange')
        #plt.legend()
        plt.grid()

        plt.subplot(3, 2, 2*j+2)
        plt.plot(decoded_true[10, :], label='true', color='tab:blue')
        plt.plot(decoded_true[50, :], color='tab:blue')
        plt.plot(decoded_true[-1, :], color='tab:blue')
        plt.plot(decoded_true[10, :], label='prediction', color='tab:orange')
        plt.plot(decoded_pred[50, :], color='tab:orange')
        plt.plot(decoded_pred[-1, :], color='tab:orange')
        #plt.legend()
        plt.grid()

        error += np.linalg.norm(decoded_true - decoded_pred)/np.linalg.norm(decoded_true)
    print(error/3)
    plt.show()


    error = 0
    for k in range(10):
        i, (state, pars) = next(enumerate(dataloader))
        for j in range(state['state_conditions'].shape[0]):
            state_conditions = state['state_conditions']
            state_conditions = state_conditions[j, 0:1]
            state_conditions = state_conditions.transpose(0, 1)
            state_conditions = state_conditions.to(device)

            par_conditions = pars['pars_conditions']
            par_conditions = par_conditions[j, 0:1]
            par_conditions = par_conditions.transpose(0, 1)
            par_conditions = par_conditions.to(device)

            state_preds = state['state_pred']
            state_preds = state_preds[j, :, 0]
            state_preds = state_transformer.inverse_transform(state_preds)
            state_preds = torch.tensor(state_preds, dtype=torch.get_default_dtype()).to(device)
            decoded_true = decoder(state_preds)
            decoded_true = decoded_true.detach().cpu().numpy()

            model_output = model(state_conditions, target_len=decoded_true.shape[0], pars=par_conditions)

            model_output = model_output[:, 0, 0:latent_dim]
            model_output = state_transformer.inverse_transform(model_output.detach().cpu().numpy())
            model_output = torch.tensor(model_output, dtype=torch.get_default_dtype()).to(device)
            decoded_pred = decoder(model_output)
            decoded_pred = decoded_pred.detach().cpu().numpy()
            model_output = model_output.detach().cpu().numpy()


            error += np.linalg.norm(decoded_true - decoded_pred)/np.linalg.norm(decoded_true)
    print(error/(k+1)/(j+1))





