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
import hamiltorch

torch.set_default_dtype(torch.float32)

def obs_operator(state, ids):
    return torch.take(state, ids)

def log_posterior(z, z_pred, decoder, observations, obs_operator, include_forecast=True, std_obs=0.1):

    zeros = torch.zeros(z.shape[0], 1, device=z.device)
    ones = torch.ones(z.shape[0], 1, device=z.device)
    log_prior = torch.distributions.Normal(zeros, ones).log_prob(z).sum()

    if include_forecast:
        forecast_likelihood = torch.distributions.Normal(
                z_pred, 0.1*ones).log_prob(z).sum()
    else:
        forecast_likelihood = 0

    generated_obs = decoder(z.view(1, z.shape[-1]))
    generated_obs = obs_operator(generated_obs[0])

    obs_likelihood = torch.distributions.Normal(
            observations, std_obs*ones).log_prob(generated_obs).sum()
    return obs_likelihood + forecast_likelihood + log_prior

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
    #data = data.reshape(-1, latent_dim)
    #state_transformer = MinMaxScaler(feature_range=(0, 1))
    #data = state_transformer.fit_transform(data)
    #data = data.reshape(10000, 512, latent_dim)
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
    encoder.eval()
    decoder.eval()

    ##### Define prediction model #####
    model = time_models.lstm_seq2seq(
            input_size=latent_dim,
            output_size=latent_dim,
            hidden_size=16,
            num_layers=2,
            par_size=2
    ).to(device)

    case = 0
    checkpoint_path = 'model_weights/seq2seq_model'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    i, (state, pars) = next(enumerate(dataloader))

    state_conditions = state['state_conditions']
    state_conditions = state_conditions[case, 0:1]
    state_conditions = state_conditions.transpose(0, 1)
    state_conditions = state_conditions.to(device)

    par_conditions = pars['pars_conditions']
    par_conditions = par_conditions[case, 0:1]
    par_conditions = par_conditions.transpose(0, 1)
    par_conditions = par_conditions.to(device)


    state_preds = state['state_pred']
    state_preds = state_preds[case, :, 0]
    #state_preds = state_transformer.inverse_transform(state_preds)
    #state_preds = torch.tensor(state_preds, dtype=torch.get_default_dtype()).to(device)
    decoded_true = decoder(state_preds)
    decoded_true = decoded_true.detach().cpu().numpy()

    model_output_mean = model(state_conditions,
                              target_len=decoded_true.shape[0],
                              pars=par_conditions)

    model_output_mean = model_output_mean[:, 0, 0:latent_dim]
    #model_output_mean = state_transformer.inverse_transform(
    #    model_output_mean.detach().cpu().numpy())
    model_output_mean = torch.tensor(model_output_mean,
                                     dtype=torch.get_default_dtype()).to(device)
    decoded_pred = decoder(model_output_mean)
    decoded_pred = decoded_pred.detach().cpu().numpy()
    model_output_mean = model_output_mean.detach().cpu().numpy()

    error = np.sum(np.power(np.abs(decoded_true - decoded_pred), 2), axis=1)


    ids = range(0,128,5)
    decoded_true_obs = np.take(decoded_true, ids, axis=1)
    observation_operator = lambda x: obs_operator(x, torch.tensor(ids))

    decoded_true_obs = torch.tensor(decoded_true_obs, dtype=torch.get_default_dtype()).to(device)

    std_obs = 0.02
    decoded_true_obs += torch.randn(decoded_true_obs.shape, device=device) * std_obs


    z = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([z], lr=0.1)

    HMC_params = {'num_samples': 1000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 750,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3
                  }


    for epoch in range(250):
        z_pred = model(state_conditions,
                      target_len=1,
                      pars=par_conditions)
        optimizer.zero_grad()
        log_prob = -log_posterior(
                z=z,
                z_pred=z_pred[0],
                decoder=decoder,
                observations=decoded_true_obs[0],
                include_forecast=True,
                std_obs=std_obs,
                obs_operator=observation_operator
        )
        log_prob.backward()
        optimizer.step()

    obs_skip = 4
    preds_z = state_conditions.clone().detach()
    preds_z = torch.cat((preds_z, z.clone().detach().view(1,1,latent_dim)), dim=0)
    obs_times = np.zeros(decoded_true.shape[0])
    obs_times[0] = 1
    for i in range(1, decoded_true.shape[0]):
        z = torch.randn(1, latent_dim, device=device)
        z.requires_grad = True
        #z = torch.tensor(z_pred[0].clone(), requires_grad=True)

        if i % obs_skip == 0:
            obs_times[i] = 1

            optimizer = torch.optim.Adam([z], lr=0.1)

            for epoch in range(250):
                optimizer.zero_grad()
                z_pred = model(preds_z[-16:],
                                  target_len=1,
                                  pars=par_conditions
                               )

                log_prob = -log_posterior(
                    z=z,
                    z_pred=z_pred[0],
                    decoder=decoder,
                    observations=decoded_true_obs[i],
                    include_forecast=True,
                    std_obs=std_obs,
                    obs_operator=observation_operator
                )
                log_prob.backward()
                optimizer.step()

        else:
            z_pred = model(preds_z[-16:],
                           target_len=1,
                           pars=par_conditions
                           )
            z = z_pred[0].detach()

        preds_z = torch.cat((preds_z, z.clone().detach().view(1,1,latent_dim)), dim=0)

        print(i)

    plt.figure()
    plt.plot(preds_z[16:,0,0].detach().cpu().numpy(), label='data assimilation')
    plt.plot(state_preds[:,0].detach().cpu().numpy(), label='true')
    plt.plot(model_output_mean[:,0], label='pred')
    plt.legend()
    plt.show()

    preds = decoder(preds_z[16:])
    preds = preds.detach().cpu().numpy()

    error_lol = np.sum(np.power(np.abs(decoded_true[0:preds.shape[0]] - preds), 2), axis=1)

    plt.figure()
    plt.plot(error, label='pred')
    plt.plot(error_lol, label='data assimilation')
    plt.plot(range(0, state_preds.shape[0], obs_skip),
             np.zeros(len(range(0, state_preds.shape[0], obs_skip))),
             '.k', markersize=5, label='obs times')
    plt.title('Error')
    plt.legend()
    plt.grid()
    plt.savefig('error.png')
    plt.show()


    plt.figure()
    plt.plot(preds[0, :], color='tab:blue', label='Data assimilation')
    plt.plot(preds[30, :], color='tab:blue')
    plt.plot(preds[75, :], color='tab:blue')
    plt.plot(decoded_pred[0, :], color='tab:orange', label='No data assimilation', alpha=0.5)
    plt.plot(decoded_pred[30, :], color='tab:orange', alpha=0.5)
    plt.plot(decoded_pred[75, :], color='tab:orange', alpha=0.5)
    plt.plot(decoded_true[0, :], color='tab:green', label='True')
    plt.plot(decoded_true[30, :], color='tab:green')
    plt.plot(decoded_true[75, :], color='tab:green')
    plt.plot(ids, decoded_true_obs[0, :], '.k', markersize=5, label='Observations')
    plt.plot(ids, decoded_true_obs[30, :], '.k', markersize=5)
    plt.plot(ids, decoded_true_obs[75, :], '.k', markersize=5)
    plt.legend()
    plt.grid()
    plt.savefig('preds.png')
    plt.show()





