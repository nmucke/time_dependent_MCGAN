import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from inference.particle_filter import ParticleFilter, LogLikelihood
import models.adv_diff_models.adversarial_AE as models
from utils.seed_everything import seed_everything
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import models.adv_diff_models.custom_transformer as time_models
from tqdm import tqdm

torch.set_default_dtype(torch.float32)


def space_obs_operator(state, ids, space_dim=1):
    return torch.index_select(state, space_dim, ids)

if __name__ == '__main__':

    seed_everything()
    with_koopman_training = False
    with_adversarial_training = True

    cuda = True

    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    input_window_size = 16
    output_window_size = 32
    latent_dim = 4
    par_dim = 2
    num_states_pr_sample = 512
    num_t = 512
    num_x = 128
    num_samples = 2000

    ##### Load data #####
    data_pars = np.load('reduced_data_pars.npy')
    data_pars = data_pars.reshape(-1, 2)
    pars_transformer = StandardScaler()
    pars_transformer.fit(data_pars)

    ##### Load encoder/decoder model#####
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
    encoder = models.Encoder(**encoder_params)
    decoder = models.Decoder(**decoder_params)
    decoder.eval()

    load_string = 'AE'
    if with_koopman_training and with_adversarial_training:
        load_string += '_koopman_adversarial'
    elif with_adversarial_training:
        load_string += '_adversarial'
    elif with_koopman_training:
        load_string += '_koopman'

    checkpoint_path = 'model_weights/' + load_string
    checkpoint = torch.load(checkpoint_path)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    ##### Define prediction model #####
    prediction_model_params = {
        'latent_dim': latent_dim,
        'pars_dim': par_dim,
        'num_layers': 1,
        'embed_dim': 64,
        'num_heads': 2,
        'hidden_mlp_dim': 64,
        'out_features': latent_dim,
        'dropout_rate': 0.0,
        'device': device,
    }

    model = time_models.Transformer(**prediction_model_params)
    checkpoint_path = 'model_weights/transformer_model'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    ##### Load Data #####
    case = 0
    data = np.load(f'data/advection_diffusion/train_data/adv_diff_{case}.npy', allow_pickle=True)
    data = data.item()
    hf_state = data['sol'].transpose()
    true_pars = data['PDE_params']
    true_pars = np.array([[true_pars['velocity'], true_pars['diffusion']]])
    true_pars = pars_transformer.transform(true_pars)
    data = np.load(f'data/advection_diffusion/train_data/adv_diff_{case+1}.npy', allow_pickle=True)
    data = data.item()
    #pars = data['PDE_params']
    #pars = np.array([[pars['velocity'], pars['diffusion']]])
    pars = np.array([[0., 0.]])
    pars = pars_transformer.transform(pars)

    hf_state = torch.tensor(hf_state, dtype=torch.get_default_dtype(), device=device)
    pars = torch.tensor(pars, dtype=torch.get_default_dtype(), device=device)

    latent_state = encoder(hf_state)

    ###### Set up observations #####
    space_obs_ids = torch.arange(0, 128, 15, device=device)
    num_obs_x = space_obs_ids.shape[0]
    time_obs_ids = torch.arange(0, 512, 20, device=device)
    time_obs_mask = torch.zeros(num_states_pr_sample, dtype=torch.bool, device=device)
    time_obs_mask[time_obs_ids] = 1

    space_observation_operator = lambda x: space_obs_operator(x, space_obs_ids)
    observations = space_observation_operator(hf_state)

    std_obs = .25

    noise_distribution = torch.distributions.Normal(
            torch.zeros(num_obs_x).to(device),
            std_obs*torch.ones(num_obs_x).to(device)
    )

    observations += noise_distribution.sample(sample_shape=(observations.shape[0],))

    ###### Set up likelihood #####
    log_likelihood = LogLikelihood(
            decoder=decoder,
            obs_operator=space_observation_operator,
            noise_distribution=noise_distribution
    )

    ###### Set up particle filter #####
    x_vec = np.linspace(-1, 1, 128)
    num_particles = 7500
    particle_filter = ParticleFilter(
            forecast_model=model,
            num_particles=num_particles,
            log_likelihood=log_likelihood,
            weight_update='bootstrap',
            x_vec=x_vec,
            model_forecast_std=.1,
            par_std=.25,
            device=device
    )

    init_pars = pars.repeat(num_particles, 1)
    #init_particles = latent_state[0:input_window_size].unsqueeze(0).repeat(num_particles, 1, 1)
    #nit_particles += torch.normal(
    #        torch.zeros(init_particles.shape),
    #        1.*torch.ones(init_particles.shape)
    #).to(device)

    init_particles = np.load('reduced_data.npy')[0:num_particles]

    particles = init_particles[:, 0:input_window_size]
    particles = torch.tensor(
            particles,
            dtype=torch.get_default_dtype(),
            device=device
    )
    pars_particles = torch.randn((num_particles, par_dim)).to(device)
    pars_particles = pars_particles.unsqueeze(1)
    pars_particles = pars_particles.repeat(1, input_window_size, 1)
    current_t_id = input_window_size
    progress_bar = tqdm(time_obs_ids[1:], total=len(time_obs_ids[1:]))
    for t_id in progress_bar:
        num_steps = t_id - current_t_id


        particles_preds, particle_pars_preds = particle_filter.generate_particles(
                state=particles[:, -input_window_size:],
                pars=pars_particles[:, -1, :],
                num_steps=num_steps
        )
        particles = torch.cat([particles, particles_preds], dim=1)

        pars_particles = torch.cat(
                [pars_particles, particle_pars_preds.unsqueeze(1).repeat(1, num_steps, 1)],
                dim=1
        )


        particle_filter.update_weights(
                particles=particles[:, -1],
                obs=observations[t_id]
        )


        resampling = 'no'
        if particle_filter.ESS.item() < particle_filter.ESS_threshold:
            resampling = 'yes'

            resample_particle_ids = particle_filter.get_resample_particle_ids()

            #particles_preds = particles_preds[resample_particle_ids]
            particles = particles[resample_particle_ids]
            pars_particles = pars_particles[resample_particle_ids]

        #particles = torch.cat([particles, particles_preds], dim=1)

        #particle_pars_preds = particle_filter.update_pars(
        #        state=particles[:, -input_window_size:],
        #        pars=pars_particles[:, -1],
        #        obs=observations[t_id],
        #        num_steps=num_steps
        #)
        '''
        particles_preds, _ = particle_filter.generate_particles(
                state=particles[:, -input_window_size:],
                pars=pars_particles[:, -1, :],
                num_steps=num_steps
        )

        particle_temp = torch.cat([particles[:, -input_window_size:], particles_preds], dim=1)
        particle_filter.update_weights(
                particles=particle_temp[:, -1],
                obs=observations[t_id],
                lol=True
        )
        '''



        current_t_id = t_id
        progress_bar.set_postfix({'Resampling': resampling})

    particles_preds, particle_pars_preds = particle_filter.generate_particles(
            state=particles[:, -input_window_size:],
            pars=pars_particles[:, -1, :],
            num_steps=num_t-t_id
    )
    particles = torch.cat([particles, particles_preds], dim=1)
    pars_particles = torch.cat(
            [pars_particles,
             particle_pars_preds.unsqueeze(1).repeat(1, num_t-t_id, 1)],
             dim=1
    )

    batch_size = 512
    hf_particles = torch.zeros((num_particles, num_t, num_x))
    for i in range(0, num_particles, batch_size):
        decoded_particles = decoder(particles[i:i+batch_size].reshape(-1, latent_dim))
        if particles[i:i+batch_size].shape[0] == batch_size:
            decoded_particles = decoded_particles.reshape(batch_size, num_t, num_x)
        else:
            decoded_particles = decoded_particles.reshape(particles[i:i+batch_size].shape[0], num_t, num_x)
        hf_particles[i:i+batch_size] = decoded_particles.cpu().detach()


    hf_particles = hf_particles.detach().cpu().numpy()
    particles = particles.detach().cpu().numpy()
    latent_state = latent_state.detach().cpu().numpy()

    hf_particles_mean = np.mean(hf_particles, axis=0)
    hf_particles_std = np.std(hf_particles, axis=0)

    particles_latent_mean = np.mean(particles, axis=0)
    particles_latent_std = np.std(particles, axis=0)

    init_particles_latent_mean = np.mean(init_particles, axis=0)
    init_particles_latent_std = np.std(init_particles, axis=0)
    init_particles = torch.tensor(init_particles, dtype=torch.get_default_dtype(), device=device)
    init_hf_particles = torch.zeros((num_particles, num_t, num_x))
    for i in range(0, num_particles, batch_size):
        decoded_particles = decoder(init_particles[i:i+batch_size].reshape(-1, latent_dim))
        if particles[i:i+batch_size].shape[0] == batch_size:
            decoded_particles = decoded_particles.reshape(batch_size, num_t, num_x)
        else:
            decoded_particles = decoded_particles.reshape(particles[i:i+batch_size].shape[0], num_t, num_x)
        init_hf_particles[i:i+batch_size] = decoded_particles.cpu().detach()

    init_hf_particles = init_hf_particles.detach().cpu().numpy()
    init_particles = init_particles.detach().cpu().numpy()
    init_hf_particles_mean = np.mean(init_hf_particles, axis=0)
    init_hf_particles_std = np.std(init_hf_particles, axis=0)

    #num_t = hf_particles_mean.shape[0]
    #hf_state = hf_state[0:num_t]
    hf_state = hf_state.detach().cpu().numpy()
    #init_hf_particles_mean = init_hf_particles_mean[0:num_t]

    error_hf_particles = \
        np.linalg.norm(hf_particles_mean - hf_state, axis=1)/np.linalg.norm(hf_state, axis=1)
    error_hf_init_particles = \
        np.linalg.norm(init_hf_particles_mean - hf_state, axis=1)/np.linalg.norm(hf_state, axis=1)

    pars_particles = pars_particles.detach().cpu().numpy()
    pars_particles_mean = np.mean(pars_particles, axis=0)
    pars_particles_std = np.std(pars_particles, axis=0)

    ###### Plot #####
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    for i in range(2):
        plt.plot(particles_latent_mean[:, i], linewidth=2, label='Filtered', color='tab:blue')
        plt.fill_between(
                range(particles_latent_mean.shape[0]),
                particles_latent_mean[:, i] + particles_latent_std[:, i],
                particles_latent_mean[:, i] - particles_latent_std[:, i],
                alpha=0.2,
                color='tab:blue'
        )

        plt.plot(init_particles_latent_mean[:, i], linewidth=2, label='Filtered', color='tab:orange')

        plt.plot(latent_state[:, i], linewidth=2, color='k', label='True')
    plt.legend(['Filtered', 'No Filter'])
    plt.xlabel('Time')
    plt.ylabel('Latent State')
    plt.grid()

    t0, t1, t2 = 0, 256, 500
    plt.subplot(4, 1, 2)
    for t in [t0, t1, t2]:
        plt.plot(x_vec, hf_particles_mean[t], linewidth=1, label='Filtered', color='tab:blue')
        plt.fill_between(
                x_vec,
                hf_particles_mean[t] + hf_particles_std[t],
                hf_particles_mean[t] - hf_particles_std[t],
                alpha=0.2,
                color='tab:blue'
        )
        plt.plot(x_vec, init_hf_particles_mean[t], linewidth=1, label='No filter', color='tab:orange')
        plt.plot(x_vec, hf_state[t], linewidth=2, color='k', label='True')
    plt.legend(['Filtered', 'No Filter', 'True'])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('High fidelity state')

    plt.subplot(4, 1, 3)
    for t in time_obs_ids:
        plt.axvline(t, color='k', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.semilogy(range(num_t), error_hf_particles, label='Filtered', color='tab:blue')
    plt.semilogy(range(num_t), error_hf_init_particles, label='No filter', color='tab:orange')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')

    plt.subplot(4, 1, 4)
    plt.plot(range(num_t), pars_particles_mean[:,0], '--', label='par 1', color='tab:blue')
    plt.fill_between(
            range(num_t),
            pars_particles_mean[:,0] + pars_particles_std[:,0],
            pars_particles_mean[:,0] - pars_particles_std[:,0],
            alpha=0.05,
            color='tab:blue'
    )
    plt.plot(range(num_t), pars_particles_mean[:,1], '--', label='par 2', color='tab:red')
    plt.fill_between(
            range(num_t),
            pars_particles_mean[:,1] + pars_particles_std[:,1],
            pars_particles_mean[:,1] - pars_particles_std[:,1],
            alpha=0.05,
            color='tab:red'
    )
    plt.plot(range(num_t), true_pars[0,0].repeat(num_t), color='tab:blue')
    plt.plot(range(num_t), true_pars[0,1].repeat(num_t), color='tab:red')
    plt.legend(['Velocity', 'Diffusion'])
    plt.xlabel('Time')
    plt.ylabel('Parameters')
    plt.show()


