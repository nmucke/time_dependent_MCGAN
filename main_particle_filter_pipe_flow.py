import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from inference.particle_filter import ParticleFilter, LogLikelihood
import models.pipe_flow_models.autoencoder as models
from utils.seed_everything import seed_everything
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import models.adv_diff_models.custom_transformer as time_models
from tqdm import tqdm
from data_handling.pipe_flow_dataloader import PipeFlowDataset, TransformState, TransformPars
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import matplotlib.animation as animation



def animateSolution(x,time,sol_list1, sol_list2, sol_list3, particle_std, gif_name='pipe_flow_simulation',
                    xlabel='Location', ylabel='Pressure', legend=[]):
    fig = plt.figure()
    ax1 = plt.axes(xlim=(x[0], x[-1]), ylim=(
        min(np.min(np.asarray(sol_list1)),np.min(np.asarray(sol_list2)), np.min(np.asarray(sol_list3))),
        max(np.max(np.asarray(sol_list1)),np.max(np.asarray(sol_list2)), np.max(np.asarray(sol_list3)))
        ))
    lines = []
    for index in range(3):
        lobj = ax1.plot([], [], lw=2, label=legend[index])[0]
        lines.append(lobj)

    lobj = ax1.plot([], [], '--', lw=1, color='tab:blue', label='Std')[0]
    lines.append(lobj)
    lobj = ax1.plot([], [], '--', lw=1, color='tab:blue')[0]
    lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        plt.title(f'{time[i]:0.2f} seconds')
        print(i)


        y1 = sol_list1[i]
        y2 = sol_list2[i]
        y3 = sol_list3[i]

        ylist = [y1, y2, y3, y1 + particle_std[i], y1 - particle_std[i]]

        plt.legend(loc='upper left')

        for lnum, line in enumerate(lines):
            line.set_data(x, ylist[lnum])

        return lines

    writergif = animation.PillowWriter(fps=10)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(sol_list1), interval=10, blit=True)

    # save the animation as mp4 video file
    anim.save(gif_name + '.gif', writer=writergif)


def space_obs_operator(state, ids, space_dim=1):
    return torch.index_select(state[:, 0, :], space_dim, ids)

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
    latent_dim = 16
    par_dim = 2
    num_states_pr_sample = 500
    num_t = 1000
    num_x = 256
    num_samples = 2000
    sample_time_ids = np.linspace(0, num_t, num_states_pr_sample,
                                  dtype=int, endpoint=False)

    transformer_state = TransformState()
    transformer_pars = TransformPars()
    dataset_params = {
        'num_files': 2000,
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

    ##### Load data #####
    data_pars = np.load('reduced_data_pipe_flow_pars_' + str(latent_dim) + '.npy')
    #data_pars = data_pars.reshape(-1, 2)
    #pars_transformer = StandardScaler()
    #pars_transformer.fit(data_pars)

    ##### Load encoder/decoder model#####
    input_dim = 256
    encoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_channels': [16, 32, 64, 128, 256],
    }

    decoder_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_channels': [256, 128, 64, 32, 16],
    }
    encoder = models.Encoder(**encoder_params)
    decoder = models.Decoder(**decoder_params)

    load_string = 'AE_pipe_flow_large_' + str(latent_dim)
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

    encoder.eval()
    decoder.eval()
    ##### Define prediction model #####
    prediction_model_params = {
        'latent_dim': latent_dim,
        'pars_dim': par_dim,
        'num_layers': 2,
        'embed_dim': 64,
        'num_heads': 4,
        'hidden_mlp_dim': 64,
        'out_features': latent_dim,
        'dropout_rate': 0.0,
        'device': device,
    }

    model = time_models.Transformer(**prediction_model_params).to(device)
    checkpoint_path = 'model_weights/transformer_model_pipe_flow'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ##### Load Data #####

    case = 0
    data = np.load(f'pipe_flow/data/pipe_flow_{case}.npy', allow_pickle=True)
    data = data.item()

    hf_u = data['u']
    hf_p = data['pressure']
    hf_state = np.stack((hf_u, hf_p), axis=1)
    hf_state = transformer_state.transform(hf_state)
    hf_state = torch.tensor(hf_state, dtype=torch.get_default_dtype(), device=device)
    hf_state = hf_state[sample_time_ids]

    true_pars = data['params']
    true_pars = np.array([[true_pars['friction'], true_pars['inflow_freq']]])
    true_pars = transformer_pars.transform(true_pars)
    true_pars = torch.tensor(true_pars, dtype=torch.get_default_dtype(), device=device)

    latent_state = encoder(hf_state)

    ###### Set up observations #####
    space_obs_ids = torch.arange(0, num_x, 20, device=device)
    num_obs_x = space_obs_ids.shape[0]
    time_obs_ids = torch.arange(0, num_states_pr_sample, 17, device=device)
    time_obs_mask = torch.zeros(num_states_pr_sample, dtype=torch.bool, device=device)
    time_obs_mask[time_obs_ids] = 1

    space_observation_operator = lambda x: space_obs_operator(x, space_obs_ids)
    observations = space_observation_operator(hf_state)

    std_obs = .1

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
    x_vec = np.linspace(0, 1041, 256)
    num_particles = 1500
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

    init_pars = true_pars.repeat(num_particles, 1)
    #init_particles = latent_state[0:input_window_size].unsqueeze(0).repeat(num_particles, 1, 1)
    #nit_particles += torch.normal(
    #        torch.zeros(init_particles.shape),
    #        1.*torch.ones(init_particles.shape)
    #).to(device)

    init_particles = np.load('reduced_data_pipe_flow_16.npy')[0:num_particles]
    init_particles = init_particles[:, sample_time_ids]

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
                num_steps=num_steps,
                batch_size=16
        )
        particles = torch.cat([particles, particles_preds], dim=1)

        pars_particles = torch.cat(
                [pars_particles, particle_pars_preds.unsqueeze(1).repeat(1, num_steps, 1)],
                dim=1
        )

        particle_filter.update_weights(
                particles=particles[:, -1],
                obs=observations[t_id],
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
            num_steps=num_states_pr_sample-t_id
    )
    particles = torch.cat([particles, particles_preds], dim=1)
    pars_particles = torch.cat(
            [pars_particles,
             particle_pars_preds.unsqueeze(1).repeat(1, num_states_pr_sample-t_id, 1)],
             dim=1
    )

    batch_size = 16
    hf_particles = torch.zeros((num_particles, num_states_pr_sample, 2, num_x))
    for i in range(0, num_particles, batch_size):
        decoded_particles = decoder(particles[i:i+batch_size].reshape(-1, latent_dim))
        if particles[i:i+batch_size].shape[0] == batch_size:
            decoded_particles = decoded_particles.reshape(batch_size, num_states_pr_sample, 2, num_x)
        else:
            decoded_particles = decoded_particles.reshape(particles[i:i+batch_size].shape[0], num_states_pr_sample, 2, num_x)
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
    init_hf_particles = torch.zeros((num_particles, num_states_pr_sample, 2, num_x))
    for i in range(0, num_particles, batch_size):
        decoded_particles = decoder(init_particles[i:i+batch_size].reshape(-1, latent_dim))
        if particles[i:i+batch_size].shape[0] == batch_size:
            decoded_particles = decoded_particles.reshape(batch_size, num_states_pr_sample, 2, num_x)
        else:
            decoded_particles = decoded_particles.reshape(particles[i:i+batch_size].shape[0], num_states_pr_sample, 2, num_x)
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
        np.linalg.norm(hf_particles_mean - hf_state, axis=(1, 2))/np.linalg.norm(hf_state, axis=(1, 2))
    error_hf_init_particles = \
        np.linalg.norm(init_hf_particles_mean - hf_state, axis=(1, 2))/np.linalg.norm(hf_state, axis=(1, 2))

    pars_particles = pars_particles.detach().cpu().numpy()
    pars_particles_mean = np.mean(pars_particles, axis=0)
    pars_particles_std = np.std(pars_particles, axis=0)

    true_pars = true_pars.cpu().detach()

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

    t0, t1, t2 = 0, 256, -1
    plt.subplot(4, 1, 2)
    for t in [t0, t1, t2]:
        plt.plot(x_vec, hf_particles_mean[t, 0], linewidth=1, label='Filtered', color='tab:blue')
        plt.fill_between(
                x_vec,
                hf_particles_mean[t, 0] + hf_particles_std[t, 0],
                hf_particles_mean[t, 0] - hf_particles_std[t, 0],
                alpha=0.2,
                color='tab:blue'
        )
        plt.plot(x_vec, init_hf_particles_mean[t, 0], linewidth=1, label='No filter', color='tab:orange')
        plt.plot(x_vec, hf_state[t, 0], linewidth=2, color='k', label='True')
    plt.legend(['Filtered', 'No Filter', 'True'])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('High fidelity state')

    plt.subplot(4, 1, 3)
    for t in time_obs_ids:
        plt.axvline(t, color='k', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.semilogy(range(num_states_pr_sample), error_hf_particles, label='Filtered', color='tab:blue')
    plt.semilogy(range(num_states_pr_sample), error_hf_init_particles, label='No filter', color='tab:orange')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')

    plt.subplot(4, 1, 4)
    plt.plot(range(num_states_pr_sample), pars_particles_mean[:,0], '--', label='par 1', color='tab:blue')
    plt.fill_between(
            range(num_states_pr_sample),
            pars_particles_mean[:,0] + pars_particles_std[:,0],
            pars_particles_mean[:,0] - pars_particles_std[:,0],
            alpha=0.05,
            color='tab:blue'
    )
    plt.plot(range(num_states_pr_sample), pars_particles_mean[:,1], '--', label='par 2', color='tab:red')
    plt.fill_between(
            range(num_states_pr_sample),
            pars_particles_mean[:,1] + pars_particles_std[:,1],
            pars_particles_mean[:,1] - pars_particles_std[:,1],
            alpha=0.05,
            color='tab:red'
    )
    plt.plot(range(num_states_pr_sample), true_pars[0,0].repeat(num_states_pr_sample), color='tab:blue')
    plt.plot(range(num_states_pr_sample), true_pars[0,1].repeat(num_states_pr_sample), color='tab:red')
    plt.legend(['Friction', 'inflow_freq'])
    plt.xlabel('Time')
    plt.ylabel('Parameters')
    plt.show()

    animateSolution(
            x_vec,
            np.linspace(0, 25, num_states_pr_sample),
            hf_particles_mean[:,0],
            init_hf_particles_mean[:,0],
            hf_state[:,0],
            particle_std=hf_particles_std[:,0],
            gif_name='pipe_flow_simulation',
            xlabel='Location',
            ylabel='Velocity',
            legend=['Filtered', 'No filter', 'True', 'Std']
    )


