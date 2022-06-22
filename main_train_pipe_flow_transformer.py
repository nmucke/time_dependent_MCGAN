import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.latent_dataloader import LatentDatasetTransformersCustom
import models.pipe_flow_models.autoencoder as models
from utils.seed_everything import seed_everything
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from torch.utils.data import DataLoader
import models.adv_diff_models.custom_transformer as time_models
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_handling.pipe_flow_dataloader import PipeFlowDataset, TransformState, TransformPars
torch.set_default_dtype(torch.float32)


if __name__ == '__main__':

    seed_everything()

    with_koopman_training = False
    with_adversarial_training = True

    continue_training = False
    train = False

    if not train:
        continue_training = True
        cuda = True
    else:
        cuda = True

    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    input_window_size = 16
    output_window_size = 32
    latent_dim = 16
    par_dim = 2
    num_states_pr_sample = 1000
    num_t = 1000
    num_x = 256
    num_samples = 2000

    ##### Load data #####
    data = np.load('reduced_data_pipe_flow_' + str(latent_dim) + '.npy')
    data = torch.tensor(data, dtype=torch.get_default_dtype())

    data_pars = np.load('reduced_data_pipe_flow_pars_' + str(latent_dim) + '.npy')
    data_pars = data_pars.reshape(-1, 2)
    #pars_transformer = StandardScaler()
    #pars_transformer.fit(data_pars)
    #data_pars = pars_transformer.transform(data_pars)
    data_pars = data_pars.reshape(num_samples, num_t, par_dim)
    data_pars = torch.tensor(data_pars, dtype=torch.get_default_dtype())

    dataset_parameters = {
        'num_states_pr_sample': num_states_pr_sample,
        'sample_size': (latent_dim, num_t),
        'window_size': (input_window_size, output_window_size),
        'num_samples': num_samples
    }
    dataloader_parameters = {
        'batch_size': 1,
        'shuffle': True,
    }
    dataset = LatentDatasetTransformersCustom(
        data,
        data_pars,
        **dataset_parameters,
        multi_step=False,
    )
    dataloader = DataLoader(dataset, **dataloader_parameters)

    one_step_dataset = LatentDatasetTransformersCustom(
        data,
        data_pars,
        **dataset_parameters,
        multi_step=False,
    )
    multi_step_dataset = LatentDatasetTransformersCustom(
        data,
        data_pars,
        **dataset_parameters,
        multi_step=True,
    )
    one_step_dataloader = DataLoader(
            dataset=one_step_dataset,
            **dataloader_parameters,
    )
    multi_step_dataloader = DataLoader(
            dataset=multi_step_dataset,
            **dataloader_parameters
    )


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

    encoder.eval()
    decoder.eval()
    ##### Define prediction model #####
    prediction_model_params = {
        'latent_dim': latent_dim,
        'pars_dim': par_dim,
        'num_layers': 2,
        'embed_dim': 64,
        'num_heads': 1,
        'hidden_mlp_dim': 64,
        'out_features': latent_dim,
        'dropout_rate': 0.0,
        'device': device,
    }


    model = time_models.Transformer(**prediction_model_params).to(device)
    model.train()
    ##### Define optimizer #####
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-10
    )

    scheduler_step_size = 5
    scheduler_gamma = 0.99

    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
    )


    ##### Train model #####

    if train:
        teacher_forcing_rate = 1.
        num_epochs = 2000
        for epoch in range(num_epochs):
            pbar = tqdm(range(len(one_step_dataloader)), total=len(one_step_dataloader))
            batch_loss = 0
            for i in pbar:
                if teacher_forcing_rate > np.random.rand(1):
                    _, (in_state, out_state, pars) = next(enumerate(one_step_dataloader))
                    pars = pars.repeat(1, in_state.shape[1], 1)
                    pars = pars.reshape(-1, par_dim)
                    pars = pars.to(device)
                    in_state = in_state.reshape(-1, input_window_size, latent_dim)
                    in_state = in_state.to(device)
                    out_state = out_state.reshape(-1, input_window_size, latent_dim)
                    out_state = out_state.to(device)
                    optimizer.zero_grad()

                    model_output, _ = model(
                        x=in_state,
                        pars=pars,
                    )
                else:
                    _, (in_state, out_state, pars) = next(enumerate(multi_step_dataloader))
                    pars = pars.repeat(1, in_state.shape[1], 1)
                    pars = pars.reshape(-1, par_dim)
                    pars = pars.to(device)
                    in_state = in_state.reshape(-1, input_window_size, latent_dim)
                    in_state = in_state.to(device)
                    out_state = out_state.reshape(-1, output_window_size, latent_dim)
                    out_state = out_state.to(device)

                    optimizer.zero_grad()

                    model_output, _ = model(
                        x=in_state,
                        pars=pars,
                        num_steps=output_window_size,
                    )

                loss = nn.MSELoss()(model_output, out_state)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                optimizer.step()

                batch_loss += loss.item()

                pbar.set_postfix({
                    "loss": batch_loss/(i+1),
                    'epoch': epoch
                })

            teacher_forcing_rate = teacher_forcing_rate * 0.98

            scheduler.step()

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_weights/transformer_model_pipe_flow_new')
    else:
        checkpoint_path = 'model_weights/transformer_model_pipe_flow_new'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sample_time_ids = np.linspace(0, num_t, num_states_pr_sample,
                                  dtype=int, endpoint=False)

    encoder = encoder.to(device)
    decoder = decoder.to(device)


    transformer_state = TransformState()
    transformer_pars = TransformPars()
    dataset_params = {
        'num_files': 2000,
        'num_states_pr_sample': num_states_pr_sample,
        'sample_size': (num_t, 256),
        'pars': True,
        'with_koopman_training': with_koopman_training,
    }
    batch_size = 4
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
    }
    data_path = 'pipe_flow/data/pipe_flow'
    dataset = PipeFlowDataset(data_path, **dataset_params)
    dataloader = DataLoader(dataset, **dataloader_params)

    for i, (state, pars) in enumerate(dataloader):
        transformer_state.partial_fit(state.numpy().reshape(batch_size*num_states_pr_sample, 2, 256))
        transformer_pars.partial_fit(pars.numpy())
    error = []
    plt.figure(figsize=(12, 12))
    for j in range(0, 3):

        data_path = f'pipe_flow/data/pipe_flow_{j}.npy'
        data = np.load(data_path, allow_pickle=True)
        data = data.item()

        true_u = data['u']
        true_p = data['pressure']
        true_state = np.stack((true_u, true_p), axis=1)
        true_state = transformer_state.transform(true_state)
        pars = data['params']
        pars = np.array([[pars['friction'], pars['inflow_freq']]])
        pars = transformer_pars.transform(pars)
        pars = torch.tensor(pars, dtype=torch.get_default_dtype(), device=device)

        true_state = true_state[sample_time_ids]
        true_state = torch.tensor(true_state, dtype=torch.get_default_dtype(), device=device)

        true_latent = encoder(true_state)

        latent_init_state = true_latent[0:input_window_size].unsqueeze(0)
        latent_preds, lol = model(
            x=latent_init_state,
            pars=pars,
            num_steps=num_states_pr_sample-input_window_size,
        )
        latent_preds = torch.cat([latent_init_state, latent_preds], dim=1)
        latent_preds = latent_preds.squeeze(0)
        state_preds = decoder(latent_preds)

        latent_preds = latent_preds.detach().cpu().numpy()
        state_preds = state_preds.detach().cpu().numpy()
        true_state = true_state.detach().cpu().numpy()
        true_latent = true_latent.detach().cpu().numpy()

        plt.subplot(7, 1, 2*j-1+2)
        plt.plot(true_latent[:, 0], label='target', color='tab:blue')
        plt.plot(true_latent[:, 1], color='tab:blue')
        #plt.plot(true_latent[:, 2], color='tab:blue')
        plt.plot(latent_preds[:, 0], label='prediction', color='tab:orange')
        plt.plot(latent_preds[:, 1], color='tab:orange')
        #plt.plot(latent_preds[:, 2], color='tab:orange')
        #plt.legend()
        plt.grid()

        plt.subplot(7, 1, 2*j+2)
        plt.plot(true_state[10, 0, :], label='true', color='tab:blue')
        plt.plot(true_state[150, 0, :], color='tab:blue')
        plt.plot(true_state[-1, 0, :], color='tab:blue')
        plt.plot(state_preds[10, 0, :], label='prediction', color='tab:orange')
        plt.plot(state_preds[150, 0, :], color='tab:orange')
        plt.plot(state_preds[-1, 0, :], color='tab:orange')
        #plt.legend()
        plt.grid()

        error.append(np.linalg.norm(true_state - state_preds, axis=(1,2))/np.linalg.norm(true_state, axis=(1,2)))

    plt.subplot(7,1,7)
    for i in range(j+1):
        plt.semilogy(error[i])
    plt.grid()
    plt.show()



    print(np.mean(np.asarray(error)))
    '''
    error = 0
    for k in range(100):

        data_path = f'data/advection_diffusion/train_data/adv_diff_{k}.npy'
        data = np.load(data_path, allow_pickle=True)
        data = data.item()

        true_state = data['sol']
        pars = data['PDE_params']
        pars = np.array([[pars['velocity'], pars['diffusion']]])
        pars = pars_transformer.transform(pars)
        pars = torch.tensor(pars, dtype=torch.get_default_dtype(),
                            device=device)

        true_state = np.transpose(true_state)
        true_state = true_state[sample_time_ids]
        true_state = torch.tensor(true_state, dtype=torch.get_default_dtype(),
                                  device=device)

        true_latent = encoder(true_state)

        init_state = true_state[0:input_window_size]
        latent_init_state = encoder(init_state[:, :])
        latent_init_state = latent_init_state.unsqueeze(1)
        latent_preds = model(
                src=latent_init_state,
                tgt=latent_init_state[-1:],
                pars=pars,
                num_steps=num_states_pr_sample - input_window_size
        )
        latent_preds = torch.cat([latent_init_state, latent_preds], dim=0)
        latent_preds = latent_preds.squeeze(1)
        state_preds = decoder(latent_preds)

        latent_preds = latent_preds.detach().cpu().numpy()
        state_preds = state_preds.detach().cpu().numpy()
        true_state = true_state.detach().cpu().numpy()
        true_latent = true_latent.detach().cpu().numpy()


        error += np.linalg.norm(true_state - state_preds) / np.linalg.norm(true_state)
    print(error/(k+1))
    '''





