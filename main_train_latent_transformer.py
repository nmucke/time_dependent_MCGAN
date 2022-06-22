import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_handling.latent_dataloader import LatentDatasetTransformers
import models.adv_diff_models.adversarial_AE as models
from utils.seed_everything import seed_everything
import torch.nn as nn
from utils.load_checkpoint import load_checkpoint
from torch.utils.data import DataLoader
import models.adv_diff_models.transformer as time_models
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

    input_window_size = 32
    output_window_size = 128
    latent_dim = 4
    par_dim = 2
    num_states_pr_sample = 256
    num_t = 1024
    num_x = 128

    ##### Load data #####
    data = np.load('reduced_data.npy')
    data = torch.tensor(data, dtype=torch.get_default_dtype())

    data_pars = np.load('reduced_data_pars.npy')
    data_pars = data_pars.reshape(-1, 2)
    pars_transformer = StandardScaler()
    pars_transformer.fit(data_pars)
    data_pars = pars_transformer.transform(data_pars)
    data_pars = data_pars.reshape(10000, 1024, 2)
    data_pars = torch.tensor(data_pars, dtype=torch.get_default_dtype())

    dataset_parameters = {
        'num_states_pr_sample': 256,
        'sample_size': (latent_dim, 1024),
        'window_size': (input_window_size, output_window_size),
    }
    dataloader_parameters = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 4
    }

    dataset = LatentDatasetTransformers(
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

    checkpoint_path = 'model_weights/AE_koopman_adversarial'
    checkpoint = torch.load(checkpoint_path)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    ##### Define prediction model #####
    prediction_model_params = {
        'latent_dim': latent_dim,
        'pars_dim': par_dim,
        'embed_dim': 8,
        'dropout_pos_enc': 0.1,
        'max_seq_len': input_window_size,
        'out_seq_len': output_window_size,
        'n_heads': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
    }


    model = time_models.Transformer(**prediction_model_params).to(device)
    model.train()
    ##### Define optimizer #####
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-10
    )


    ##### Train model #####

    if train:
        num_epochs = 2000
        for epoch in range(num_epochs):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            batch_loss = 0
            for i, (src, tgt, tgt_y, pars) in pbar:
                pars = pars.repeat(1, tgt.size(1), 1)
                pars = pars.reshape(-1, par_dim)
                pars = pars.to(device)
                src = src.reshape(-1, input_window_size, latent_dim)
                src = src.transpose(0, 1)
                src = src.to(device)
                tgt = tgt.reshape(-1, output_window_size, latent_dim)
                tgt = tgt.transpose(0, 1)
                tgt = tgt.to(device)
                tgt_y = tgt_y.reshape(-1, output_window_size, latent_dim)
                tgt_y = tgt_y.transpose(0, 1)
                tgt_y = tgt_y.to(device)

                optimizer.zero_grad()

                model_output = model(
                    src=src,
                    tgt=tgt,
                    pars=pars,
                )

                loss = nn.MSELoss()(model_output, tgt_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                optimizer.step()

                batch_loss += loss.item()

                pbar.set_postfix({
                    "loss": batch_loss/(i+1e-12),
                    'epoch': epoch
                })


            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_weights/transformer_model')
    else:
        checkpoint_path = 'model_weights/transformer_model'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sample_time_ids = np.linspace(0, num_t, num_states_pr_sample,
                                  dtype=int, endpoint=False)
    error = 0
    plt.figure()
    for j in range(3):

        data_path = f'data/advection_diffusion/train_data/adv_diff_{j}.npy'
        data = np.load(data_path, allow_pickle=True)
        data = data.item()

        true_state = data['sol']
        pars = data['PDE_params']
        pars = np.array([[pars['velocity'], pars['diffusion']]])
        pars = pars_transformer.transform(pars)
        pars = torch.tensor(pars, dtype=torch.get_default_dtype(), device=device)

        true_state = np.transpose(true_state)
        true_state = true_state[sample_time_ids]
        true_state = torch.tensor(true_state, dtype=torch.get_default_dtype(), device=device)

        true_latent = encoder(true_state)

        init_state = true_state[0:input_window_size]
        latent_init_state = encoder(init_state[:, :])
        latent_init_state = latent_init_state.unsqueeze(1)
        latent_preds = model(
            src=latent_init_state,
            tgt=latent_init_state[-1:],
            pars=pars,
            num_steps=num_states_pr_sample-input_window_size
        )
        latent_preds = torch.cat([latent_init_state, latent_preds], dim=0)
        latent_preds = latent_preds.squeeze(1)
        state_preds = decoder(latent_preds)

        latent_preds = latent_preds.detach().cpu().numpy()
        state_preds = state_preds.detach().cpu().numpy()
        true_state = true_state.detach().cpu().numpy()
        true_latent = true_latent.detach().cpu().numpy()

        plt.subplot(3, 2, 2*j-1+2)
        plt.plot(true_latent[:, 0], label='target', color='tab:blue')
        plt.plot(true_latent[:, 1], color='tab:blue')
        plt.plot(true_latent[:, 2], color='tab:blue')
        plt.plot(latent_preds[:, 0], label='prediction', color='tab:orange')
        plt.plot(latent_preds[:, 1], color='tab:orange')
        plt.plot(latent_preds[:, 2], color='tab:orange')
        #plt.legend()
        plt.grid()

        plt.subplot(3, 2, 2*j+2)
        plt.plot(true_state[10, :], label='true', color='tab:blue')
        plt.plot(true_state[150, :], color='tab:blue')
        plt.plot(true_state[-1, :], color='tab:blue')
        plt.plot(state_preds[10, :], label='prediction', color='tab:orange')
        plt.plot(state_preds[150, :], color='tab:orange')
        plt.plot(state_preds[-1, :], color='tab:orange')
        #plt.legend()
        plt.grid()

        error += np.linalg.norm(true_state - state_preds)/np.linalg.norm(true_state)
    print(error/3)
    plt.show()


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





