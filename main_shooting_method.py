import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import models.adv_diff_models.latent_time_gan as time_models
from inference.data_assimilation import PrepareData



def get_forecast_model(path, forecast_model_params):
    forecast_model = time_models.lstm_seq2seq(**forecast_model_params)
    checkpoint = torch.load(path)
    forecast_model.load_state_dict(checkpoint['model_state_dict'])
    forecast_model.eval()
    return forecast_model

def LSTM_time_integrator(forecast_model, init_history, pars, n_steps):
    init_history = init_history.unsqueeze(1)
    pars = pars.unsqueeze(1)
    state = forecast_model(
            input_tensor=init_history,
            target_len=n_steps,
            pars=pars
    )
    return state.squeeze(1)

def BC_residual(forecast_model, init_history, pars, n_steps, left_BC, right_BC):
    state = LSTM_time_integrator(
            forecast_model=forecast_model,
            init_history=init_history,
            pars=pars,
            n_steps=n_steps
    )
    return nn.MSELoss()(left_BC, init_history[-1]) \
           + nn.MSELoss()(right_BC, state[-1])



if __name__ == '__main__':
    latent_dim = 3

    ##### Load model #####
    forecast_model_params = {
        'input_size': latent_dim,
        'output_size': latent_dim,
        'hidden_size': 16,
        'num_layers': 2,
        'par_size': 2
    }
    checkpoint_path = 'model_weights/seq2seq_model'
    forecast_model = get_forecast_model(checkpoint_path, forecast_model_params)

    ##### Load data #####
    num_states_pr_sample = 256
    total_steps = 1024
    sample_time_ids = np.linspace(0, total_steps, num_states_pr_sample,
                                  dtype=int, endpoint=False)
    data = PrepareData(
        data_path='reduced_data',
        device='cpu',
        sample_time_ids=sample_time_ids,
        total_time_steps=total_steps
    )

    true_z_state, true_pars = data.get_data(state_case=10, par_case=10)
    _, pars = data.get_data(state_case=10, par_case=0)

    t_start, t_end = 0, 1.75
    time_vec = np.linspace(t_start, t_end, num_states_pr_sample, endpoint=False)
    dt = time_vec[1] - time_vec[0]

    num_steps = 22
    input_window = 16
    init_history = true_z_state[0:input_window]
    pars = pars[0:1]
    LSTM_state = LSTM_time_integrator(
            forecast_model=forecast_model,
            init_history=init_history,
            pars=pars,
            n_steps=num_steps
    )

    true_z_state = true_z_state[input_window:]
    true_pars = true_pars[input_window:]

    right_BC_error = BC_residual(
            forecast_model=forecast_model,
            init_history=init_history,
            pars=pars,
            n_steps=num_steps,
            right_BC=true_z_state[num_steps-1],
            left_BC=true_z_state[0]
    )
    print(right_BC_error)

    init_history = init_history.detach().requires_grad_(True)
    pars = pars.detach().requires_grad_(True)
    optimizer = optim.Adam([init_history, pars], lr=.1)
    '''
    optimizer = torch.optim.LBFGS(
            [init_history, pars],
            history_size=5,
            max_iter=5,
            line_search_fn="strong_wolfe",
    )
    '''
    for i in range(200):
        optimizer.zero_grad()
        def closure():
            optimizer.zero_grad()
            loss = BC_residual(
                    forecast_model=forecast_model,
                    init_history=init_history,
                    pars=pars,
                    n_steps=num_steps,
                right_BC=true_z_state[num_steps-1],
                left_BC=true_z_state[0]
            )
            loss.backward()
            if i % 10 == 0:
                print(i, loss.item(), pars)
            return loss
        #loss.backward()
        optimizer.step(closure)


    LSTM_state_corrected = LSTM_time_integrator(
            forecast_model=forecast_model,
            init_history=init_history,
            pars=pars,
            n_steps=num_steps
    )


    LSTM_state = LSTM_state.detach().numpy()
    LSTM_state_corrected = LSTM_state_corrected.detach().numpy()
    true_z_state = true_z_state.detach().numpy()
    true_pars = true_pars.detach().numpy()
    time_vec = time_vec[input_window:]

    plt.figure()
    plt.plot(time_vec[:num_steps], true_z_state[:num_steps, 0], label='True state', color='tab:blue')
    plt.plot(time_vec[:num_steps], true_z_state[:num_steps, 1], color='tab:blue')
    plt.plot(time_vec[:num_steps], true_z_state[:num_steps, 2], color='tab:blue')
    plt.plot(time_vec[:num_steps], LSTM_state[:, 0], label='LSTM state', color='tab:red')
    plt.plot(time_vec[:num_steps], LSTM_state[:, 1], color='tab:red')
    plt.plot(time_vec[:num_steps], LSTM_state[:, 2], color='tab:red')
    plt.plot(time_vec[:num_steps], LSTM_state_corrected[:, 0], label='LSTM state corrected', color='tab:green')
    plt.plot(time_vec[:num_steps], LSTM_state_corrected[:, 1], color='tab:green')
    plt.plot(time_vec[:num_steps], LSTM_state_corrected[:, 2], color='tab:green')
    plt.legend()
    plt.show()


