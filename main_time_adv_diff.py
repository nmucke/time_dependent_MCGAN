import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pdb
from models.adv_diff_models.transformer import TransAm
from models.adv_diff_models.causal_convolutional_NN import CCNN
from scipy.integrate import solve_ivp
import data_handling.data_handling_utils as utils
from torch.utils.data import DataLoader
from training.train_transformer import TrainTransformer
from utils.time_series_utils import multistep_pred

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_tensor_type(torch.FloatTensor)


if __name__ == '__main__':

    cuda = False
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    '''
    data_path_state = 'data/advection_diffusion/train_data/adv_diff'
    for i in range(100000):
        pars = np.load(f"{data_path_state}_{i}.npy", allow_pickle=True)
        pars = pars.item()
        pars = pars['PDE_params']
        pars = np.array([pars['velocity'], pars['diffusion']])
    '''

    data = np.load('reduced_data.npy')
    data = data[0:1024]
    num_train_data = data.shape[0]
    num_steps = data.shape[1]
    num_channels = data.shape[2]

    num_channels = num_channels
    hidden_channels = [num_channels//2 for i in range(6)]
    kernel_size = 2

    memory = 2 ** (len(hidden_channels) - 1) * kernel_size
    print(memory)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(num_train_data*num_steps, num_channels))
    data = data.reshape(num_train_data, num_steps, num_channels)

    dataset = utils.TimeSeriesDataset(data=data, memory=memory)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)

    '''
    model = TransAm(
        num_channels=num_channels,
        feature_size=64,
        num_layers=1,
        dropout=0.1,
        nhead=2,
        dense_neurons=64,
        num_pars=None
    ).to(device)
    '''

    model = CCNN(
        num_channels=num_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    trainer = TrainTransformer(
        transformer=model,
        transformer_optimizer=optimizer,
        scheduler=scheduler,
        save_string='transformer',
        n_epochs=150,
        memory=memory,
        device=device)
    trainer.train(dataloader=dataloader,
                  temporal_batch_size=32)
    model.inference = True

    num_steps = 100
    test_case1 = 2
    test_case2 = 3

    plt.figure()
    features, targets = utils.get_batch(dataset[test_case1][0].unsqueeze(0),
                                        dataset[test_case1][1].unsqueeze(0),
                                        time_step=0,
                                        batch_size=1)
    preds = multistep_pred(model, features, num_steps=num_steps, device=device)

    targs = []
    for i in range(num_steps):
        features, targets = utils.get_batch(dataset[test_case1][0].unsqueeze(0),
                                            dataset[test_case1][1].unsqueeze(0),
                                            time_step=i,
                                            batch_size=1)
        targs.append(targets[-1,0].cpu().detach().numpy())
    targets = np.asarray(targs)


    plt.subplot(2,2,1)
    plt.plot(preds[:,0], label='pred x(t)')
    plt.plot(targets[:,0], label='True x(t)')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(preds[:,1], label='pred y(t)')
    plt.plot(targets[:,1], label='True y(t)')
    plt.legend()

    features, targets = utils.get_batch(dataset[test_case2][0].unsqueeze(0),
                                        dataset[test_case2][1].unsqueeze(0),
                                        time_step=0,
                                        batch_size=1)
    preds = multistep_pred(model, features, num_steps=num_steps, device=device)

    targs = []
    for i in range(num_steps):
        features, targets = utils.get_batch(dataset[test_case2][0].unsqueeze(0),
                                            dataset[test_case2][1].unsqueeze(0),
                                            time_step=i,
                                            batch_size=1)
        targs.append(targets[-1,0].cpu().detach().numpy())
    targets = np.asarray(targs)


    plt.subplot(2,2,3)
    plt.plot(preds[:,0], label='pred x(t)')
    plt.plot(targets[:,0], label='True x(t)')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(preds[:,1], label='pred y(t)')
    plt.plot(targets[:,1], label='True y(t)')
    plt.legend()
    plt.show()


    preds = []
    targs = []
    for i in range(num_steps):
        features, targets = utils.get_batch(dataset[test_case1][0].unsqueeze(0),
                                            dataset[test_case1][1].unsqueeze(0),
                                            time_step=i,
                                            batch_size=1)
        features = features.to(device)
        targets = targets.to(device)
        pred = model(features)
        preds.append(pred[-1,0].cpu().detach().numpy())
        targs.append(targets[-1,0].cpu().detach().numpy())
    preds = np.asarray(preds)
    targs = np.asarray(targs)

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(preds[:,0], label='pred x(t)')
    plt.plot(targs[:,0], label='True x(t)')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(preds[:,1], label='pred y(t)')
    plt.plot(targs[:,1], label='True y(t)')
    plt.legend()


    preds = []
    targs = []
    for i in range(num_steps):
        features, targets = utils.get_batch(dataset[test_case2][0].unsqueeze(0),
                                            dataset[test_case2][1].unsqueeze(0),
                                            time_step=i,
                                            batch_size=1)
        features = features.to(device)
        targets = targets.to(device)
        pred = model(features)
        preds.append(pred[-1, 0].cpu().detach().numpy())
        targs.append(targets[-1, 0].cpu().detach().numpy())
    preds = np.asarray(preds)
    targs = np.asarray(targs)

    plt.subplot(2, 2, 3)
    plt.plot(preds[:, 0], label='pred x(t)')
    plt.plot(targs[:, 0], label='True x(t)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(preds[:, 1], label='pred y(t)')
    plt.plot(targs[:, 1], label='True y(t)')
    plt.legend()
    plt.show()
