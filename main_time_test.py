import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pdb
from models.adv_diff_models.transformer import TransAm
from scipy.integrate import solve_ivp
import data_handling.data_handling_utils as utils
from torch.utils.data import DataLoader
from training.train_transformer import TrainTransformer
from utils.time_series_utils import multistep_pred
from models.adv_diff_models.causal_convolutional_NN import CCNN

torch.manual_seed(0)
np.random.seed(0)


def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def lorenz_data(sigma=10, beta=2.667, rho=28):
    # Lorenz paramters and initial conditions.

    u0, v0, w0 = 0, 1, 1.05

    # Maximum time point and total number of time points.
    tmax, n = 100, 10000

    # Integrate the Lorenz equations.
    soln = solve_ivp(lambda t,x: lorenz(t, x, sigma, beta, rho), (0, tmax), (u0, v0, w0),
                     dense_output=True)
    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, tmax, n)
    x, y, z = soln.sol(t)

    return np.stack((x,y,z), axis=1)

if __name__ == '__main__':

    cuda = True
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    data_source = 'lorenz'

    num_channels = 3
    hidden_channels = [num_channels for i in range(7)]
    kernel_size = 2

    memory = 2 ** (len(hidden_channels) - 1) * kernel_size
    print(memory)

    #train_data, val_data = get_data(memory=100, data_source=data_source)
    data = []
    num_train_data = 1
    rho_list = np.linspace(28, 28, num_train_data)
    for i in range(num_train_data):
        data.append(lorenz_data(rho=rho_list[i]))
    data = np.asarray(data)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(num_train_data*10000,3))
    data = data.reshape(num_train_data, 10000, 3)

    dataset = utils.TimeSeriesDataset(data=data, memory=memory)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    #model = TransAm(num_channels=data.shape[-1]).to(device)


    model = CCNN(
        num_channels=num_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)

    trainer = TrainTransformer(
        transformer=model,
        transformer_optimizer=optimizer,
        scheduler=scheduler,
        save_string='transformer',
        n_epochs=150,
        memory=memory,
        device=device)
    trainer.train(dataloader=dataloader,
                  temporal_batch_size=10)

    model.inference = True

    num_steps = 500
    test_case1 = 0
    test_case2 = 0

    plt.figure()
    features, targets = utils.get_batch(dataset[test_case1][0][9000:].unsqueeze(0),
                                        dataset[test_case1][1][9000:].unsqueeze(0),
                                        time_step=0,
                                        batch_size=1)
    preds = multistep_pred(model, features, num_steps=num_steps, device=device)


    targs = []
    for i in range(num_steps):
        features, targets = utils.get_batch(dataset[test_case1][0][9000:].unsqueeze(0),
                                            dataset[test_case1][1][9000:].unsqueeze(0),
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

    features, targets = utils.get_batch(dataset[test_case2][0][9000:].unsqueeze(0),
                                        dataset[test_case2][1][9000:].unsqueeze(0),
                                        time_step=0,
                                        batch_size=1)
    preds = multistep_pred(model, features, num_steps=num_steps, device=device)

    targs = []
    for i in range(num_steps):
        features, targets = utils.get_batch(dataset[test_case2][0][9000:].unsqueeze(0),
                                            dataset[test_case2][1][9000:].unsqueeze(0),
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
        features, targets = utils.get_batch(dataset[test_case1][0][9000:].unsqueeze(0),
                                            dataset[test_case1][1][9000:].unsqueeze(0),
                                            time_step=i,
                                            batch_size=1)
        features = features.to(device)
        targets = targets.to(device)
        pred = model(features)
        preds.append(pred[-1,0].cpu().detach().numpy())
        targs.append(targets[-1,0].cpu().detach().numpy())
    preds = np.asarray(preds)
    targs = np.asarray(targs)

    plt.figure(figsize=(18,15))
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
        features, targets = utils.get_batch(dataset[test_case2][0][9000:].unsqueeze(0),
                                            dataset[test_case2][1][9000:].unsqueeze(0),
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
