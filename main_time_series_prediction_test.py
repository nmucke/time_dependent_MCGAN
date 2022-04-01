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

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)

output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 64
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_inout_sequences(input_data, memory):
    inout_features = []
    inout_targets = []
    L = len(input_data)
    for i in range(L - memory):
        train_seq = input_data[i:i + memory]
        train_label = input_data[i + output_window:i + memory + output_window]
        inout_features.append(train_seq)
        inout_targets.append(train_label)
    return torch.FloatTensor(inout_features), torch.FloatTensor(inout_targets)



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

def get_data(memory=100, data_source='lorenz'):
    if data_source == 'sine_cosine':
        time = np.linspace(0, 10, 2000)
        feature_1 = np.sin(time)
        feature_2 = np.cos(time)
        data = np.stack((feature_1, feature_2), axis=1)

    elif data_source == 'mackey_glass':
        N = 2000

        b = 0.1
        c = 0.2
        tau = 17

        y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
             1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

        for n in range(17, N + 99):
            y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))
        data = y[100:]
        data = np.expand_dims(np.asarray(data), axis=1)

    elif data_source == 'lorenz':
        data = lorenz_data()

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    num_samples = data.shape[0]
    num_train = int(0.75*num_samples)
    num_test = num_samples - num_train

    train_data = data[0:num_train]
    test_data = data[-num_test:]

    #train_data += np.random.normal(0, 0.05, size=train_data.shape)

    train_features, train_targets = create_inout_sequences(
        input_data=train_data,
        memory=memory
    )

    #shuffled_ids = np.arange(0,7400)
    #np.random.shuffle(shuffled_ids)
    #train_features = train_features[shuffled_ids]
    #train_targets = train_targets[shuffled_ids]

    train_features = np.swapaxes(train_features, 1, 2)
    train_targets = np.swapaxes(train_targets, 1, 2)

    test_features, test_targets = create_inout_sequences(
        input_data=test_data,
        memory=memory
    )
    test_features = np.swapaxes(test_features, 1, 2)
    test_targets = np.swapaxes(test_targets, 1, 2)

    train_data = {'features': train_features,
                  'targets': train_targets}
    test_data = {'features': test_features,
                  'targets': test_targets}

    return train_data, test_data

def get_batch(source_features, source_targets, i, batch_size):
    input = source_features[i:i+batch_size]
    input = torch.swapaxes(input, 1, 2)
    input = torch.swapaxes(input, 0, 1)

    target = source_targets[i:i+batch_size]
    target = torch.swapaxes(target, 1, 2)
    target = torch.swapaxes(target, 0, 1)

    return input, target




def train(train_data):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    train_features = train_data['features'].to(device)
    train_targets = train_data['targets'].to(device)
    for batch, i in enumerate(range(0, len(train_features) - 1, batch_size)):
        pdb.set_trace()
        data, targets = get_batch(train_features,train_targets, i, batch_size)
        pdb.set_trace()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.detach().item()
    return total_loss/batch

def evaluate(eval_model, data_source):

    eval_model.eval()

    data_features = data_source['features'].to(device)
    data_targets = data_source['targets'].to(device)

    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_features) - 1, batch_size)):

            data, targets = get_batch(data_features, data_targets, i, batch_size)
            output = eval_model(data)
            total_loss += criterion(output, targets).cpu().detach().item()

    return total_loss/i


def multistep_pred(model, data, num_steps=2000):

    pred = model(data[:,0:1,:].to(device))
    pred_list = [pred]
    pred_list_out = [pred[-1,0].cpu().detach().numpy()]
    for i in range(num_steps-1):
        pred = model(pred_list[-1])
        pred_list.append(pred)
        pred_list_out.append(pred[-1,0].cpu().detach().numpy())
    return np.asarray(pred_list_out)

if __name__ == '__main__':


    data_source = 'lorenz'
    batch_size = 100
    train_data, val_data = get_data(memory=100, data_source=data_source)
    model = TransAm(num_channels=train_data['features'].shape[-1]).to(device)

    criterion = nn.MSELoss()
    lr = 0.005
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 300  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(train_data)

        val_loss = evaluate(model, val_data)
        print(f'{epoch}, {val_loss}')

        scheduler.step()

    num_steps = 500

    model.eval()

    #preds = preds[-1, :].cpu().detach()

    plt.figure(figsize=(18,15))
    features, targets = get_batch(val_data['features'], val_data['targets']
                                  , 0, len(val_data['features']))
    preds = model(features.to(device))
    preds = preds[-1, :].cpu().detach()
    preds = preds[0:num_steps]

    targets = targets[-1, :].cpu().detach()
    targets = targets[0:num_steps]


    if data_source == 'lorenz':
        plt.subplot(2,3,1)
        plt.plot(preds[:,0], label='pred x(t)')
        plt.plot(targets[:,0], label='True x(t)')
        plt.legend()

        plt.subplot(2,3,2)
        plt.plot(preds[:,1], label='pred y(t)')
        plt.plot(targets[:,1], label='True y(t)')
        plt.legend()

        plt.subplot(2,3,3)
        plt.plot(preds[:,2], label='pred z(t)')
        plt.plot(targets[:,2], label='True z(t)')
        plt.legend()

        features, targets = get_batch(val_data['features'], val_data['targets']
                                      , 0, len(val_data['features']))
        preds = multistep_pred(model, features, num_steps=num_steps)

        targets = targets[-1, :].cpu().detach()
        targets = targets[0:num_steps]

        plt.subplot(2,3,4)
        plt.plot(preds[:,0], label='pred x(t)')
        plt.plot(targets[:,0], label='True x(t)')
        plt.legend()

        plt.subplot(2,3,5)
        plt.plot(preds[:,1], label='pred y(t)')
        plt.plot(targets[:,1], label='True y(t)')
        plt.legend()

        plt.subplot(2,3,6)
        plt.plot(preds[:,2], label='pred z(t)')
        plt.plot(targets[:,2], label='True z(t)')
        plt.legend()
        plt.show()
    else:
        plt.subplot(2, 1, 1)
        plt.plot(preds[:, 0], label='pred x(t)')
        plt.plot(targets[:, 0], label='True x(t)')
        plt.legend()

        features, targets = get_batch(val_data['features'], val_data['targets']
                                      , 0, len(val_data['features']))
        preds = multistep_pred(model, features, num_steps=num_steps)

        targets = targets[-1, :].cpu().detach()
        targets = targets[0:num_steps]

        plt.subplot(2, 1, 2)
        plt.plot(preds[:, 0], label='pred x(t)')
        plt.plot(targets[:, 0], label='True x(t)')
        plt.legend()
        plt.show()
