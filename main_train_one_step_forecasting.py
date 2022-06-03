import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pdb
from models.adv_diff_models.pre_train_time_gan import ForecastingNet
from scipy.integrate import solve_ivp
import data_handling.data_handling_utils as utils
from torch.utils.data import DataLoader
from training.train_forecasting_net import TrainForecastingNet
from utils.time_series_utils import multistep_pred
from data_handling.conditional_gan_adv_diff_dataloader import AdvDiffDataset

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


if __name__ == '__main__':

    train = False

    cuda = False
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'



    data_path = 'data/advection_diffusion/train_data/adv_diff'

    dataset_params = {
        'data_path': data_path,
        'num_files': 10000,
        'num_states_pr_sample': 128,
        'sample_size': (128, 512),
        'window_size': (32 ,1),
        'transformer_state': None,
        'transformer_pars': None,
    }
    dataloader_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 8,
        'drop_last': True,
    }

    dataset = AdvDiffDataset(**dataset_params)
    dataloader = DataLoader(dataset, **dataloader_params)

    model = ForecastingNet(
            latent_dim=16,
            in_channels=1,
            hidden_channels=2,
            bilinear=False
    ).to(device)

    if train:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-10)

        trainer = TrainForecastingNet(
            model=model,
            model_optimizer=optimizer,
            scheduler=None,
            save_string='forecasting_net',
            n_epochs=100,
            device=device
        )
        trainer.train(dataloader=dataloader,
                      temporal_batch_size=32)
    else:
        checkpoint = torch.load('model_weights/forecasting_net')
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    num_steps = 100
    test_case = 100

    pred_list = []
    true = dataset[test_case][1].cpu().detach().numpy()

    init = dataset[test_case][0][0:1].unsqueeze(0).to(device)
    for i in range(num_steps-5):
        #init = dataset[test_case][0][i:i+1].unsqueeze(0).to(device)
        pred = model(init, return_input=True)
        init = pred[:, : , :, -32:]
        pred_list.append(pred[0, 0, :, -1:])

    pred_list = torch.stack(pred_list).cpu().detach().numpy()
    plt.figure()
    plt.plot(true[10, :, 0], label='true', color='blue')
    plt.plot(true[50, :, 0], label='true', color='blue')
    plt.plot(true[90, :, 0], label='true', color='blue')
    plt.plot(pred_list[10, :, 0], label='pred', color='red')
    plt.plot(pred_list[50, :, 0], label='pred', color='red')
    plt.plot(pred_list[90, :, 0], label='pred', color='red')
    plt.show()
    pdb.set_trace()
    


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
