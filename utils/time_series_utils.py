import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch


def multistep_pred(model, data, num_steps=2000, device='cpu'):
    pred = model(data[:,0:1,:].to(device))
    pred_list = [pred]
    pred_list_out = [pred[-1,0].cpu().detach().numpy()]
    for i in range(num_steps-1):
        pred = model(pred_list[-1])
        pred_list.append(pred)
        pred_list_out.append(pred[-1,0].cpu().detach().numpy())
    return np.asarray(pred_list_out)
