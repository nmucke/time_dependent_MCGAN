
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from inference.data_assimilation import DataAssimilation, PrepareData, PrepareModels, space_obs_operator

from pysindy.optimizers import STLSQ
import pysindy as ps

torch.set_default_dtype(torch.float32)
from sklearn.linear_model._base import _preprocess_data

if __name__ == '__main__':

    device = torch.device('cpu')

    ##### Prepare data #####

    num_states_pr_sample = 1024
    total_steps = 1024
    sample_time_ids = np.linspace(0, total_steps, num_states_pr_sample,
                                  dtype=int, endpoint=False)
    data = PrepareData(
            data_path='reduced_data',
            device=device,
            sample_time_ids=sample_time_ids,
            total_time_steps=total_steps
    )

    true_z_state, true_pars = data.get_data(state_case=0, par_case=0)
    true_z_state = true_z_state.detach().numpy()
    true_pars = true_pars.detach().cpu().numpy()

    X = np.concatenate([true_z_state, true_pars], axis=1)

    t_vec = np.linspace(0, 1.75, num_states_pr_sample, endpoint=False)

    differentiation_method = ps.FiniteDifference(order=2)
    feature_library = ps.PolynomialLibrary(degree=2)
    optimizer = ps.FROLS(alpha=.5)

    model = ps.SINDy(
            differentiation_method=differentiation_method,
            feature_library=feature_library,
            optimizer=optimizer,
            feature_names=["z1", "z2", "z3", "p1", "p2"]
    )
    model.fit(X, t=t_vec)
    print(model.print())

    true_z_state, true_pars = data.get_data(state_case=0, par_case=0)
    true_z_state = true_z_state.detach().numpy()
    true_pars = true_pars.detach().cpu().numpy()

    X = np.concatenate([true_z_state, true_pars], axis=1)

    z1_0 = X[0, 0]
    z2_0 = X[0, 1]
    z3_0 = X[0, 2]
    p1_0 = X[0, 3]
    p2_0 = X[0, 4]
    init = [z1_0, z2_0, z3_0, p1_0, p2_0]

    t_vec = np.linspace(0, 1.75, 1024, endpoint=False)
    sim = model.simulate(init, t=t_vec)

    plt.figure()
    plt.plot(t_vec, X[:,0], label='true')
    plt.plot(t_vec, X[:,1], label='true')
    plt.plot(t_vec, X[:,2], label='true')
    plt.plot(t_vec, sim[:,0], label='sindy')
    plt.plot(t_vec, sim[:,1], label='sindy')
    plt.plot(t_vec, sim[:,2], label='sindy')
    plt.legend()
    plt.show()
