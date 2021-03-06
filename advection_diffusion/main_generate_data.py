import pdb

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import ray
#from scipy.special import lpmv
import scipy.special
from adv_diff_solver import AdvectionDiffusion
import os

def init_condition(coef_vec, x_vec):
    init = 0
    for i in range(1, len(coef_vec) + 1):
        y = scipy.special.lpmv(1, i, x_vec)
        init += coef_vec[i - 1] * y
    init = init / np.max(np.abs(init))
    return init


def init_condition_gaussian(coef_vec, x_vec):
    mu = coef_vec[0]
    sigma = coef_vec[1]
    return np.exp(-np.power(x_vec - mu, 2.) / (2 * np.power(sigma, 2.)))


def compute_advection_diffusion_sol(solver_params,
                                    PDE_params,
                                    output_time=False):

    '''
    coef_mean = np.zeros(solver_params['init_num_coefs'])
    coef_std = np.arange(solver_params['init_num_coefs'],0,-1)

    coef_vec = np.random.normal(coef_mean, coef_std)
    init = init_condition(coef_vec, x_vec)
    init = init[1:-1]


    '''
    mu = -0.5
    sigma = np.random.normal(0.10, 0.05)

    #coef_vec = np.array([mu, sigma])
    coef_vec = np.array([mu, sigma])
    init = init_condition_gaussian(coef_vec, x_vec)
    init = init[1:-1]

    adv_diff = AdvectionDiffusion(solver_params['xmin'],
                                          solver_params['xmax'],
                                          solver_params['num_x'],
                                          params=PDE_params)

    t, sol = adv_diff.solve_adv_diff(init, solver_params['t_eval'])

    if output_time:
        return t, sol
    else:
        return sol
@ray.remote
def generate_and_save_data(save_string,
                           save_id,
                           solver_params,
                           PDE_params,
                           train_data=True):
    '''

    @ray.remote
    def generate_and_save_data(save_id):
    '''

    t, sol = compute_advection_diffusion_sol(solver_params=solver_params,
                                          PDE_params=PDE_params,
                                          output_time=True)


    save_dict = {'sol': sol,
                 't_eval': t,
                 'PDE_params': PDE_params}

    if train_data:
        dir_save_string = f'../data/advection_diffusion/train_data/{save_string}_{save_id}'
    else:
        dir_save_string = f'../data/advection_diffusion/test_data/{save_string}_{save_id}'

    np.save(dir_save_string, save_dict)

    print(save_id)

if __name__ == '__main__':

    #lol = np.load('../data/advection_diffusion/adv_diff_1.npy', allow_pickle=True)

    solver_params = {'xmin': -1,
                     'xmax': 1,
                     'num_x': 128,
                     't_eval': np.linspace(0,1.75,512),
                     'init_num_coefs': 3}

    x_vec = np.linspace(solver_params['xmin'],
                        solver_params['xmax'],
                        solver_params['num_x'])
    save_string = 'adv_diff'
    id_list = range(2000, 10000)
    train_data = True

    ray.init(num_cpus=30)

    velocity = np.random.normal(0.45, 0.05, len(id_list))
    diffusion = np.random.normal(0.007, 0.001, len(id_list))
    for idx in id_list:
        PDE_params = {'velocity': velocity[idx],
                      'diffusion': diffusion[idx]}
        generate_and_save_data.remote(save_string=save_string,
                                     save_id=idx,
                                     solver_params=solver_params,
                                     PDE_params=PDE_params,
                                     train_data=train_data)
    '''

    PDE_params = {'velocity': 0.58,  # velocity[idx],
                  'diffusion': 0.003}  # diffusion[idx]}
    t, sol = compute_advection_diffusion_sol(solver_params=solver_params,
                                             PDE_params=PDE_params,
                                             output_time=True)

    X,T = np.meshgrid(x_vec,solver_params['t_eval'])

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_vec,sol[:, 0])
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.pcolor(X,T,np.transpose(sol))
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.plot(x_vec,sol[:,40])
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(x_vec,sol[:,-1])
    plt.grid()

    plt.show()

    '''

