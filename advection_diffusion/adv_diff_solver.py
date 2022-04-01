import pdb

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.special import lpmv


class AdvectionDiffusion():
    def __init__(self, xmin=0, xmax=1, num_x=50, params=None):

        self.num_x = num_x
        self.x_vec = np.linspace(xmin,xmax,num_x)
        self.dx = np.abs(self.x_vec[1]-self.x_vec[0])

        self.D_xx = -2*np.eye(num_x-2) + np.eye(num_x-2, k=-1) + np.eye(num_x-2, k=1)
        self.D_xx /= self.dx*self.dx

        self.D_x = np.eye(num_x-2, k=1) - np.eye(num_x-2, k=-1)
        self.D_x /= 2*self.dx

        self.params = params


        self.BCs = lambda t: self.BC_fun(t, params)


        self.left_bc = lambda t: 0
        self.right_bc = lambda t: 0



    def BC_fun(self, t, params):


        bcs = np.zeros(self.num_x - 2)
        bcs[0] = (params['diffusion'] * 1 / self.dx / self.dx + params[
            'velocity'] * 1 / 2 / self.dx) * self.left_bc(t)
        bcs[-1] = (params['diffusion'] * 1 / self.dx / self.dx - params[
            'velocity'] * 1 / 2 / self.dx) * self.right_bc(t)
        return bcs

    def f_func(self, t, x, c, k):
        func = 0*2*np.sin(2*np.pi*t)*np.cos(2*np.pi*x)
        return func

    def RHS(self, t, u, BCs, f, params):
        rhs = np.dot(params['diffusion'] * self.D_xx \
                     - params['velocity'] * self.D_x,u) \
              + f(t) + BCs(t)
        return rhs

    def solve_adv_diff(self, init, t_eval):
        f = lambda t: self.f_func(t=t,x=self.x_vec,
                                  c=self.params['velocity'],
                                  k=self.params['diffusion'])[1:-1]

        sol = solve_ivp(lambda t, u: self.RHS(t, u, self.BCs, f, self.params),
                        t_span=[0, t_eval[-1]],
                        y0=init, t_eval=t_eval, method='RK45')
        t = sol.t
        sol = sol.y

        left_bc_vec = []
        right_bc_vec = []
        for t_i in t_eval:
            left_bc_vec.append(self.left_bc(t_i))
            right_bc_vec.append(self.right_bc(t_i))
        left_bc_vec = np.asarray(left_bc_vec).reshape((1, len(t_eval)))
        right_bc_vec = np.asarray(right_bc_vec).reshape((1, len(t_eval)))

        sol = np.concatenate((left_bc_vec, sol, right_bc_vec))

        return t, sol





