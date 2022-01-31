import numpy as np
import matplotlib.pyplot as plt
import pdb

import DG_routines
import time_integrators


class DG_solver(DG_routines.DG_1D):
    def __init__(self,xmin=0,xmax=1,K=10,N=5,
                 integrator='BDF2', num_states=1,
                 **stabilizer):

        super(DG_solver,self).__init__(xmin=xmin,xmax=xmax,K=K,N=N)

        stabilizer = stabilizer['stabilizer']
        if stabilizer['stabilizer_type'] == 'filter':
            self.Filter1D(N=self.N,Nc=stabilizer['Nc'],s=stabilizer['s'])
            self.stabilizer = self.apply_filter
        elif stabilizer['stabilizer_type'] == 'slope_limit':
            self.stabilizer = self.apply_slopelimitN
        else:
            self.stabilizer = self.identity

        self.integrator = integrator
        if integrator == 'SSPRK':
            self.integrator_func = time_integrators.SSPRK(stabilizer=self.stabilizer,
                                                          num_states=num_states)
        elif integrator == 'LowStorageRK':
            self.integrator_func = time_integrators.LowStorageRK(stabilizer=self.stabilizer,
                                                                 num_states=num_states)
        elif integrator == 'BDF2':
            self.integrator_func = time_integrators.BDF2(stabilizer=self.stabilizer,
                                                         num_states=num_states)
        elif integrator == 'ImplicitEuler':
            self.integrator_func = time_integrators.ImplicitEuler(stabilizer=self.stabilizer,
                                                         num_states=num_states)


    def solve_pde(self,q_init,t_end,step_size,rhs):
        """Solve PDE from given initial condition"""
        q_sol = [q_init]
        t_vec = [0]

        t = 0

        if self.integrator == 'BDF2':
            q_new, t_new = self.integrator_func.initial_step(time=0,
                                                      q_init=q_init,
                                                      rhs=rhs,
                                                      step_size=step_size)
            q_sol.append(q_new)
            t_vec.append(t_new)

        while t < t_end:
            q_new, t_new = self.integrator_func.update_state(q_sol=q_sol,
                                                             t_vec=t_vec,
                                                             step_size=step_size,
                                                             rhs=rhs)

            t = t_new

            q_sol.append(q_new)
            t_vec.append(t_new)

        return q_sol, t_vec











