import numpy as np
from scipy.special import gamma
import scipy.special as sci
import scipy.sparse as sps
import pdb
import DG_routines
import matplotlib.pyplot as plt
from scipy.optimize import line_search
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

class BDF2():
    def __init__(self, stabilizer, num_states, max_newton_iter=50, newton_tol=1e-5):

        self.stabilizer = stabilizer
        self.num_states = num_states

        self.time = 0
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol

        self.alpha = np.array([1, -4/3, 1/3])
        self.beta = 2/3

        self.idx = 0

    def compute_jacobian(self,time,U,state_len,rhs):

        epsilon = np.finfo(float).eps

        J = np.zeros((state_len,state_len))

        F = rhs(time,U)
        for col in range(state_len):
            pert = np.zeros(state_len)
            pert_jac = np.sqrt(epsilon) * np.maximum(np.abs(U[col]), 1)
            pert[col] = pert_jac

            Upert = U + pert

            Fpert = rhs(time,Upert)

            J[:,col] = (Fpert - F) / pert_jac

        return J

    def initial_step(self, time, q_init, rhs, step_size):

        self.state_len =  q_init.shape[0]

        self.J = self.compute_jacobian(time,q_init,self.state_len,rhs)
        LHS = 1/step_size*np.eye(self.state_len) - self.J

        newton_error = 1e8
        iterations = 0
        q_old = q_init
        while newton_error > self.newton_tol and \
                iterations < self.max_newton_iter:
            RHS = -(1/step_size*(q_old - q_init) - rhs(time,q_old))

            delta_q = np.linalg.solve(LHS,RHS)

            q_old = q_old + delta_q

            newton_error = np.max(np.abs(delta_q))
            iterations = iterations + 1

        return self.stabilizer(q_old,self.num_states), time+step_size

    def update_state(self, q_sol, t_vec, step_size, rhs):

        if self.idx%25 == 0:
            self.J = self.compute_jacobian(t_vec[-1],q_sol[-1],self.state_len,rhs)

        LHS = 1 / step_size * np.eye(self.state_len) - self.beta*self.J

        newton_error = 1e8
        iterations = 0
        q_old = q_sol[-1]
        while newton_error > self.newton_tol and iterations < self.max_newton_iter:
            RHS = -(1 / step_size * (self.alpha[0] * q_old +
                                     self.alpha[1] * q_sol[-1] +
                                     self.alpha[2] * q_sol[-2]) -
                    self.beta * rhs(t_vec[-1], q_old))

            delta_q = np.linalg.solve(LHS, RHS)

            q_old = q_old + delta_q

            newton_error = np.max(np.abs(delta_q))
            iterations = iterations + 1

        self.idx += 1
        return self.stabilizer(q_old,self.num_states), t_vec[-1]+step_size

class ImplicitEuler():
    def __init__(self, stabilizer, num_states, max_newton_iter=200, newton_tol=1e-5):

        self.stabilizer = stabilizer
        self.num_states = num_states

        self.time = 0
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol

        self.alpha = np.array([1, -4/3, 1/3])
        self.beta = 2/3

        self.idx = 0

    def compute_jacobian(self,time,U,state_len,rhs):

        epsilon = np.finfo(float).eps

        J = np.zeros((state_len,state_len))

        F = rhs(time,U)
        for col in range(state_len):
            pert = np.zeros(state_len)
            pert_jac = np.sqrt(epsilon) * np.maximum(np.abs(U[col]), 1)
            pert[col] = pert_jac

            Upert = U + pert

            Fpert = rhs(time,Upert)

            J[:,col] = (Fpert - F) / pert_jac

        return J

    def linesearch(self, time, q_old, q_init, delta_q, step_size,
                   LHS, rhs, RHS, epsilon, sigma):

        func = lambda alpha: 0.5*np.dot(rhs(time,q_old + alpha*delta_q),
                                        rhs(time, q_old + alpha*delta_q))

        #func_jac = lambda alpha: np.dot(np.transpose(LHS),rhs(time,q_old+alpha*delta_q))

        #func = lambda alpha: rhs(time, q_old + alpha * delta_q)

        options = {'maxiter': 200}
        #alpha = line_search(func, func_jac, q_old, delta_q,maxiter=200)
        alpha = minimize_scalar(func,method='bounded',bounds=(0.1,2),
                                options=options)
        #alpha = minimize(func,x0=np.array([1]),jac=func_jac,options=options,tol=1e-2)
        print(alpha.x)

        return alpha.x


    '''
    RHS_delta = rhs(time,q_old + delta_q)
    FtF_delta = 0.5*np.dot(RHS_delta,RHS_delta)

    FtF = 0.5*np.dot(RHS,RHS)

    alpha = 1
    k = 0
    while np.any(FtF_delta <= FtF + epsilon*alpha*np.dot(np.dot(np.transpose(LHS),RHS),delta_q)) and\
            k < self.max_newton_iter:

        alpha = 1/sigma * alpha

        RHS_delta = rhs(time, q_old + alpha*delta_q)
        FtF_delta = 0.5 * np.dot(RHS_delta, RHS_delta)

        k = k + 1
        #pdb.set_trace()
    print(k)
    return alpha
    '''


    '''

    h_delta = -(1 / step_size * (q_old - q_init) - rhs(time, q_old + delta_q))
    delta_q_new = np.linalg.solve(LHS, RHS)

    h_dash = np.dot(delta_q_new, delta_q)

    h_hat = RHS + delta_q * epsilon * h_dash

    k = 0
    while k < self.max_newton_iter and np.any(h_delta > h_hat + 1e-7):

        delta_q = 1 / sigma * delta_q

        h_delta = -(1 / step_size * (q_old - q_init) - rhs(time, q_old + delta_q))
        RHS_new = -(1 / step_size * (q_old - q_init) - rhs(time, q_old))
        delta_q_new = np.linalg.solve(LHS, RHS_new)

        h_dash = np.dot(delta_q_new, delta_q)
        h_hat = RHS + epsilon * delta_q * h_dash


        k = k + 1
    '''

    def update_state(self, q_sol, t_vec, step_size, rhs):

        self.state_len =  q_sol[-1].shape[0]

        time=t_vec[-1]
        q_init = q_sol[-1]

        if self.idx%2 == 0:
            self.J = self.compute_jacobian(time,q_init,self.state_len,rhs)


        lam = 0.1
        LHS = 1/step_size*np.eye(self.state_len) - self.J

        newton_error = 1e2
        iterations = 0

        epsilon = 1e-4
        sigma = 2

        q_old = q_init
        while newton_error > self.newton_tol and \
                iterations < self.max_newton_iter:

            RHS = -(1/step_size*(q_old - q_init) - rhs(time,q_old))

            delta_q = np.linalg.solve(LHS,RHS)
            #print(np.linalg.det(LHS))

            #alpha = self.linesearch(time, q_old, q_init, delta_q, step_size,
            #                          LHS, rhs, RHS, epsilon, sigma)
            alpha = 1.
            q_old = q_old + alpha*delta_q

            newton_error = np.max(np.abs(delta_q))
            iterations = iterations + 1

            #if iterations%10 == 0:
            #    print(f'iterations: {iterations}, error: {newton_error}')

        self.idx += 1

        return self.stabilizer(q_old,self.num_states), time+step_size


class LowStorageRK():
    def __init__(self, stabilizer,num_states):

        self.num_states = num_states
        self.time = 0
        self.stabilizer = stabilizer

        self.rk4a = np.array([0.0, -567301805773.0 / 1357537059087.0,
                              -2404267990393.0 / 2016746695238.0,
                              -3550918686646.0 / 2091501179385.0,
                              -1275806237668.0 / 842570457699.0])

        self.rk4b = np.array([1432997174477.0 / 9575080441755.0,
                              5161836677717.0 / 13612068292357.0,
                              1720146321549.0 / 2090206949498.0,
                              3134564353537.0 / 4481467310338.0,
                              2277821191437.0 / 14882151754819.0])

        self.rk4c = np.array([0.0, 1432997174477.0 / 9575080441755.0,
                              2526269341429.0 / 6820363962896.0,
                              2006345519317.0 / 3224310063776.0,
                              2802321613138.0 / 2924317926251.0])

    def update_state(self, q_sol, t_vec, step_size, rhs):

        q_new = q_sol[-1]

        resq = np.zeros(q_new.shape)

        for INTRK in range(0, 5):

            rhsq = rhs(t_vec[-1],q_new)

            resq = self.rk4a[INTRK] * resq + step_size * rhsq

            q_new = q_new + self.rk4b[INTRK] * resq

            q_new = self.stabilizer(q_new,self.num_states)

        return q_new, t_vec[-1] + step_size

class SSPRK():
    def __init__(self, stabilizer,num_states):

        self.num_states = num_states
        self.time = 0
        self.stabilizer = stabilizer

        self.a = np.array([[1, 0, 0],
                               [3/4, 1/4, 0],
                               [1/3, 0, 2/3]])
        self.b = np.array([1, 1/4, 2/3])
        self.c = np.array([0, 1, 1/2])

    def update_state(self, q_sol, t_vec, step_size, rhs):

        q_new = self.a[0,0]*q_sol[-1] + \
             self.b[0]*step_size*rhs(t_vec[-1] + self.c[0]*step_size,q_sol[-1])
        q_new = self.stabilizer(q_new,self.num_states)

        q_new = self.a[1,0]*q_sol[-1] + self.a[1,1]*q_new + \
            self.b[1]*step_size*rhs(t_vec[-1] + self.c[1]*step_size,q_new)
        q_new = self.stabilizer(q_new,self.num_states)

        q_new = self.a[2,0]*q_sol[-1] + self.a[2,2]*q_new + \
            self.b[2]*step_size*rhs(t_vec[-1] + self.c[2]*step_size,q_new)
        q_new = self.stabilizer(q_new,self.num_states)

        return q_new, t_vec[-1] + step_size
