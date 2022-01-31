import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib.animation as animation
import DG_routines
import DG_solver
import DG_models
import time as timing
import multiprocessing
import time
from multiprocessing.pool import ThreadPool
import concurrent.futures
import ray
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd


def animateSolution(x,time,sol_list,gif_name='pipe_flow_simulation',
                    xlabel='Location',ylabel='Pressure'):
    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(np.min(sol_list),np.max(sol_list)))
    #ax = plt.axes(xlim=(x[0], x[-1]), ylim=(-1,1))

    #ax = plt.axes(xlim=(x[0], x[-1]), ylim=(np.min(sol_list),1003))

    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        plt.title(f'{time[i]:0.2f} seconds')
        y = sol_list[i]
        line.set_data(x, y)
        return line,

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writergif = animation.PillowWriter(fps=10)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(sol_list), interval=20, blit=True)

    # save the animation as mp4 video file
    anim.save(gif_name + '.gif',writer=writergif)

def pressure_func(rho, velocity, rhoref=0., pref=0.):
    """Compute the pressure"""
    return velocity * velocity * (rho - rhoref) + pref

def get_uniform_solutions(x_uniform, t_uniform, q, gas_model):
    num_t_uniform_points = len(t_uniform)
    num_x_uniform_points = len(x_uniform)
    Np, K = gas_model.Np, gas_model.K

    rho = np.zeros((num_t_uniform_points, num_x_uniform_points))
    u = np.zeros((num_t_uniform_points, num_x_uniform_points))
    pressure = np.zeros((num_t_uniform_points, num_x_uniform_points))

    for i, q_i in enumerate(t_uniform):
        rhoA = q[q_i, 0:(Np * K)]

        rhoA = gas_model.EvaluateSol(x_uniform,
                                     np.reshape(rhoA, (Np, K), 'F'))

        rhouA = q[q_i, -(Np * K):]
        rhouA = gas_model.EvaluateSol(x_uniform,
                                      np.reshape(rhouA, (Np, K), 'F'))

        u[i, :] = rhouA / rhoA
        rho[i, :] = rhoA / gas_model.A
        pressure[i, :] = gas_model.pressure_func(rho[i, :],
                                                 velocity=gas_model.velocity,
                                                 rhoref=gas_model.rhoref,
                                                 pref=gas_model.pref)

    return rho, u, pressure

#@ray.remote
def run(xl,Cd,leak_start):
    diameter = 0.193675
    velocity = 1227.
    inflow = 4.
    outPressure = 3.5e5
    pamb = 1e5
    pref = 1e5
    p0 = 3.5e5
    rhoref = 1000
    rho0 = (p0 - pref) / velocity / velocity + rhoref
    pipe_roughness = lambda x: 2.8e-5*np.ones(x.shape)
    mu = 1e-3
    A = np.pi * (diameter / 2) ** 2
    x_y_coordinates = np.array([[0, 0],
                                [620, 0],
                                [967, 3.32],
                                [967, 45],
                                [1000, 45]])
    D_orifice = 0.03
    A_orifice = np.pi*(D_orifice/2)**2
    Cv = A/np.sqrt(rhoref/2 * ((A/(A_orifice*Cd))**2-1))
    Cv = [Cv]
    xl = [xl]
    leak_times = [[leak_start, 100*60]]

    inlet_condition = lambda t: inflow + 0.025*inflow*np.sin(t/1.8*2*np.pi)
    outlet_condition = lambda t: outPressure + 0.025*outPressure*np.sin(t/1.2*2*np.pi)

    inlet_noise_var = 0.1
    outlet_noise_var = 1e1

    params = {'velocity': velocity,
              'inlet_condition': inlet_condition,
              'outlet_condition': outlet_condition,
              'inlet_noise_var': inlet_noise_var,
              'outlet_noise_var': outlet_noise_var,
              'pamb': pamb,
              'pref': pref,
              'p0': p0,
              'rho0': rho0,
              'rhoref': rhoref,
              'diameter': diameter,
              'Cv': Cv,
              'Cd': Cd,
              'A_orifice': A_orifice,
              'xl': xl,
              'pipe_roughness': pipe_roughness,
              'mu': mu,
              'leak_times': leak_times,
              'x_y_coordinates': x_y_coordinates}

    smin = 0
    smax = 1041.6958820572913

    K = 75
    N = 3

    integrator = 'ImplicitEuler'
    gas_model = DG_models.GasPipeflow(xmin=smin, xmax=smax, K=K, N=N,
                                      integrator=integrator,
                                      params=params,
                                      pressure_func=pressure_func,
                                      stabilizer_type='slope_limit', Nc=1, s=2,
                                      )

    final_time = 15.

    x_vec = np.reshape(gas_model.x, (N + 1) * K, 'F')

    rhoAinit = A * rho0 * np.ones(x_vec.shape)
    rhouAinit = rhoAinit * inflow

    qinit = np.concatenate((rhoAinit, rhouAinit))

    q, time = gas_model.solve(qinit,
                              t_end=final_time,
                              step_size=.025,
                              print_progress=True)
    q = np.asarray(q)

    leak = []
    for i in range(len(time)):
        rho = q[i,0:(K*(N+1))]/A
        pres = pressure_func(rho, velocity, rhoref=rhoref, pref=pref)
        l = gas_model.Leakage(time=time[i], pressure=pres, rho=rho, post=True)
        leak.append(l*60*60)

    num_x_uniform_points = 256
    x_uniform = np.linspace(0, smax, num_x_uniform_points)

    if integrator == 'ImplicitEuler' or integrator == 'BDF2':
        timestep_skip = 1
    else:
        timestep_skip = 250
    t_uniform = np.arange(0,len(time),timestep_skip)

    rho, u, pressure = get_uniform_solutions(x_uniform, t_uniform, q, gas_model)

    return u, rho, pressure, x_uniform, np.asarray(time)[t_uniform]




if __name__ == '__main__':

    u, rho, pressure, x_uniform, time = run(800, .84, 5)
    pressure *= 1e-5

    animateSolution(x_uniform, time, u, gif_name='velocity',
                    xlabel='Location', ylabel='velocity')
    animateSolution(x_uniform, time, pressure, gif_name='pressure',
                    xlabel='Location', ylabel='pressure')