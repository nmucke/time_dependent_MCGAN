import numpy as np
import pdb
import DG_solver
import DG_routines
import matplotlib.pyplot as plt

class GasPipeflow(DG_solver.DG_solver):
    def __init__(self, xmin=0, xmax=1, K=10, N=5,
                 integrator='BDF2', num_states=2,
                 params=None,
                 pressure_func=lambda rho,velocity,rhoref,pref:rho,
                 **stabilizer,
                 ):

        super(GasPipeflow, self).__init__(xmin=xmin, xmax=xmax, K=K, N=N,
                                             integrator=integrator,
                                             num_states=num_states,
                                             stabilizer=stabilizer
                                             )
        self.nx = self.nx.flatten('F')

        self.velocity = params['velocity']
        self.pamb = params['pamb']
        self.p0 = params['p0']
        self.pref = params['pref']
        self.rhoref = params['rhoref']
        self.rho0 = params['rho0']
        self.diameter = params['diameter']
        self.A = np.pi * (self.diameter / 2) ** 2
        self.mu = params['mu']
        self.pressure_func = pressure_func
        self.g = 9.8
        self.Cd = params['Cd']
        self.A_orifice = params['A_orifice']

        self.inlet_condition = params['inlet_condition']
        self.outlet_condition = params['outlet_condition']
        self.inlet_noise_var = params['inlet_noise_var']
        self.outlet_noise_var = params['outlet_noise_var']
        self.inlet_noise = np.random.normal(loc=0,
                                            scale=self.inlet_noise_var)
        self.outlet_noise = np.random.normal(loc=0,
                                             scale=self.outlet_noise_var)



        self.pipe_roughness = np.reshape(params['pipe_roughness'](self.x),
                                         (self.Np*self.K),'F')
        self.leak_times = params['leak_times']

        self.Cv = params['Cv']
        self.xl = params['xl']


        self.x_y_coordinates = params['x_y_coordinates']
        self.num_sections = self.x_y_coordinates.shape[0]-1
        self.piecewise_dx = [self.x_y_coordinates[i+1,0]-self.x_y_coordinates[i,0]
                             for i in range(self.num_sections)]
        self.piecewise_dy = [self.x_y_coordinates[i+1,1]-self.x_y_coordinates[i,1]
                             for i in range(self.num_sections)]
        self.length_of_sections = []
        for i in range(self.num_sections):
            len_x = self.x_y_coordinates[i+1,0]-self.x_y_coordinates[i,0]
            len_y = self.x_y_coordinates[i+1,1]-self.x_y_coordinates[i,1]
            len_section = np.sqrt(len_x**2+len_y**2)
            self.length_of_sections.append(len_section)
        self.s_bend_points = [np.sum(self.length_of_sections[0:i])
                              for i in range(0,self.num_sections)]
        self.s_bend_points.append(self.x[-1,-1])
        self.piecewise_slopes = [dy / (dx+1e-13) for (dx, dy) in
                                 zip(self.piecewise_dx, self.piecewise_dy)]

        self.y_of_s = np.zeros(self.x.shape)
        for i in range(self.num_sections):
            x_ids = np.where((self.x > self.s_bend_points[i]) &
                                 (self.x <= self.s_bend_points[i+1]))
            if self.piecewise_slopes[i] > 1e8:
                y_min = self.x_y_coordinates[i,1]
                y_max = self.x_y_coordinates[i+1,1]
                self.y_of_s[x_ids] = (self.x[x_ids] - np.min(self.x[x_ids]))*(y_max-y_min)/\
                                     (np.max(self.x[x_ids]) - np.min(self.x[x_ids])) + y_min
            else:
                self.y_of_s[x_ids] = self.x[x_ids]*self.piecewise_slopes[i]\
                                     - self.x_y_coordinates[i,0]*self.piecewise_slopes[i]\
                                     + self.x_y_coordinates[i,1]

        self.phi_func = self.rx * np.dot(self.Dr, self.y_of_s)
        self.phi_func = self.phi_func.flatten('F')

        self.xElementL = [np.int(xl / self.xmax * self.K) for xl in self.xl]

        self.lagrange = []
        for xl, xElementL in zip(self.xl,self.xElementL):
            l = np.zeros(self.N + 1)
            rl = 2 * (xl - self.VX[xElementL]) / self.deltax - 1
            for i in range(0, self.N + 1):
                l[i] = DG_routines.JacobiP(np.array([rl]), 0, 0, i)
            self.lagrange.append(np.linalg.solve(np.transpose(self.V), l))

    def BoundaryConditions(self, time, q1, q2):
        """Set boundary conditions"""

        q1in = q1[self.vmapI]
        out_pres = self.outlet_condition(time) + self.outlet_noise
        q1out = (out_pres-self.pref)/self.velocity/self.velocity \
                + self.rhoref
        q1out = self.A*q1out

        q2in = (self.inlet_condition(time)+self.inlet_noise) * q1in
        q2out = q2[self.vmapO]

        return q1in, q1out, q2in, q2out



    def Leakage(self, time, pressure=0, rho=0, post=False):
        """Compute leakage"""

        f_l = np.zeros((self.x.shape))

        for xl,xElementL,leak_times,Cv,l in zip(self.xl,self.xElementL,
                                                self.leak_times,self.Cv,
                                                self.lagrange):
            if time >= leak_times[0] and time < leak_times[1]:
                pressureL = np.reshape(pressure, (self.Np, self.K), 'F')
                pressureL = self.EvaluateSol(np.array([xl]), pressureL)[0]
                rhoL = np.reshape(rho, (self.Np, self.K), 'F')
                rhoL = self.EvaluateSol(np.array([xl]), rhoL)[0]

                delta_P = pressureL - self.pamb
                discharge_sqrt_coef = self.A**2 * 2*delta_P
                discharge_sqrt_coef /= (self.A/(self.A_orifice*self.Cd))**2-1
                discharge_sqrt_coef = np.sqrt(discharge_sqrt_coef/rhoL)
                f_l[:, xElementL] = discharge_sqrt_coef * rhoL * l
                f_l[:, xElementL] = np.dot(self.invMk,f_l[:,xElementL])

        if not post:
            return f_l

        elif post:
            for xElementL, leak_times in zip(self.xElementL,self.leak_times):
                if time >= leak_times[0] and time < leak_times[1]:
                    #f_l[:, xElementL] = np.dot(self.Mk, f_l[:, xElementL])/rhoL
                    #f_l[:, xElementL] = f_l[:, xElementL]
                    return discharge_sqrt_coef
                else:
                    return 0



    def Friction(self, rho, rhou, u):
        """Compute friction term"""

        Red = self.diameter * np.abs(rhou) / self.mu
        friction_term = (self.pipe_roughness/self.diameter/3.7)**1.11 + 6.9/Red
        friction_term = -1.8*np.log10(friction_term)
        friction_term = friction_term*friction_term
        friction_term = 0.25/friction_term

        friction_term = 1/2*self.diameter*np.pi*friction_term*rho*u*u

        return np.reshape(friction_term, (self.Np, self.K), 'F')

    def inclined_term(self, rho):

        inclined = self.g * self.A * rho * self.phi_func

        return np.reshape(inclined, (self.Np, self.K), 'F')

    def q1_flux(self,time,q1,q2):
        return q2

    def q2_flux(self,time,q1,q2,pressure):
        return q2*q2/q1 + self.A*pressure


    def Lax_Friedrichs_flux(self,q,flux,LFc):

        dq = q[self.vmapM] - q[self.vmapP]

        dqFlux = flux[self.vmapM] - flux[self.vmapP]
        dqFlux = self.nx*dqFlux/2. - LFc/2.*dq

        return dqFlux

    def LF_boundary(self, lm, q, qFlux, qBC, qFluxqBC):

        lmIn = np.abs(lm[self.vmapI]) / 2
        nxIn = self.nx[self.mapI]

        lmOut = np.abs(lm[self.vmapO]) / 2
        nxOut = self.nx[self.mapO]

        dq_inflow = nxIn * (qFlux[self.vmapI] - qFluxqBC[0]) / 2 \
                     - lmIn * (q[self.vmapI] - qBC[0])

        dq_outflow = nxOut * (qFlux[self.vmapO] - qFluxqBC[1]) / 2 \
                     - lmOut * (q[self.vmapO] - qBC[1])

        return dq_inflow, dq_outflow


    def rhs(self, time, q):
        """Compute right hand side of PDE"""

        q1 = q[0:int(self.Np * self.K)]
        q2 = q[-int(self.Np * self.K):]

        # Compute velocity
        u = q2/q1
        rho = q1/self.A
        rhou = q2/self.A

        # Compute pressure
        pressure = self.pressure_func(rho=rho,velocity=self.velocity,
                                      rhoref=self.rhoref,pref=self.pref)

        # Compute eigenvalue
        lm = np.abs(u) + self.velocity/np.sqrt(self.A)

        # Compute LFc
        LFc = np.maximum((lm[self.vmapM]), (lm[self.vmapP]))
        # q1 flux
        q1Flux = self.q1_flux(time,q1,q2)
        dq1Flux = self.Lax_Friedrichs_flux(q1, q1Flux, LFc)

        # q2 flux
        q2Flux = self.q2_flux(time,q1,q2,pressure)
        dq2Flux = self.Lax_Friedrichs_flux(q2, q2Flux, LFc)


        ### Boundary conditions ###
        q1in, q1out, q2in, q2out = self.BoundaryConditions(time, q1, q2)

        pressure_in = self.pressure_func(q1in/self.A,self.velocity,
                                         rhoref=self.rhoref,pref=self.pref)
        pressure_out = self.pressure_func(q1out/self.A,self.velocity,
                                          rhoref=self.rhoref,pref=self.pref)

        # Inflow flux
        q1FluxIn = self.q1_flux(time,q1in,q2in)
        q2FluxIn = self.q2_flux(time,q1in,q2in,pressure_in)

        # Outflow flux
        q1FluxOut = self.q1_flux(time,q1out,q2out)
        q2FluxOut = self.q2_flux(time,q1out,q2out,pressure_out)

        # q1 boundary Lax-Friedrichs flux
        dq1Flux[self.mapI], dq1Flux[self.mapO] = self.LF_boundary(lm=lm,
                                                  q=q1,
                                                  qFlux=q1Flux,
                                                  qBC=[q1in, q1out],
                                                  qFluxqBC=[q1FluxIn,q1FluxOut])
        # q2 boundary Lax-Friedrichs flux
        dq2Flux[self.mapI], dq2Flux[self.mapO] = self.LF_boundary(lm=lm,
                                                  q=q2,
                                                  qFlux=q2Flux,
                                                  qBC=[q2in,q2out],
                                                  qFluxqBC=[q2FluxIn,q2FluxOut])

        # Reshape flux vectors
        q1Flux = np.reshape(q1Flux, (self.Np, self.K), 'F')
        q2Flux = np.reshape(q2Flux, (self.Np, self.K), 'F')

        dq1Flux = np.reshape(dq1Flux, ((self.Nfp * self.Nfaces, self.K)), 'F')
        dq2Flux = np.reshape(dq2Flux, ((self.Nfp * self.Nfaces, self.K)), 'F')

        # Compute leakage and friction
        leakage_term = self.Leakage(time, pressure=pressure, rho=rho)
        friction_term = self.Friction(rho=rho, rhou=rhou, u=u)
        inclined_term = self.inclined_term(rho=rho)

        # Compute RHS
        rhsq1 = - self.rx * np.dot(self.Dr, q1Flux) \
                + np.dot(self.LIFT, self.Fscale * dq1Flux) \
                - leakage_term
        rhsq2 = - self.rx * np.dot(self.Dr, q2Flux) \
                + np.dot(self.LIFT, self.Fscale * dq2Flux) \
                - friction_term - inclined_term

        rhsq1 = rhsq1.flatten('F')
        rhsq2 = rhsq2.flatten('F')


        return np.concatenate((rhsq1, rhsq2))

    def compute_jacobian(self, time, U, state_len, rhs):
        """Compute Jacobian of right hand side"""

        epsilon = np.finfo(float).eps

        J = np.zeros((state_len, state_len))

        F = rhs(time, U)
        for col in range(state_len):
            pert = np.zeros(state_len)
            pert_jac = np.sqrt(epsilon) * np.maximum(np.abs(U[col]), 1)
            pert[col] = pert_jac

            Upert = U + pert

            Fpert = rhs(time, Upert)

            J[:, col] = (Fpert - F) / pert_jac

        return J

    def newton_solver(self, time, q, rhs):
        """Solve steady state with Newton solver"""

        self.state_len = q.shape[0]

        self.J = self.compute_jacobian(time, q, self.state_len, rhs)
        LHS = self.J

        newton_error = 1e2
        iterations = 0
        q_old = q
        while newton_error > 1e-6 and iterations < 100:
            RHS = -rhs(time, q_old)

            delta_q = np.linalg.solve(LHS, RHS)
            q_old = q_old + delta_q

            newton_error = np.max(np.abs(delta_q))
            iterations = iterations + 1

        if iterations > 100:
            print(f'Steady state Newton did not converge. Newton error: {newton_error}')
        return q_old

    def solve(self, q_init, t_end, step_size, print_progress=True):
        """Solve PDE from given initial condition"""
        q_init = self.stabilizer(q_init,num_states=2)
        q_init = self.newton_solver(0, q_init, self.rhs)
        q_init = self.stabilizer(q_init,num_states=2)
        #p = self.pressure_func(q_init[0:(self.Np*self.K)]/self.A,
        #                       velocity=self.velocity,
        #                       rhoref=self.rhoref,
        #                       pref=self.pref)

        q_sol = [q_init]
        t_vec = [0]

        t = 0

        if self.integrator == 'BDF2':
            q_new, t_new = self.integrator_func.initial_step(time=0,
                                                             q_init=q_init,
                                                             rhs=self.rhs,
                                                             step_size=step_size)
            q_sol.append(q_new)
            t_vec.append(t_new)

            t = t_new

        idx = 0
        while t < t_end:
            idx += 1
            self.inlet_noise = np.random.normal(loc=self.inlet_noise,
                                                scale=self.inlet_noise_var)
            self.outlet_noise = np.random.normal(loc=self.outlet_noise,
                                                scale=self.outlet_noise_var)

            if self.integrator == 'LowStorageRK' or \
               self.integrator == 'SSPRK':

                C = self.velocity/np.sqrt(self.A) + \
                    np.abs(np.max(q_sol[-1][-int(self.Np * self.K):] /
                                   q_sol[-1][0:int(self.Np * self.K)]))
                CFL = 0.5
                step_size = CFL * self.dx / C

                q_new, t_new = self.integrator_func.update_state(q_sol=q_sol,
                                                                 t_vec=t_vec,
                                                                 step_size=step_size,
                                                                 rhs=self.rhs)
                if print_progress:
                    if idx % 1000 == 0:
                        print(f'{t / t_end * 100:.1f}% Done')

            else:
                q_new, t_new = self.integrator_func.update_state(q_sol=q_sol,
                                                                 t_vec=t_vec,
                                                                 step_size=step_size,
                                                                 rhs=self.rhs)
                if print_progress:
                    if idx % 10 == 0:
                        print(f'{t / t_end * 100:.1f}% Done')

            t = t_new

            q_sol.append(q_new)
            t_vec.append(t_new)

        return q_sol, t_vec







