import numpy as np 
import quaternion
from ig_vpsto.vptraj import VPTraj


class TrajectorySampler:
    def __init__(self): 
        self.N_via = 2
        self.ndof = 7 
        self.N_eval = 100
        self.init_vpsto()
        
    def init_vpsto(self):
        self.P_prior = np.zeros((self.ndof*self.N_via, self.ndof*self.N_via))
        self.R = 1
        Q1 = 1e5
        self.Q = 1e5
        self.P_prior[:self.ndof,:self.ndof] = Q1 * np.eye(self.ndof)
        vel_lim = 0.5 * np.array([0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6]) # max. rad/s for each DoF
        acc_lim = 5.0 * np.ones(self.ndof) # max. rad/s^2 for each DoF
        self.vptraj = VPTraj(ndof=self.ndof, N_eval=self.N_eval, N_via=self.N_via, vel_lim=vel_lim, acc_lim=acc_lim)

    def get_final_configs_poking(self, xd, q, system, add_via_pt_above=False):
        N_cands = xd.shape[0]
        if add_via_pt_above:
            qd = np.empty((N_cands, 14))
        else: 
            qd = np.empty((N_cands, 7))    
        for i in range(N_cands):
            if add_via_pt_above:
                x_via = xd[i].copy()
                x_via[2] += 0.1
                q_via, success = system.inverse_kinematics(x_via, q_ref=q[:7], max_iter=10)
                qT_i, success = system.inverse_kinematics(xd[i], q_ref=q_via, max_iter=10)
                if not success: 
                    print('IK failed for particle: ', i)
                qd[i] = np.concatenate([q_via, qT_i])
            else: 
                qT_i, success = system.inverse_kinematics(xd[i], q_ref=q[:7], max_iter=10)
                if not success: 
                    print('IK failed for particle: ', i)
                    qT_i, success = system.inverse_kinematics(xd[i], q_ref=system.q_home[:7])
                qd[i] = qT_i
        return qd

    def generate_trajs(self, q_start, system, xd):
        N_cands = xd.shape[0]
        # qd contains final configuration and via-point above box for every candidate (N_cands, 2*ndof)
        qd = self.get_final_configs_poking(
            xd, q_start, system, add_via_pt_above=True)
        q_trajs = np.zeros((N_cands, self.N_eval, self.ndof))
        dq_trajs = np.zeros((N_cands, self.N_eval, self.ndof))
        p_via = np.zeros((N_cands, self.ndof*self.N_via))
        Ts = np.zeros(N_cands)
        for i in range(N_cands): 
            mu_prior = np.zeros(self.ndof*self.N_via)
            mu_prior[:self.ndof] = qd[i, :self.ndof]
            mu_prior[-self.ndof:] = qd[i, self.ndof:]
            q_trajs[i], dq_trajs[i], _, p_via[i], T = \
                self.vptraj.sample_trajectories(
                    np.zeros(self.ndof*self.N_via), 
                    q_start, 
                    dq0=np.zeros_like(q_start), 
                    qT=qd[i, self.ndof:], 
                    dqT=np.zeros_like(q_start), 
                    Q=self.Q, R=self.R, 
                    mu_prior=mu_prior, P_prior=self.P_prior
                    )
            Ts[i] = T.squeeze()

        q_trajs_rescaled = []
        for i in range(N_cands): 
            # bring trajs into right resolution 
            q = q_trajs[i]
            t_horizon = np.linspace(system.model.opt.timestep, Ts[i], int(Ts[i] / system.model.opt.timestep))
            q_rescaled, _, _ = self.vptraj.get_trajectory_at_time(
                t_horizon, p_via[i], q0=q_start, dq0=np.zeros_like(q_start), 
                qT=None, dqT=np.zeros_like(q_start), T=Ts[i])
            q_trajs_rescaled.append(q_rescaled) 
        return q_trajs_rescaled
    
    def generate_random_points_in_box(self, obj_dims, z, num_points):
        x_coords = np.random.uniform(-obj_dims[0] / 2, obj_dims[0] / 2, num_points)
        y_coords = np.random.uniform(-obj_dims[1] / 2, obj_dims[1] / 2, num_points)
        z_coords = np.full(num_points, z) 
        points = np.vstack((x_coords, y_coords, z_coords)).T
        return points

    def transform_points_to_global(self, local_points, R, t):
        points_global = local_points @ R.T + t
        return points_global

    def get_final_points_from_importance_sampling(self, particles, system, 
                                                  obj_dims, N_cands,  
                                                  sample_within_object=True):
        mu_z = 0.01 
        weights = system.weights.copy()
        idx_particles = system.sample_particle_idx(weights, N_cands)
        xT_particles = particles[idx_particles, system.model.nq-7:system.model.nq] # pose of particles 
        xT_particles[:, 2] += mu_z
        if sample_within_object:
            xd = np.empty((N_cands, 3))
            R_particles = quaternion.as_rotation_matrix(
                    quaternion.from_float_array(xT_particles[:, 3:]))
            points = self.generate_random_points_in_box(obj_dims, particles[-1,2], N_cands)
            for i in range(N_cands):
                xd[i] = self.transform_points_to_global(points[i], R_particles[i], xT_particles[i, :3])
        else: 
            xd = xT_particles
        return xd

    def sample_trajs(self, system, N_cands, q0):
        xd = self.get_final_points_from_importance_sampling(
            system.particles.copy(), system, N_cands)
        trajs = self.generate_trajs(q0, system, xd)
        return trajs
