import sys 
import os

distancepy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../distance-py/build')
sys.path.append(distancepy_path)

import copy
import mujoco
import mujoco.viewer as viewer
import numpy as np
import quaternion
import distancepy
from mujoco import rollout
from scipy.optimize import minimize, Bounds, NonlinearConstraint


class Robot:
    def __init__(
      self,
      model,
      q_init,
      pose_obj_init,
      num_particles,
      object_type='box', 
      verbose=True
    ):
        self.model = model
        self.data = mujoco.MjData(self.model)
        q0_all = np.zeros(self.model.nq) # initialize all 
        q0_all[:8] = q_init
        q0_all[8] = q_init[-1]
        q0_all[-7:] = pose_obj_init
        mujoco.mju_copy(self.data.qpos, q0_all)

        self.pose_obj_init = pose_obj_init.copy()
        initial_state = np.concatenate((self.data.qpos, self.data.qvel, self.data.act)) # data.act = current control inputs 
        self.num_particles = num_particles
        self.particles = np.tile(initial_state, (num_particles+1, 1))
        self.weights = np.ones(num_particles) / num_particles
        self.nu = self.model.nu
        self.qmin = self.model.actuator_ctrlrange[:,0]
        self.qmax = self.model.actuator_ctrlrange[:,1]
        self.bounds = Bounds(self.qmin[:7], self.qmax[:7])
        self.q_home = np.array([0, 0.7, 0, -1.57079, 0, 1.57079+0.7, 0.7853, 0.04]) # home position of the robot
        self.eps = 1e-3 # tolerance for the constraints of the inverse kinematics
        self.EE_name = 'grasp_frame'
        # self.EE_name = 'hand'
        self.object_type = object_type
        self.verbose = verbose

        self.std_pos = 1e-2
        self.std_yaw = 2e-1

    def launch_viewer(self, right_ui=False, left_ui=False):
        # initialise viewer 
        self.viewer = viewer.launch_passive(self.model, self.data, 
                                            show_right_ui=right_ui, 
                                            show_left_ui=left_ui)
        
    def set_ctrl(self, qd):
        self.data.ctrl = qd
    
    def tau_ext(self):
        return self.data.efc_J.T @ self.data.efc_force
        
    def forward(self):
        mujoco.mj_forward(self.model, self.data)
        
    def step(self):
        mujoco.mj_step(self.model, self.data)
        
    def reset_keyframe(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 1)

    def reset(self, particles=None): 
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:8] = self.q_home.copy()
        self.data.qpos[8] = self.q_home[-1].copy()
        self.data.qpos[-7:] = self.pose_obj_init.copy()
        if particles is not None: 
            self.particles = particles.copy()
    
    def set_state(self, qpos, sync_viewer=True):
        mujoco.mju_copy(self.data.qpos, qpos)
        mujoco.mj_forward(self.model, self.data)
        if sync_viewer:
            self.viewer.sync()
            
    def rollout_batch(self, controller, duration, t0=0.):
        K = int(duration / self.model.opt.timestep)
        ctrl = np.empty((K, self.nu))
        for k in range(K):
            # Compute controller and store
            ctrl[k] = controller(t0 + k*self.model.opt.timestep)
        
        initial_state = np.concatenate((np.zeros((self.num_particles+1, 1)), self.particles), axis=1)
        mj_ctrl = np.tile(ctrl, (self.num_particles+1,1, 1))
        rollout_states, _ = rollout.rollout(self.model, self.data, initial_state=initial_state, control=mj_ctrl)
        return rollout_states[:, :, 1:], ctrl
    
    def initialize_particles(self, pose_obj_init, num_particles, robot_pos, 
                             yaw_std=1., pos_std=0.06, bimodal=False):
        quat_mean = pose_obj_init[3:]
        rot_vec_mean = quaternion.as_rotation_vector(quaternion.from_float_array(quat_mean))
        # initialize particles by sampling from normal distribution around the initial object position
        particles_pos_obj = np.random.normal(pose_obj_init[:2], pos_std, (num_particles, 2))
        self.particles[:, 0:self.model.nq-7] = robot_pos.copy()
        self.particles[-1, self.model.nq-7:self.model.nq] = pose_obj_init.copy() # set one particle to actual object pose
        self.particles[:-1, self.model.nq-7:self.model.nq-5] = particles_pos_obj
        self.particles[:-1, self.model.nq-5] = np.repeat(pose_obj_init[2], num_particles)
        
        if bimodal:
            self.particles[:num_particles//2, self.model.nq-6] -= 0.2
            self.particles[num_particles//2:, self.model.nq-6] += 0.2

        self.weights = np.ones(num_particles)/num_particles
        
        # sample from normal distribution around the initial object orientation
        rot_perturb = np.random.normal(rot_vec_mean[2], yaw_std, num_particles)
        rot_vec_batch = np.zeros((num_particles, 3))
        rot_vec_batch[:,2] = rot_perturb
        quat_batch = quaternion.from_rotation_vector(rot_vec_batch)
        self.particles[:-1, self.model.nq-4:self.model.nq] = quaternion.as_float_array(quat_batch)


    def sample_particle_idx(self, weights, n_samples): 
        """
        Sample n_samples particle indices with likelihood proportional to the weights
        """
        # weights assumed to be already normalized 
        cumulative_sum = np.cumsum(weights)
        # Generate random numbers
        random_numbers = np.random.rand(n_samples)
        # Select the index where the random number falls in the cumulative sum
        indices = np.searchsorted(cumulative_sum, random_numbers)
        return indices
    
    def compute_mj_contact_belief_batch(self, particle_states): 
        """
        Compute the observations for each particle and all time steps in the rollout.
        """
        # print(particle_states[-1, :, 9:12])
        q_err_mj = particle_states[:, :, :6] - particle_states[-1, :, :6] # leave away EE joint 
        q_err_norm = np.linalg.norm(q_err_mj, axis=2)
        num_particles = particle_states.shape[0]
        contact_belief_particles = np.zeros((num_particles, q_err_mj.shape[1]))
        # particle states believed to be in contact 
        contact_belief_particles[q_err_norm > self.mj_pos_err_thresh] = 1 
        contact_belief_particles = contact_belief_particles.astype(int)
        return contact_belief_particles

    def compute_mj_contact_belief(self, particle_states, q_err_mj_arr): 
        """
        Compute the observations only for the last state of the particle states
        """
        q_err_mj = particle_states[:, -1, :4] - particle_states[-1, -1, :4] # take 4 first joints only 
        q_err_norm = np.linalg.norm(q_err_mj, axis=1)
        q_err_mj_arr.append(q_err_norm)
        contact_belief_particles = np.zeros(self.num_particles+1)
        # particle states believed to be in contact 
        contact_belief_particles[q_err_norm > self.mj_pos_err_thresh] = 1 
        contact_belief_particles = contact_belief_particles.astype(int)
        return contact_belief_particles, q_err_mj_arr

    def compute_grasp_succ_prob(self, q_grasp_traj=None, z_thresh=0.07, weights=None, 
                                gripper_val=0.038, rollout_gt=False):
        """
        Compute the probability of a successful grasp given the grasp configuration.
        Args: 
            q_grasp_traj: grasp trajectory that should be simulated (nq x 8)
            z_thresh: threshold for the final z position of the object particles defining
                      whether object is successfully grasped
            weights: particle weights
            gripper_val: value defining width of gripper when closed
            rollout_gt: whether to rollout the last particle, which is the ground truth 
        """
        initial_robot_state = np.concatenate([q_grasp_traj[0, :7], np.ones(2)*gripper_val]) #for mug gripper should be closed and not open
        data_cpy = copy.copy(self.data)
        data_cpy.qpos[:9] = initial_robot_state
        if rollout_gt:
            initial_states = self.particles.copy()
            num_particles = self.num_particles + 1
        else: 
            initial_states = self.particles[:-1].copy()
            num_particles = self.num_particles
        initial_states[:, :9] = initial_robot_state
        dq_traj = np.concatenate(
            (np.zeros((1,7)), 
             np.diff(q_grasp_traj[:,:7], axis=0)/self.model.opt.timestep), 
             axis=0
            )
        ctrl_traj_up = np.concatenate([q_grasp_traj[:,:7], dq_traj, q_grasp_traj[:, -1].reshape(-1,1)], axis=1)
        rollout_ctrl = np.tile(ctrl_traj_up, (num_particles, 1,1))
        initial_state = np.concatenate((np.zeros((num_particles, 1)), initial_states), axis=1)
        # rollout the trajectory from the grasp config to the up config
        post_grasp_states, _ = mujoco.rollout.rollout(self.model, data_cpy, 
                                                    initial_state=initial_state, 
                                                    control=rollout_ctrl)
        post_grasp_states = post_grasp_states[:, :, 1:]
        # check how many particles have z > thresh
        z_box = post_grasp_states[:, -1, self.model.nq-5]
        print('Percentage of particles with z > thresh: ', np.sum(z_box > z_thresh)/self.num_particles)
        if weights is None: 
            weights = self.weights
        succ_prob = np.sum((z_box[:len(weights)] > z_thresh) * weights) / np.sum(weights)
        return succ_prob, post_grasp_states   

    def particle_filter_binary(self, q_robot_state, contact_obs, contact_belief_particles, 
                               particle_states, weights): 
        probs_measurement = self.measurement_model_binary(contact_obs, contact_belief_particles)
        # print('probs_measurement: ', probs_measurement)
        weights *= probs_measurement
        particles_norm, weights_norm = self.normalize_and_resample_particles(q_robot_state, particle_states, weights)
        return particles_norm, weights_norm

    def measurement_model_binary(self, contact_obs, contact_belief_particles):
        """
        p(measurement=1 | particle=1) = 0.9
        p(measurement=1 | particle=0) = 0.1
        p(measurement=0 | particle=1) = 0.1
        p(measurement=0 | particle=0) = 0.9

        Args: contact_obs: binary observation (0 or 1) - scalar 
              contact_belief_particles: binary belief state (0 or 1) - array of shape (num_particles,)
        Returns: probs_measurement: array of shape (num_particles,)
        """
        p_measurement_given_particle = np.array([
            [0.9, 0.22],  # Probabilities for measurement=0
            [0.1, 0.78]   # Probabilities for measurement=1
        ])
        # print(contact_belief_particles)
        probs_measurement = p_measurement_given_particle[contact_obs, contact_belief_particles]
        return probs_measurement
    
    def measurement_model_binary_batch(self, contact_obs, contact_belief_particles):
        """
        This is used for a sequence of possible observations, which is needed in the computation of 
        the information gain over time in the vpsto cost. 
        Args: 
            contact_obs: binary observations, shape (num_particles, num_time_steps)
            contact_belief_particles: binary belief states, shape (num_particles, num_time_steps)
        Returns: probs_measurement: array of shape (num_particles, num_particles, num_time_steps)
                 Logic: first dimension is the particle that generated the measurement, 
                        second dimension is the particle serving as ground truth 
                 --> probs_measurement[i, j, t] represents the probability of observing 
                 contact_obs[i, t] given the belief state contact_belief_particles[j, t]
        """
        p_measurement_given_particle = np.array([
            [0.9, 0.7],  # Probabilities for measurement=0
            [0.1, 0.3]   # Probabilities for measurement=1
        ])
        # p_measurement_given_particle = np.array([
        #     [0.9, 0.22],  # Probabilities for measurement=0
        #     [0.1, 0.78]   # Probabilities for measurement=1
        # ])
        num_particles, num_time_steps = contact_obs.shape    
        # Expand dimensions of contact_obs and contact_belief_particles to broadcast properly
        contact_obs_expanded = np.expand_dims(contact_obs, axis=1)
        contact_belief_particles_expanded = np.expand_dims(contact_belief_particles, axis=0)
        # Use broadcasting to calculate probabilities directly
        probs_measurement = np.zeros((num_particles, num_particles, num_time_steps))
        probs_measurement[:, :, :] =\
            p_measurement_given_particle[contact_obs_expanded, contact_belief_particles_expanded]
        return probs_measurement
    
    def particle_filter_distance_based(self, q_real_arr, particle_states, weights, contact_obs, obj_dims, 
                                       full_sequence=False): 
        """
        Update the particle filter belief based on the distance of the object particles to the robot position
        This function supports either a sequence of observations, where the sequence is of length dt_pf/dt_control. 
        Or we only use the last observation at the end of the sequence. 
        """
        if full_sequence:
            if self.verbose:
                print('Using full sequence of observations')
            probs_measurement = self.measurement_model_distance_batch(particle_states, q_real_arr, obj_dims, contact_obs) # shape (num_particles, num_time_steps)
            # probs_measurment has shape (num_particles, num_time_steps)        
            weights *= np.prod(probs_measurement, axis=1)
            # print(list(np.around(weights, 20)))
            # for t in range(q_real_arr.shape[0]):
            #     weights *= probs_measurement[:, t]
                # weights /= np.sum(weights)
        else: 
            if self.verbose: 
                print('Using single observation at the end of the sequence')
            probs_measurement = self.measurement_model_distance(particle_states[:, -1], q_real_arr[-1], obj_dims, contact_obs[-1])
            weights *= probs_measurement
        # particles_norm, weights_norm = self.normalize_and_resample_particles_distance(
        #     q_real_arr[-1], particle_states[:, -1], weights)
        particles_norm, weights_norm = self.importance_resampling(
            q_real_arr[-1], particle_states[:, -1], weights, obj_dims, contact_obs[-1])
        # Make sure z position of particles is at 0.5 * obj_dims[2]
        particles_norm[:, self.model.nq-5] = 0.5*obj_dims[2]
        return particles_norm, weights_norm
    
    def measurement_model_distance_batch(self, particle_states, q_real_arr, obj_dims, contact_obs):
                                        #  alpha_c=0.3, alpha_nc=0.1, lambda_c=100):
        """
        Compute the probability of a contact observation based on the distance of each object particle 
        to the actual robot position. 
        P(o=c|x)= alpha_nc + (alpha_c-alpha_nc)*exp(-d(x,o)*lambda_c)
        lambda_c = rate of decay of the probability with distance

        Args: particle_states: array of shape (num_particles, num_time_steps, nq+nv+na)
              q_real_arr: array of shape (num_time_steps, ndof_robot)
              contact_obs: binary contact observations from robot, shape (num_time_steps,)
        Returns: p_measurement: array of shape (num_particles, num_time_steps)
        """
        num_particles, num_time_steps, _ = particle_states.shape
        object_poses = particle_states[:, :, self.model.nq-7:self.model.nq] # (num_particles, num_time_steps, 7) 
        q_robot_poses = q_real_arr[:np.min([num_time_steps, q_real_arr.shape[0]])] # (num_time_steps, ndof_robot)
        # append 2 0-vectors to robot positions for two fingers 
        q_robot_poses = np.concatenate([q_robot_poses, np.zeros((q_robot_poses.shape[0], 2))], axis=1)
        # convert object positions into a list of matrices across time steps
        object_poses_list = [object_poses[:, i, :] for i in range(num_time_steps)] # list of length num_time_steps with shape (num_particles, 7)
        # compute the distance between the robot and the object particles
        dists = distancepy.computeRobotObjectDistances(
            q_robot_poses, object_poses_list, obj_dims) # shape (num_time_steps, num_particles)
        # compute the probability of contact observation based on the distance
        p_contact_given_dist = self.alpha_nc + (self.alpha_c-self.alpha_nc)*np.exp(-dists * self.lambda_c)
        p_contact_given_dist = np.clip(p_contact_given_dist, 0.01, self.alpha_c).T # shape (num_particles, num_time_steps)
        p_measurement = np.zeros((num_particles, num_time_steps))
        p_measurement[:, contact_obs == 0] = 1 - p_contact_given_dist[:, contact_obs == 0]
        p_measurement[:, contact_obs == 1] = p_contact_given_dist[:, contact_obs == 1]
        return p_measurement   
    
    def measurement_model_distance(self, particle_states, q_real, obj_dims, contact_obs):
                                #    alpha_c=0.6, alpha_nc=0.1, lambda_c=120):
                                #    alpha_c=0.75, alpha_nc=0.1, lambda_c=200):
        """
        SAME AS ABOVE but only for single observation (contact_obs) and robot state (q_real) 
        Compute the probability of a contact observation based on the distance of each object particle 
        to the actual robot position. 
        P(o=c|x)= alpha_nc + (alpha_c-alpha_nc)*exp(-d(x,o)*lambda_c)
        lambda_c = rate of decay of the probability with distance

        Args: 
            particle_states: array of shape (num_particles, nq+nv+na)
            q_real: array of shape (ndof_robot,)
            contact_obs: binary contact observation from robot
        Returns: p_measurement: array of shape (num_particles,)
        """
        num_particles = particle_states.shape[0]
        object_poses = particle_states[:, self.model.nq-7:self.model.nq]
        q_robot_pose = np.concatenate([q_real.copy(), np.zeros(2)]) # append 2 0-vectors to robot positions for two fingers
        dists = distancepy.computeRobotObjectDistances(q_robot_pose.reshape(1, -1), [object_poses], obj_dims) # shape (1, num_particles)
        dists = dists.squeeze(axis=0)
        p_contact_given_dist = self.alpha_nc + (self.alpha_c-self.alpha_nc)*np.exp(-dists * self.lambda_c)
        p_contact_given_dist = np.clip(p_contact_given_dist, 0.01, self.alpha_c) # shape (num_particles,)
        p_measurement = np.zeros(num_particles)
        p_measurement[contact_obs == 0] = 1 - p_contact_given_dist[contact_obs == 0]
        p_measurement[contact_obs == 1] = p_contact_given_dist[contact_obs == 1]
        return p_measurement
    
    def importance_resampling(self, q_robot_state, particles, weights, obj_dims, contact_obs):
        """
        Resample 1000 new particles based on noisy importance sampling,
        compute the probability of the new samples matching the last observation,
        and replace the bad previous particles with the new best samples.
        q_robot_state: current robot state
        particles: updated particle belief
        weights: updated unnormalized particle weights
        obj_dims: dimensions of the object
        contact_obs: binary contact observation for the current robot state
        """
        sum_w = np.sum(weights)
        weights_normal = weights / sum_w # normalize particle weights
        bad_particle_flag = (weights_normal < 1e-6)
        num_outcasts = np.sum(bad_particle_flag)

        if num_outcasts >= self.num_particles-1:
            print('No PF update, as signal does not match the belief')
            return particles, weights
        
        if num_outcasts > 0:
            if self.verbose:
                print(f'sample {num_outcasts} new particles')

            std_pos_belief = np.std(particles[:,self.model.nq-7:self.model.nq-5], axis=0)
            rot_vecs = quaternion.as_rotation_vector(
                quaternion.from_float_array(
                    particles[:, self.model.nq-4:self.model.nq]))
            std_yaw_belief = np.std(rot_vecs[:,2])
            std_pos = np.minimum(std_pos_belief, self.std_pos) # (2,)
            std_yaw = np.min([std_yaw_belief, self.std_yaw])

            if contact_obs == 0:
                N_resample = num_outcasts
            else:
                N_resample = np.max([1000, num_outcasts])
            
            # Select num_outcasts particles based on the weights
            new_samples = particles[np.random.choice(np.arange(self.num_particles), N_resample, p=weights_normal)]
            # Add noise to the new samples
            new_samples[:,self.model.nq-7] += np.random.normal(0., std_pos[0], N_resample)
            new_samples[:,self.model.nq-6] += np.random.normal(0., std_pos[1], N_resample)
            rot_vecs = quaternion.as_rotation_vector(
                quaternion.from_float_array(new_samples[:,self.model.nq-4:self.model.nq]))
            rot_vecs[:,2] += np.random.normal(0., std_yaw, N_resample)
            new_samples[:,self.model.nq-4:self.model.nq] = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vecs))

            if contact_obs == 0:
                best_samples = new_samples
            else:
                # Compute how well the new samples match the last observation
                p_new_samples = self.measurement_model_distance(new_samples, q_robot_state[:7], obj_dims, contact_obs)
                # Take the best num_outcasts samples
                best_samples = new_samples[np.argsort(p_new_samples)[-num_outcasts:]]

            # Replace the bad particles with the best samples
            particles[bad_particle_flag] = best_samples
            weights_normal[bad_particle_flag] = np.mean(weights_normal[~bad_particle_flag])

        return particles, weights_normal/np.sum(weights_normal)
    
    def fk_pose(self, q=None, EE_name=None):
        if q is not None:
            data_copy = copy.copy(self.data)
            data_copy.qpos[:len(q)] = q
            mujoco.mj_fwdPosition(self.model, data_copy)
            data = data_copy
        else:
            data = self.data
        if EE_name is None:
            EE_name = self.EE_name
        
        if EE_name == 'grasp_frame':
            site = data.site(EE_name)
            x = site.xpos.copy()
            R = np.reshape(site.xmat.copy(), (3,3))
        else:# if EE_name == 'hand':
            body = data.body(EE_name)
            x = body.xpos.copy() # position of the EE (3D vector)
            R = np.reshape(body.xmat.copy(), (3,3)) # rotation matrix of the EE (3x3 matrix)
        return x, R
    
    def fk_jac(self, q=None):
        if q is not None: 
            data_copy = copy.copy(self.data)
            data_copy.qpos[:len(q)] = q
            # self.data.qpos[:7] = q[:7]
            mujoco.mj_fwdPosition(self.model, data_copy)
            data = data_copy
        else:
            data = self.data
        jac_pos = np.empty((3, self.model.nv), dtype=data.qpos.dtype)
        jac_rot = np.empty((3, self.model.nv), dtype=data.qpos.dtype)

        if self.EE_name == 'hand':
            body = data.body(self.EE_name)
            mujoco.mj_jacBody(self.model, data, jac_pos, jac_rot, body.id)
        elif self.EE_name == 'grasp_frame':
            site = data.site(self.EE_name)
            mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, site.id)
        
        jac = np.concatenate((jac_pos[:, :7], jac_rot[:, :7]), axis=0)
        return jac
    
    def inverse_kinematics(self, x_des, R_des=None, q_ref=None, EE_name=None, 
                           max_iter=100, tol=1e-6):
        if q_ref is None:
            q_ref = self.q_home
        
        def fun_and_grad(q):
            c = 0.5 * np.sum((q-q_ref)**2)
            dc = q-q_ref
            return c, dc
        
        def con_pos(q):
            x, _ = self.fk_pose(q, EE_name=EE_name)
            return x-x_des
        def con_pos_jac(q):
            jac = self.fk_jac(q)
            return jac[:3]
        nlc_pos = NonlinearConstraint(con_pos, -self.eps, self.eps, jac=con_pos_jac)

        def con_pose(q):
            x, R = self.fk_pose(q)
            e_x = x-x_des
            dR = R_des.T @ R
            # Compute dphi as the euler angle representation of dR
            dphi = np.zeros(3)
            dphi[0] = np.arctan2(dR[2,1], dR[2,2])
            dphi[1] = -np.arcsin(dR[2,0])
            dphi[2] = np.arctan2(dR[1,0], dR[0,0])
            return np.concatenate((e_x, dphi))        
        nlc_pose = NonlinearConstraint(con_pose, -self.eps, self.eps)#, jac=con_grad)

        if R_des is None:
            nlc = nlc_pos
        else:
            nlc = nlc_pose
        
        x0 = q_ref.copy()
        res = minimize(fun_and_grad, 
                       x0, 
                       method='SLSQP', 
                       jac=True,
                       bounds=self.bounds, 
                       constraints=nlc, 
                       options={'maxiter': 100, 'ftol': 1e-6, 'disp': False})
        return res.x, res.success

class TrackingController:
    def __init__(
        self,
        T,
        ctrl_ref, # [K, nu]
    ) -> None:
        self.T = T
        self.K = len(ctrl_ref)
        self.dt = self.T / self.K
        self.dt_inv = 1. / self.dt
        self.ctrl_ref = ctrl_ref
        self.m = 1.0
        self.g = 9.81
        self.k = 100
    
    def controller(self, t):
        # Get the correct reference index
        k_float = t * self.dt_inv
        k = np.floor(k_float)
        idx = int(k)
        frac = k_float - k
        
        if idx > self.K-2:
            idx = self.K-2
            frac = 1.0
        
        # Compute references
        ctrl_des = frac * self.ctrl_ref[idx+1] + (1. - frac) * self.ctrl_ref[idx]
        # Compute controller
        return ctrl_des # + np.array([0., 0., self.m * self.g / self.k, 0., 0., self.m * self.g / self.k])





    