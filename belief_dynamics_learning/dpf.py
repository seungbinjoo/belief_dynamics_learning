import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils
import matplotlib.pyplot as plt
from belief_dynamics_learning.belief_plotting_2d import *
from belief_dynamics_learning.world2d import *
from belief_dynamics_learning.dpf_utils import *

class DPF():

    def __init__(self, propose_ratio, proposer_keep_ratio, min_obs_likelihood, num_particles, world: World2D, xy_only=False):
        """
        Apply differentiable particle filter (DPF) to current belief of the object pose (represented by particle set)
        to predict the belief at the next time step.
        Args:
            propose_ratio: ratio of proposed to resampled particles
            proposer_keep_ratio: dropout ratio for particle proposer network, used during training and inference to introduce randomness to particle proposer network
            min_obs_likelihood: lower limit for observation likelihood produced by the observation likelihood estimate
            world: object of class World2D
        """
        # store hyperparameters
        self.propose_ratio = propose_ratio
        self.proposer_keep_ratio = proposer_keep_ratio
        self.min_obs_likelihood = min_obs_likelihood
        self.world = world
        
        # define more parameters
        if xy_only == False:
            self.state_dim = 3 # --> x, y, cos theta, sin theta for the object pose
        else:
            self.state_dim = 2
        self.num_particles_float = num_particles
        self.num_particles = int(num_particles)

        # build learnable networks: observation likelihood estimator, particle proposer
        self.build_networks(xy_only)

    def build_networks(self, xy_only):
        
        # device configuration --> use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        print('Using device:', self.device)

        # PARTICLE PROPOSER --> maps observations and robot poses to particles
        self.particle_proposer = ParticleProposer(self.proposer_keep_ratio, xy_only).to(self.device)

        # OBSERVATION LIKELIHOOD ESTIMATOR --> maps observations and robot poses to probabilities
        self.obs_like_estimator = ObsLikelihoodEstimator(self.min_obs_likelihood).to(self.device)

    # compute observation likelihood (probabilities) for a particle set
    def measurement_update(self, q_r_desired, q_r_achieved, observation, particles, means, stds):
        """
        For each particle in the particle-based belief, compute likelihood of the observation.
        Args:
            batch: for one timestep, batches of object pose, robot pose, observation, action
            particles: for one timestep, shape [batch_size, num_particles, 3]
            means: dictionary {'q_o': [1, 1, 3], 'q_r': [1, 1, 3], ...}
            stds: dictionary {'q_o': [1, 1, 3], 'q_r': [1, 1, 3], ...}
        """

        # prepare input to the observation likelihood estimator network
        q_r_desired = q_r_desired.float().to(self.device) # shape [batch_size, 2]
        q_r_achieved = q_r_achieved.float().to(self.device) # shape [batch_size, 2]
        observation = observation.float().to(self.device) # shape [batch_size, 1]
        observation_input = torch.cat((q_r_desired, q_r_achieved, observation), dim=-1) # shape [batch_size, 5]
        observation_input = torch.tile(observation_input[:, None, :], (1, particles.shape[1], 1)) # shape [batch_size, num_particles, 5]
        particle_input = self.transform_particles_as_input(particles, means, stds) # shape [batch_size, num_particles, 4]
        input = torch.cat((observation_input, particle_input), dim=-1) # shape [batch_size, num_particles, 9]
        input = input.view(-1, input.shape[-1]).float().to(self.device) # shape [batch_size * num_particles, 9]

        # for each particle, estimate the likelihood based on the observation
        obs_likelihood = self.obs_like_estimator(input) # pass input particle set through the observaton likelihood estimator network
        obs_likelihood = obs_likelihood.view(q_r_desired.shape[0], particles.shape[1])

        return obs_likelihood # shape [batch_size, num_particles]
    
    # normalise 'particles' tensor
    def transform_particles_as_input(self, particles, means, stds):
        
        # ensure means and stds are torch tensors
        means_q_o = torch.tensor(means['q_o'], device=self.device)
        stds_q_o = torch.tensor(stds['q_o'], device=self.device)

        input = torch.cat((
            (particles[:, :, :2] - means_q_o[None, None, :2]) / stds_q_o[None, None, :2],
            torch.cos(particles[:, :, 2:3]),
            torch.sin(particles[:, :, 2:3])),
            dim=-1)

        return input # should the angle state theta be divided into two states, cos and sin theta?
    
    def propose_particles(self, q_r_desired, q_r_achieved, observation, num_particles, state_mins, state_maxs, xy_only):
        
        # achieved_minus_desired = q_r_achieved - q_r_desired
        input_pp = torch.cat((q_r_desired, q_r_achieved), dim=-1) # shape [batch_size, 4]
        duplicated_input_pp = torch.tile(input_pp[:, None, :], (1, num_particles, 1)) # duplicate the input 'num_particles' times so we generate multiple particles, shape [batch_size, num_particles, 4]
        duplicated_input_pp = duplicated_input_pp.view(-1, 4).float().to(self.device) # shape [batch_size * num_particles, 4]
        
        # normalise??
        # duplicated_input_pp = torch.nn.functional.normalize(duplicated_input_pp, p=2.0, dim=2)
        # print(duplicated_input_pp)
        
        # output dimension of particle proposer
        if xy_only == False:
            output_dim_pp = 4
        else:
            output_dim_pp = 2
        
        proposed_particles = self.particle_proposer(duplicated_input_pp) # shape [batch_size * num_particles, output_dim_pp]
        proposed_particles = proposed_particles.view(q_r_achieved.shape[0], num_particles, output_dim_pp) # shape [batch_size, num_particles, output_dim_pp]
        
        # outputs from particle proposer are between -1 and 1 --> scale according to state maximums and minimums
        if xy_only == False:
            proposed_particles = torch.cat((
                proposed_particles[:, :, :1] * (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0,
                proposed_particles[:, :, 1:2] * (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0,
                torch.atan2(proposed_particles[:, :, 3:4], proposed_particles[:, :, 2:3]))
                , dim=-1)
        else:
            proposed_particles = torch.cat((
                proposed_particles[:, :, :1] * (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0,
                proposed_particles[:, :, 1:2] * (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0),
                dim=-1)
        
        # EXPERIMENT: keep proposed particles states as x, y, sin, cos
        # proposed_particles = torch.cat((
        #     proposed_particles[:, :, :1] * (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0,
        #     proposed_particles[:, :, 1:2] * (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0,
        #     proposed_particles[:, :, 2:3],
        #     proposed_particles[:, :, 3:4]),
        #     dim=-1)
        
        return proposed_particles
    
    # for now training loop ignores resampling --> all the particles in the loop are from the particle proposer (no resampling from the previous particle set)
    def fit(self, data, split_ratio, batch_size, seq_len, num_epochs, patience, learning_rate, num_particles, xy_only):
        
        # split data into training and validation set
        train_data, val_data = split_data(data, split_ratio)

        train_dataset = TrajectoriesDataset(train_data, seq_len)
        val_dataset = TrajectoriesDataset(val_data, seq_len)

        # create dataloaders for training and validation set
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # compute some statistics about training data
        means, stds, q_o_maxs, q_o_mins, q_r_step_sizes, q_r_maxs, q_r_mins = compute_statistics(train_data)

        # optimiser and loss
        # EXPERIMENT: disable ole network and just have the particle_prob_list be uniform probability every sequence
        # optimiser_e2e = torch.optim.Adam([
        #     {'params': self.particle_proposer.parameters()},
        #     {'params': self.obs_like_estimator.parameters()}],
        #     lr=learning_rate)
        optimiser_e2e = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
        loss_fn = loss_fn_e2e()

        # initialise variables needed in the training loop
        epoch = 0
        train_loss_list = np.zeros((num_epochs,))

        # training loop: go through epochs (number of epochs = 'num_epochs')
        while epoch < num_epochs:
            # print the epoch number at the start of each epoch
            print(f"Epoch {epoch+1}/{num_epochs}")

            # training loop: for each epoch, go through multiple batches --> e.g. q_o batch has dimensions [batch_size, seq_len, 3]
            for i, batch in enumerate(train_dataloader):
                # load to gpu 
                batch = {k: v.float().to(self.device) for k, v in batch.items()}
                # initialise matrix to store particles
                particle_list = torch.zeros([batch_size, seq_len, num_particles, self.state_dim], 
                                            dtype=torch.float64, device=self.device)
                particle_prob_list = torch.zeros([batch_size, seq_len, num_particles], 
                                                 dtype=torch.float64, device=self.device)

                # for each time step
                for t in range(seq_len):
                    batch_t = {'q_o': batch['q_o'][:, :, :],
                            'q_r': batch['q_r'][:, t, :],
                            'o': batch['o'][:, t, :],
                            'a': batch['a'][:, t, :]}

                    # data for one timestep                
                    q_o = batch_t['q_o'].squeeze(1) # shape [batch_size, 3]
                    q_r_desired = batch_t['a'].squeeze(1) # desired robot pose is action from previous time step --> shape [batch_size, 2]
                    q_r_achieved = batch_t['q_r'].squeeze(1) # shape [batch_size, 2]
                    observation = batch_t['o'] # shape [batch_size, 1]

                    # test print - for debugging
                    # print(q_o)
                    # print(q_r_desired)
                    # print(q_r_achieved)
                    # print(observation)
                    
                    # propose particles
                    proposed_particles = self.propose_particles(q_r_desired, q_r_achieved, observation, num_particles, q_o_mins, q_o_maxs, xy_only) # shape [batch_size, num_particles, 3]
                    print(proposed_particles.shape)
                    particle_list[:, t, :, :] = proposed_particles
                    
                    # observation likelihood estimator network
                    # particle_probs = self.measurement_update(q_r_desired, q_r_achieved, observation, proposed_particles, means, stds)
                    # particle_prob_list[:, t, :] = particle_probs

                # EXPERIMENT: disable ole network and just have the particle_prob_list be uniform probability every sequence
                particle_prob_list = torch.ones([batch_size, seq_len, num_particles], dtype=torch.float64) / num_particles

                # compute loss
                train_loss = loss_fn(batch, particle_list, particle_prob_list, q_r_step_sizes, self.world, xy_only) # do we need step sizes here?? object doesn't move...

                # Backward and optimize
                optimiser_e2e.zero_grad()
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.particle_proposer.parameters(), 1)
                optimiser_e2e.step()

            # track training and validation loss at each epoch
            # // need to implement //
            train_loss_list[epoch] = train_loss

            # increment epoch
            epoch += 1

        # plot training and validation loss
        epochs = range(1, num_epochs+1)
        plt.plot(epochs, train_loss_list, label='Training loss')

        # labels and axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.legend()

        # return most recent batch of proposed particles
        return q_o.detach().numpy(), particle_list.detach().numpy(), particle_prob_list.detach().numpy()

    # testing particle update given an observation
    def test(self, state_mins, state_maxs, means, stds, xy_only):
        
        with torch.no_grad():
        
            # sample new object and robot pose
            self.world.sample_gt_object_pose()
            self.world.sample_robot_pose()

            observation_count = 0

            # perform rollouts of length 100, until a trajectory with at least one observation is found
            while observation_count < 1:
                q_r_hist, o_hist, a_hist = self.world.rollout(100, self.world.qo_gt.copy())
                observation_count = np.sum(o_hist)

            # only keep bits of sequence around positive contact measurements
            contact_idx = np.where(o_hist==1)[0]
            min_idx = np.min(contact_idx)
            max_idx = np.max(contact_idx)
            buffer_lb = np.min([min_idx, 10])
            buffer_ub = np.min([len(o_hist)-max_idx, 10])
            keep_idx = [min_idx-buffer_lb, max_idx+buffer_ub]
            q_r_hist = q_r_hist[keep_idx[0]:keep_idx[1]]
            a_hist = a_hist[keep_idx[0]:keep_idx[1]]
            o_hist = o_hist[keep_idx[0]:keep_idx[1]]
            contact_idx_new = np.where(o_hist==1)[0]
            self.world.plot_rollout(q_r_hist, o_hist)

            # propose particles
            q_r_desired = torch.from_numpy(a_hist[contact_idx_new[0]-1, :])
            q_r_achieved = torch.from_numpy(q_r_hist[contact_idx_new[0], :])
            observation = torch.from_numpy(np.array([o_hist[contact_idx_new[0]]]))
            print('Desired robot position during contact event:', q_r_desired[None, :])
            print('Achieved robot position during contact event:', q_r_achieved[None, :])
            print('Observation during contact event:', observation[None, :])
            state_mins = torch.from_numpy(state_mins).to(self.device)
            state_maxs = torch.from_numpy(state_maxs).to(self.device)
            self.world.particles = self.propose_particles(q_r_desired[None, :], q_r_achieved[None, :], observation[None, :], self.world.num_particles, state_mins, state_maxs, xy_only) # shape [1, num_particles, 3]
            
            # set all particle angles to zero (just for visualisation purposes)
            if xy_only == True:
                zero_angles = torch.zeros(1, self.world.num_particles, 1)
                self.world.particles = torch.cat((self.world.particles, zero_angles), dim=-1)
            
            # EXPERIMENT: use cos and sin as states for angles, instead of theta
            # self.world.particles[:, :, 2:3] = torch.atan2(self.world.particles[:, :, 3:4], self.world.particles[:, :, 2:3])
            # self.world.particles = self.world.particles[:, :, :3]

            self.world.particles = self.world.particles.squeeze(0)
            self.world.q_r_0 = q_r_hist[contact_idx_new[0], :]

            # print(q_r_desired[None, :].shape)
            # print(q_r_achieved[None, :].shape)
            # print(observation[None, :].shape)
            # print(self.world.particles[None, :, :].shape)
            # self.world.weights = self.measurement_update(q_r_desired[None, :], q_r_achieved[None, :], observation[None, :], self.world.particles[None, :, :], means, stds)
            # print(self.world.weights)

            # plot belief
            self.world.plot_belief_with_obs(q_r_desired, q_r_achieved)

    def plot_GMM(self, q_o, particle_list, particle_prob_list, xy_only):
        """
        Args:
            q_o: [batch_size, 3]
            particle_list: [batch_size, seq_len, num_particles, 3]
            particle_prob_list: [batch_size, seq_len, num_particles]
        Returns:
            plots of GMM used to calculate loss (separate plots for x, y, theta)
        """
        # randomly select trajectory to plot
        num_trajectories = particle_list.shape[0]
        traj = np.random.randint(0, num_trajectories)

        # randomly select time step to plot
        # seq_len = particle_list.shape[1]
        # t = np.random.randint(0, seq_len)
        # if contact_only=True
        t = 0

        # randomly select particles to plot
        num_particles = particle_list.shape[2]
        # number of particles we wish to plot
        num_particles_plot = 100
        particles_plot_indices = np.random.randint(0, num_particles, size=num_particles_plot)

        if xy_only == False:
            keys = ['x', 'y', 'theta']
        else:
            keys = ['x', 'y']
        
        particle_list_plot = {key: particle_list[traj, t, particles_plot_indices, i] for i, key in enumerate(keys)} # dictionary values all have shapes [num_particles_plot]
        particle_prob_list_plot = particle_prob_list[traj, t, particles_plot_indices] # [num_particles_plot]

        # ground truth object pose
        q_o_plot = {key: q_o[traj, i] for i, key in enumerate(keys)} # dictionary values all have shapes []
        
        # plotting ranges for each of the states
        if xy_only == False:
            ranges = {keys[0]: [-0.5, 0.5],
                      keys[1]: [0.5, 1.5],
                      keys[2]: [-3.14, 3.14]}
        else:
            ranges = {keys[0]: [-0.5, 0.5],
                      keys[1]: [0.5, 1.5]}
        
        # scalings for different state dimensions
        if xy_only == False:
            scalings = [0.1, 0.1, 1]
        else:
            scalings = [0.1, 0.1]
        
        # initialise plots which will contain subplots
        fig, axs = plt.subplots(len(keys)*2, figsize=(12, 12))
        fig.suptitle('Gaussian Mixtures for all states + NLL plots')

        num_points = 500
        # dictionary containing values of the GMM for x, y, theta
        y_sum = {key: np.zeros(num_points) for key in keys}

        # dictionary containing closest index of state values closest to the ground truth object state
        closest_index = {key: np.zeros(1) for key in keys}

        # dictionary containing state values closest to the ground truth object state
        closest_x = {key: np.zeros(1) for key in keys}
        
        # sigma parameter for the GMMs
        sigma = 0.2

        for i, ax in enumerate(axs):
            
            # GMM plots
            if i < len(keys):
                particle_states_to_plot = particle_list_plot[keys[i]] # [num_particles_plot]
                
                # x = np.linspace(mu - 3*sigma, mu + 3*sigma, num_points)
                x = np.linspace(ranges[keys[i]][0], ranges[keys[i]][1], num_points)

                for j in range(num_particles_plot):
                    # plot individual Gaussians
                    mu = particle_states_to_plot[j]
                    sq_distance = ((x - mu) / scalings[i]) ** 2.0
                    Z = np.sqrt(2.0 * np.pi * (sigma ** 2.0))
                    y = (particle_prob_list_plot[j] / Z) * np.exp(-sq_distance / (2.0 * (sigma ** 2.0)))
                    ax.plot(x, y, 'k', '--', linewidth=0.4, zorder=1)

                    # sum the individual Gaussians
                    y_sum[keys[i]] += y
                
                # plot Gaussian mixture
                ax.plot(x, y_sum[keys[i]], color='tab:blue', zorder=0, label='Gaussian mixture from particles')

                # plot ground truth object state --> value of GMM
                gt_object_state = q_o_plot[keys[i]]
                absolute_differences = np.abs(x - gt_object_state)
                closest_index[keys[i]] = np.argmin(absolute_differences)
                closest_x[keys[i]] = x[closest_index[keys[i]]]
                gt_y_sum = y_sum[keys[i]][closest_index[keys[i]]]
                ax.plot([closest_x[keys[i]], closest_x[keys[i]]], [0, gt_y_sum], color='tab:red', zorder=2, label='Value of GMM at ground truth object state')
                ax.plot([closest_x[keys[i]]], [0], color='tab:red', marker='x', markersize=12, zorder=2)

                # Label the ground truth point with its value
                ax.text(closest_x[keys[i]], gt_y_sum, f'{gt_y_sum:.2f}', fontsize=10, color='tab:red', verticalalignment='bottom', horizontalalignment='right')
                
                # plot limits
                ax.set_xlim([ranges[keys[i]][0], ranges[keys[i]][1]])  # set x-axis limits based on ranges
                ax.set_ylim([0, np.max(y_sum[keys[i]]) * 1.1])
                ax.set_yticks(np.linspace(0, np.max(y_sum[keys[i]]) * 1.1, 5))  # create 5 evenly spaced ticks
                ax.grid(True)
                ax.legend()
            
            # NLL plots
            else:
                print(keys[i-len(keys)])
                print('max and min:', np.max(y_sum[keys[i-len(keys)]]), np.min(y_sum[keys[i-len(keys)]]))

                # plot NLL
                x = np.linspace(ranges[keys[i-len(keys)]][0], ranges[keys[i-len(keys)]][1], num_points)
                ax.plot(x, -np.log(1e-16 + y_sum[keys[i-len(keys)]]), color='tab:green', label='Negative log likelihood')
                ax.set_xlim([ranges[keys[i-len(keys)]][0], ranges[keys[i-len(keys)]][1]])  # set x-axis limits based on ranges

                # plot ground truth object state --> value of NLL
                nll_value_at_gt = -np.log(1e-16 + y_sum[keys[i-len(keys)]][closest_index[keys[i-len(keys)]]])
                ax.plot([closest_x[keys[i-len(keys)]], closest_x[keys[i-len(keys)]]], [0, nll_value_at_gt], color='tab:red', zorder=2, label='Value of NLL at ground truth object state')

                # Label the ground truth point with its value on NLL plot
                ax.text(closest_x[keys[i-len(keys)]], nll_value_at_gt, f'{nll_value_at_gt:.2f}', fontsize=10, color='tab:red', verticalalignment='bottom', horizontalalignment='right')

                ax.grid(True)
                ax.legend()
            
        plt.tight_layout()

    # belief update loop with resampling, prediction, and measurement update steps
    def belief_update_loop(self, batch, num_particles, means, stds, state_mins, state_maxs, state_step_sizes):
        # get shapes and define parameters
        self.batch_size = batch['q_o'].shape[0]
        self.seq_len = batch['q_o'].shape[1]
        self.num_particles = num_particles

        # initialise particles --> samples particles randomly according to uniform distribution between state minimums and maximums
        initial_particles = [torch.rand(self.batch_size, self.num_particles, 1, device=self.device) for d in range(self.state_dim)] # generate tensors of shape [batch_size, num_particles, 1] with values between 0 and 1
        initial_particles = torch.cat([(state_mins[d] - state_maxs[d]) * initial_particles[d] + state_maxs[d] for d in range(self.state_dim)], dim=-1) # rescale to range (state_min, state_max) and concatenate

        # initial particle probabilities --> distribute particle weights uniformly
        initial_particle_probs = torch.ones(self.batch_size, num_particles) / self.num_particles

        # 'samples' has shape [batch_size, num_resampled] --> for each batch/trajectory, stores indices of the particles that we want to resample
        # 'particles' has shape [batch_size, num_particles, state_dim] --> contains a batch of particle sets (at one specific time step)
        # this function picks and permutes particle samples from 'particles' according to indices specified in 'samples'
        # 'result' has shape [batch_size, sample_size, state_dim], where sample_size = num_resampled
        def permute_batch(particles, samples):

            # get shapes
            batch_size = particles.shape[0]
            num_particles = particles.shape[1]
            sample_size = samples.shape[1]

            # create a 2D array that contains indices of the samples
            batch_indices_1D = torch.reshape(torch.arange(batch_size), (batch_size, 1)) # shape: [batch_size, 1]
            batch_indices_2D = torch.tile(batch_indices_1D, (1, sample_size)) # shape: [batch_size, sample_size]
            idx = samples + num_particles * batch_indices_2D

            # pick out particle samples from batch of particles
            particles_flattened = torch.reshape(particles, (batch_size * num_particles, -1))
            result = torch.gather(particles_flattened, dim=0, index=idx)
            result = torch.reshape(result, (particles[:, :sample_size].shape))

            return result
        
        def loop(particles, particle_probs, particle_list, particle_probs_list, i):

            # determine number of particles to propose and number of particles to resample, based on propose ratio
            # propose_ratio is ratio of proposed to resampled particles --> this follows an exponential function (gamma)^(t-1)
            num_proposed_float = torch.round((self.propose_ratio ** int(i)) * self.num_particles_float)
            num_proposed = num_proposed_float.int()
            num_resampled_float = self.num_particles_float - num_proposed_float
            num_resampled = num_resampled_float.int()

            # as long as propose ratio is less than 1.0, execute resampling and measurement update
            if self.propose_ratio < 1.0:
                
                # resampling
                evenly_spaced_markers = torch.linspace(0.0, (num_resampled_float - 1.0) / num_resampled, num_resampled) # 'num_resampled' evenly spaced markers from 0 to 1
                random_offset = torch.rand(self.batch_size, device=self.device) # generate a tensor of shape [batch_size] filled with random values between 0 and 1
                random_offset *= random_offset * (1 / num_resampled_float) # tensor has shape [batch_size] and is filled with random values between 0 and 1 / num_resampled
                markers = random_offset[:, None] + evenly_spaced_markers[None, :] # broadcast to get shape [batch_size, num_resampled]
                cum_probs = torch.cumsum(particle_probs, dim=1) # particle_probs has shape [batch_size, num_particles] --> take cum_sum along second dimension
                marker_matching = markers[:, :, None] < cum_probs[:, None, :] # broadcasting to create a boolean tensor [batch_size, num_resampled, num_particles], where entry is 'True' if marker < cum prob
                samples = torch.argmax(marker_matching.int(), dim=2).int() # for each batch, each particle to resample, extract index of the first 'True' along the last dimension of 'marker_matching'
                standard_particles = permute_batch(particles, samples) # pick and permute particle samples from 'particles' according to indices specified in 'samples' --> shape [batch_size, sample_size, state_dim]
                standard_particle_probs = torch.ones(self.batch_size, num_resampled, device=self.device) # initialise tensor to store resampled particle probabilities --> shape [batch_size, sample_size]
                standard_particles = standard_particles.detach() # stop gradient computation??
                standard_particle_probs = standard_particle_probs.detach() # stop gradient computation??

                # motion update --> unnecessary as object pose does not change?
                # measurement update
                # only pass one time step of the batch into propose_particles function
                batch_i = {'q_o': batch['q_o'][:, :, :],
                            'q_r': batch['q_r'][:, i, :],
                            'o': batch['o'][:, i, :],
                            'a': batch['a'][:, i, :]}
                
                q_r_desired = batch_i['a'].squeeze(1) # desired robot pose is action from previous time step --> shape [batch_size, 2]
                q_r_achieved = batch_i['q_r'].squeeze(1) # shape [batch_size, 2]
                observation = batch_i['o'].squeeze(1) # shape [batch_size, 1]

                standard_particle_probs *= self.measurement_update(q_r_desired, q_r_achieved, observation, standard_particles, means, stds) # particle probs = ones * obs_likelihoods
            
            # as long as propose ratio is greater than 0.0, execute particle proposing step
            if self.propose_ratio > 0.0:
                
                # proposed particles
                proposed_particles = self.propose_particles(q_r_desired, q_r_achieved, observation, num_proposed, state_mins, state_maxs)
                proposed_particle_probs = torch.ones(self.batch_size, num_proposed) # initialise tensor to store proposed particle probabilities

            # combine standard particles (particles that were resampled in the beginning of the loop and went through measurement update) with proposed particles
            # 'particles' represents the particle set at the end of the loop
            # if propose_ratio is 1
            if self.propose_ratio == 1.0:
                
                # then all of the proposed particles are used in the new particle set
                particles = proposed_particles
                particle_probs = proposed_particle_probs

            # if propose_ratio is 0
            elif self.propose_ratio == 0.0:
                
                # then all of the resampled particles are used in the new particle set
                particles = standard_particles
                particle_probs = standard_particle_probs

            # otherwise, the new particle set is a combination of resampled (standard) particles and proposed particles --> propose ratio = ratio of proposed to resampled particles
            else:
                
                # prepare to combine resampled particles and proposed particle probabilities
                standard_particle_probs *= (num_resampled_float / self.num_particles_float) / torch.sum(standard_particle_probs, dim=1, keepdim=True)
                proposed_particle_probs *= (num_proposed_float / self.num_particles_float) / torch.sum(proposed_particle_probs, dim=1, keepdim=True)

                # combine resampled and proposed particles
                particles = torch.cat([standard_particles, proposed_particles], dim=1)
                particle_probs = torch.cat([standard_particle_probs, proposed_particle_probs], dim=1)

            # normalise probabilites
            particle_probs /= torch.sum(particle_probs, dim=1, keepdim=True)

            # add particles and particle probabilities from this timestep to the list
            particle_list = torch.cat([particle_list, particles[:, None]], dim=1)
            particle_probs_list = torch.cat([particle_probs_list, particle_probs[:, None]], dim=1)

            return particles, particle_probs, particle_list, particle_probs_list
        
        # reshape particle_list and particle_probs_list so they are the right shapes for the loop() function
        particle_list = torch.reshape(initial_particles, (self.batch_size, -1, self.num_particles, self.state_dim))
        particle_probs_list = torch.reshape(initial_particle_probs, (self.batch_size, -1, self.num_particles))

        # initialise particles and their probabilities for the loop
        particles = initial_particles
        particle_probs = initial_particle_probs

        # run particle filter loop
        for i in range(self.seq_len):
            particles, particle_probs, particle_list, particle_probs_list = loop(particles, particle_probs, particle_list, particle_probs_list, i)

        return particles, particle_probs, particle_list, particle_probs_list


# USEFUL CLASSES
# class that defines particle proposer network
class ParticleProposer(nn.Module):

    # initialise neural network architecture
    def __init__(self, proposer_keep_ratio, xy_only):
        super().__init__()
        if xy_only == False:
            self.linear_stack = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Dropout(p=proposer_keep_ratio),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
                nn.Tanh(), # tanh maps to between -1 and 1
            )
        else:
            self.linear_stack = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Dropout(p=proposer_keep_ratio),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Tanh(), # tanh maps to between -1 and 1
            )
    #     self._initialize_weights()

    # # Initialize weights with Xavier initialization
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight, gain=2)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    # define the forward pass of the network (how input data x moves through network layers)
    def forward(self, x):
        x = self.linear_stack(x)
        return x

# class that defines observation likelihood estimator network
class ObsLikelihoodEstimator(nn.Module):

    # initialise neural network architecture
    def __init__(self, min_obs_likelihood):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.min_obs_likelihood = min_obs_likelihood
    
    # define the forward pass of the network (how input data x moves through network layers)
    def forward(self, x):
        x = self.linear_stack(x)
        x = x * (1 - self.min_obs_likelihood) + self.min_obs_likelihood # ensures that all probabilities outputted are higher than minimum observation likelihood (a design parameter)
        return x
    
# class which inherits from Dataset Pytorch class --> allows us to obtain one subsequence sample from our dataset
class TrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.num_trajectories = data['q_r'].shape[0]
        self.trajectory_len = data['q_r'].shape[1]
        self.num_subsequences = self.trajectory_len // self.seq_len

    def __len__(self):
        return self.num_trajectories * self.num_subsequences

    def __getitem__(self, idx):
        traj_idx = math.floor(idx // self.num_subsequences)
        subseq_idx = idx % self.num_subsequences
        start_idx = subseq_idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        sample = {
            'q_o': self.data['q_o'][traj_idx, :, :],
            'q_r': self.data['q_r'][traj_idx, start_idx:end_idx, :],
            'o': self.data['o'][traj_idx, start_idx:end_idx, :],
            'a': self.data['a'][traj_idx, start_idx:end_idx, :]
        }
        return sample

# class that defines end to end training loss function
class loss_fn_e2e(nn.Module):
    def __init__(self):
        super(loss_fn_e2e, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes, world, xy_only):
        # GMM LOSS
        # std = 0.001
        # particle_std = 0.25
        std = 0.2
        particle_std = 0.2
        print('particle_list', particle_list[0, 0, :10, :])
        print('batch_q_o:', batch['q_o'][0, 0, :])
        # print('state_step_sizes:', state_step_sizes)
        sq_distance = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes, xy_only) # shape: [batch_size, seq_len, num_particles]
        print('sq_distance:', sq_distance[0, 0, :10])
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        # print('std_tensor:', std_tensor)
        activations = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=-1)))
        print('activations:', activations[0, 0, :10])
        print('sum of activations:', torch.sum(activations, dim=-1))
        print('sum of activations plus 1e-16:', 1e-16 + torch.sum(activations, dim=-1))
        print('negative log:', -torch.log(1e-16 + torch.sum(activations, dim=-1)))
        print('loss:', loss)

        # DISTANCE LOSS
        # q_r = batch['q_r']
        # distances = dist_to_object_torch(q_r, particle_list, world.obj_dims, world.r_robot, xy_only)
        # squared_distances = distances ** 2.0
        # dist_loss = torch.mean(torch.sum(squared_distances, dim=-1)) * 10
        # print('dist_loss:', dist_loss)

        # SPREAD LOSS
        # particle_list: [batch_size, seq_len, num_particles, state_dim] --> take std along num_particles dimension
        # loss_spread = torch.std(particle_list, dim=2) # shape [batch_size, seq_len, state_dim]
        # loss_spread = torch.mean(loss_spread) * 10
        # print('loss_spread:', loss_spread)

        # particle_list: [batch_size, seq_len, num_particles, state_dim]
        # batch['q_o']: [batch_size, seq_len, state_dim]
        # sq_distance = torch.mean(torch.sum(sq_distance, dim=-1))
        # print('sq_distance:', sq_distance)

        return loss
    
# maximise GMM at true state, minimise GMM at all other states
class loss_fn_max_min(nn.Module):
    def __init__(self):
        super(loss_fn_max_min, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes, world, xy_only):
        # compute and set some parameters
        batch_size = batch['q_o'].shape[0]
        seq_len = batch['q_r'].shape[1]
        num_particles = particle_list.shape[2]
        state_dim = batch['q_o'].shape[-1]
        std = 0.1
        
        # GMM at true state
        sq_distance_true = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes, xy_only) # shape: [batch_size, seq_len, num_particles]
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        activations_true = ((1/num_particles) / torch.sqrt(std_tensor)) * torch.exp(-sq_distance_true / (2.0 * (std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        loss_true = torch.mean(-torch.log(1e-16 + torch.sum(activations_true, dim=-1)))
        print('true loss:', loss_true)

        # GMM at all other states
        buffer = 0.01
        num_states_other = 100
        sq_distance_other = compute_sq_distance_other(batch['q_o'], num_states_other, buffer)
        activations_other = ((1/num_particles) / torch.sqrt(std_tensor)) * torch.exp(-sq_distance_other / (2.0 * (std ** 2.0)))
        loss_other = torch.mean(-torch.log(1e-16 + activations_other))
        print('other loss:', loss_other)

        # DISTANCE LOSS
        # q_r = batch['q_r']
        # distances = dist_to_object_torch(q_r, particle_list, world.obj_dims, world.r_robot, xy_only)
        # squared_distances = distances ** 2.0
        # dist_loss = torch.mean(torch.sum(squared_distances, dim=-1)) * 10
        # print('dist_loss:', dist_loss)

        # overall loss
        return loss_true - loss_other
    
# geometric loss
class loss_fn_geometric(nn.Module):
    def __init__(self):
        super(loss_fn_geometric, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes, world, xy_only):
        
        q_r = batch['q_r']
        distances = dist_to_object_torch(q_r, particle_list, world.obj_dims, world.r_robot, xy_only)
        squared_distances = distances ** 2.0
        dist_loss = torch.mean(torch.sum(squared_distances, dim=-1))
        # print('dist_loss:', dist_loss)

        # std = 0.03
        # particle_std = 0.03
        # sq_distance = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes) # shape: [batch_size, seq_len, num_particles]
        # std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        # activations = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        # GMM_loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=-1))) * (1/40)
        # print('GMM_loss:', GMM_loss)

        # return dist_loss + GMM_loss
        return dist_loss

# during contact (when observation = 1) we maximise particles-based GMM at true object state, during no contact (when observation = 0) we minimise particle-based GMM at robot state
class loss_fn_cross_GMM(nn.Module):
    def __init__(self):
        super(loss_fn_cross_GMM, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        std = 0.001
        particle_std = 0.8
        print('particle_list', particle_list[0, 0, :10, :])
        print('batch_q_o:', batch['q_o'][0, 0, :])
        print('state_step_sizes:', state_step_sizes)
        sq_distance_q_o = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes) # shape: [batch_size, seq_len, num_particles]
        sq_distance_q_r = compute_sq_distance(particle_list, batch['q_r'], state_step_sizes)
        print('sq_distance:', sq_distance_q_o[0, 0, :10])
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        print('std_tensor:', std_tensor)
        activations_q_o = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance_q_o / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        activations_q_r = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance_q_r / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        activations = batch['o'] * activations_q_o + 0.01 * (1 - batch['o']) * (-activations_q_r)
        loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=2)))
        # print(torch.sum(activations, dim=-1))
        print('loss:', loss)
        return loss

# class that defines loss function that computes the distance between each particle and the robot pose, sums the distances up for the particle set, averages the sum of distances across timesteps and batches
# class loss_fn_dist_to_robot_A(nn.Module):
#     def __init__(self):
#         super(loss_fn_dist_to_robot_A, self).__init__()

#     def forward(self, batch, particle_list, particle_prob_list, state_step_sizes, world):
#         batch_q_o = batch['q_o'] # shape [batch_size, 1, state_dim]
#         batch_q_o = batch_q_o[:, :, None, :] # add dimension to tensor containing batch of object poses --> shape [batch_size, 1, 1, state_dim]
#         batch_q_r = batch['q_r'] # shape [batch_size, seq_len, 2]
#         distances = torch.zeros(batch_q_o.shape[0], particle_list.shape[1], particle_list.shape[2]) # shape [batch_size, seq_len, num_particles]
#         # particle_list has shape [batch_size, seq_len, num_particles, state_dim]
#         for batch in range(batch_q_o.shape[0]):
#             for t in range(particle_list.shape[1]):
#                 for particle in range(particle_list.shape[2]):
#                     distances[batch, t, particle] = dist_to_object_torch(batch_q_r[batch, t, :], particle_list[batch, t, particle, :], world.obj_dims, world.r_robot)
#         loss = torch.mean(torch.sum(particle_prob_list * distances, dim=-1))
#         # normalise smth?
#         return loss

# loss function above but simpler --> simply calculate squared distance between robot and particle (ignore angle of object pose)
class loss_fn_dist_to_robot_B(nn.Module):
    def __init__(self):
        super(loss_fn_dist_to_robot_B, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        batch_q_r = batch['q_r'][:, :, None, :] # shape [batch_size, seq_len, 1, 2] = [20, 100, 1, 2]
        
        # distances has shape [batch_size, seq_len, num_particles]
        # particle_list has shape [batch_size, seq_len, num_particles, state_dim]
        x_distance = (batch_q_r[:, :, :, 0] - particle_list[:, :, :, 0])
        y_distance = (batch_q_r[:, :, :, 1] - particle_list[:, :, :, 1])
        squared_distance = (x_distance ** 2.0) + (y_distance ** 2.0)
        loss = torch.mean(torch.sum(squared_distance, dim=-1))
        return loss
    
class loss_combined(nn.Module):
    def __init__(self):
        super(loss_combined, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        batch_q_r = batch['q_r'][:, :, None, :] # shape [batch_size, seq_len, 1, 2] = [20, 100, 1, 2]
        
        # distances has shape [batch_size, seq_len, num_particles]
        # particle_list has shape [batch_size, seq_len, num_particles, state_dim]
        x_distance = (batch_q_r[:, :, :, 0] - particle_list[:, :, :, 0])
        y_distance = (batch_q_r[:, :, :, 1] - particle_list[:, :, :, 1])
        squared_distance = (x_distance ** 2.0) + (y_distance ** 2.0)
        # threshold = 0.067**2.0 + 0.14**2.0
        # squared_distance[squared_distance < threshold] = 0 # boolean masking
        loss_dist = torch.mean(torch.sum(squared_distance, dim=-1))

        std = 0.01
        particle_std = 0.05
        sq_distance = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes) # shape: [batch_size, seq_len, num_particles]
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        activations = particle_prob_list / torch.sqrt(std_tensor) * torch.exp(-sq_distance / (2.0 * particle_std ** 2.0)) # shape: [batch_size, seq_len, num_particles]
        loss_prob = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=-1)))

        # print(loss_dist)
        # print(loss_prob)

        w1= 1.0
        w2 = 0.4
        loss = (w1 * loss_dist) + (w2 * loss_prob)

        return loss
    
# like loss function B but uses object pose instead of robot pose
class loss_fn_dist_to_robot_C(nn.Module):
    def __init__(self):
        super(loss_fn_dist_to_robot_C, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        batch_q_o = batch['q_o'][:, :, None, :] # shape [batch_size, 1, 1, 2] = [20, 1, 1, 2]
        
        # distances has shape [batch_size, seq_len, num_particles]
        # particle_list has shape [batch_size, seq_len, num_particles, state_dim]
        x_distance = (batch_q_o[:, :, :, 0] - particle_list[:, :, :, 0])
        y_distance = (batch_q_o[:, :, :, 1] - particle_list[:, :, :, 1])
        squared_distance = (x_distance ** 2.0) + (y_distance ** 2.0)
        loss = torch.mean(torch.sum(squared_distance, dim=-1))
        return loss
    
class GMM_plus_distance_loss(nn.Module):
    def __init__(self):
        super(GMM_plus_distance_loss, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        std = 0.001
        particle_std = 0.6
        print('particle_list', particle_list[0, 0, :10, :])
        print('batch_q_o:', batch['q_o'][0, 0, :])
        print('state_step_sizes:', state_step_sizes)
        sq_distance = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes) # shape: [batch_size, seq_len, num_particles]
        print('sq_distance:', sq_distance[0, 0, :10])
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        print('std_tensor:', std_tensor)
        activations = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        print('activations:', activations[0, 0, :10])
        loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=-1)))
        # print(torch.sum(activations, dim=-1))
        print('loss:', loss)

        # MSE distance loss
        batch_q_r = batch['q_r'][:, :, None, :] # shape [batch_size, seq_len, 1, 2] = [20, 100, 1, 2]
        
        # distances has shape [batch_size, seq_len, num_particles]
        # particle_list has shape [batch_size, seq_len, num_particles, state_dim]
        x_distance = (batch_q_r[:, :, :, 0] - particle_list[:, :, :, 0])
        y_distance = (batch_q_r[:, :, :, 1] - particle_list[:, :, :, 1])
        squared_distance = (x_distance ** 2.0) + (y_distance ** 2.0)
        threshold = (np.sqrt((0.067 ** 2) + (0.14 ** 2)) + 0.05) ** 2
        squared_distance = torch.where(squared_distance > threshold, torch.tensor(0.0), squared_distance)
        dist_loss = torch.mean(torch.sum(squared_distance, dim=-1))
        print('dist_loss:', dist_loss)

        return loss + 3*dist_loss
    
# alternative e2e loss that switches the order of the sq_distance, Gaussian activation, logs
class loss_fn_e2e_alt(nn.Module):
    def __init__(self):
        super(loss_fn_e2e_alt, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        # std = 0.001
        # particle_std = 0.25
        std = 0.02
        particle_std = 0.02
        # print('particle_list', particle_list[0, 0, :10, :])
        # print('batch_q_o:', batch['q_o'][0, 0, :])

        batch_size = batch['q_o'].shape[0]
        seq_len = batch['q_o'].shape[1]
        num_particles = particle_list.shape[2]
        state_dim = batch['q_o'].shape[-1]

        # print('state_step_sizes:', state_step_sizes)
        sq_distance = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes) # shape: [batch_size, seq_len, num_particles, state_dim]
        # print('sq_distance:', sq_distance[0, 0, :10])
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        # print('std_tensor:', std_tensor)
        activations = torch.zeros(batch_size, seq_len, num_particles, state_dim)
        loss = torch.zeros(state_dim)
        for i in range(state_dim):
            activations[:, :, :, i] = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance[:, :, :, i] / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
            loss[i] = torch.mean(-torch.log(1e-16 + torch.sum(activations[:, :, :, i], dim=-1)))
            # print('activations:', activations[0, 0, :10])
            # print('sum of activations:', torch.sum(activations, dim=-1))
            # print('sum of activations plus 1e-16:', 1e-16 + torch.sum(activations, dim=-1))
            # print('negative log:', -torch.log(1e-16 + torch.sum(activations, dim=-1)))
            # print('loss:', loss)

        return torch.sum(loss)