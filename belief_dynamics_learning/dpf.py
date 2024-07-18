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

    def __init__(self, propose_ratio, proposer_keep_ratio, min_obs_likelihood, num_particles, world: World2D):
        """
        Apply differentiable particle filter (DPF) to current belief of the object pose (represented by particle set)
        to predict the belief at the next time step.
        Args:
            propose_ratio: ratio of proposed to resampled particles
            proposer_keep_ratio: dropout ratio for particle proposer network, used in inference to introduce randomness to particle proposer network
            min_obs_likelihood: lower limit for observation likelihood produced by the observation likelihood estimate
            world: object of class World2D
        """
        # store hyperparameters
        self.propose_ratio = propose_ratio
        self.proposer_keep_ratio = proposer_keep_ratio
        self.min_obs_likelihood = min_obs_likelihood
        self.world = world
        
        # define more parameters
        self.state_dim = 3 # --> x, y, theta for the object pose
        self.num_particles_float = num_particles
        self.num_particles = int(num_particles)

        # build learnable networks: observation likelihood estimator, particle proposer
        self.build_networks()

    def build_networks(self):
        
        # device configuration --> use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        # PARTICLE PROPOSER --> maps observations and robot poses to particles
        self.particle_proposer = ParticleProposer(self.proposer_keep_ratio).to(self.device)

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
        observation_input = torch.cat((q_r_desired, q_r_achieved, observation), dim=-1) # shape [batch_size, 5]
        observation_input = torch.tile(observation_input[:, None, :], (1, particles.shape[1], 1)) # shape [batch_size, num_particles, 5]
        particle_input = self.transform_particles_as_input(particles, means, stds) # shape [batch_size, num_particles, 3]
        input = torch.cat((observation_input, particle_input), dim=-1) # shape [batch_size, num_particles, 8]
        input = input.view(-1, 8).float() # shape [batch_size * num_particles, 8]

        # for each particle, estimate the likelihood based on the observation
        obs_likelihood = self.obs_like_estimator(input) # pass input particle set through the observaton likelihood estimator network
        obs_likelihood = obs_likelihood.view(q_r_desired.shape[0], particles.shape[1])

        return obs_likelihood
    
    # normalise 'particles' tensor
    def transform_particles_as_input(self, particles, means, stds):
        
        # ensure means and stds are torch tensors
        means_q_o = torch.tensor(means['q_o'], device=self.device)
        stds_q_o = torch.tensor(stds['q_o'], device=self.device)

        return (particles - means_q_o) / stds_q_o # should the angle state theta be divided into two states, cos and sin theta?
    
    def propose_particles(self, q_r_desired, q_r_achieved, observation, num_particles, state_mins, state_maxs):

        input_pp = torch.cat((q_r_desired, q_r_achieved, observation), dim=-1) # shape [batch_size, 5]
        duplicated_input_pp = torch.tile(input_pp[:, None, :], (1, num_particles, 1)) # duplicate the input 'num_particles' times so we generate multiple particles, shape [batch_size, num_particles, 5]
        duplicated_input_pp = duplicated_input_pp.view(-1, 5).float() # shape [batch_size * num_particles, 5]
        proposed_particles = self.particle_proposer(duplicated_input_pp) # shape [batch_size * num_particles, 3]
        proposed_particles = proposed_particles.view(q_r_achieved.shape[0], num_particles, 3) # shape [batch_size, num_particles, 3]

        # outputs from particle proposer are between 0 and 1 --> scale according to state maximums and minimums
        proposed_particles = torch.cat([
            proposed_particles[:, :, :1] * (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0,
            proposed_particles[:, :, 1:2] * (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0,
            proposed_particles[:, :, 2:3] * (state_maxs[2] - state_mins[2]) / 2.0 + (state_maxs[2] + state_mins[2]) / 2.0,
        ], dim=-1)
        
        return proposed_particles
    
    # for now training loop ignores resampling --> all the particles in the loop are from the particle proposer (no resampling from the previous particle set)
    def fit(self, data, split_ratio, batch_size, seq_len, num_epochs, patience, learning_rate, num_particles):
        
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
        optimiser_e2e = torch.optim.Adam([
            {'params': self.particle_proposer.parameters()},
            {'params': self.obs_like_estimator.parameters()}],
            lr=learning_rate)
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

                # initialise matrix to store particles
                particle_list = torch.zeros([batch_size, seq_len, num_particles, self.state_dim], dtype=torch.float64)
                particle_prob_list = torch.zeros([batch_size, seq_len, num_particles], dtype=torch.float64)

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
                    
                    # propose particles
                    proposed_particles = self.propose_particles(q_r_desired, q_r_achieved, observation, num_particles, q_o_mins, q_o_maxs)
                    particle_list[:, t, :, :] = proposed_particles
                    
                    # observation likelihood estimator network
                    particle_probs = self.measurement_update(q_r_desired, q_r_achieved, observation, proposed_particles, means, stds)
                    particle_prob_list[:, t, :] = particle_probs

                # compute loss
                train_loss = loss_fn(batch, particle_list, particle_prob_list, q_r_step_sizes) # do we need step sizes here?? object doesn't move...

                # Backward and optimize
                optimiser_e2e.zero_grad()
                train_loss.backward()
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

    # testing particle update given an observation
    def test(self, state_mins, state_maxs):
        
        with torch.no_grad():
        
            # sample new object and robot pose
            self.world.sample_gt_object_pose()
            self.world.sample_robot_pose()

            observation_count = 0

            # perform rollouts of length 100, until a trajectory with at least one observation is found
            while observation_count == 0:
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
            o_hist = o_hist[keep_idx[0]:keep_idx[1]]
            contact_idx_new = np.where(o_hist==1)[0]
            self.world.plot_rollout(q_r_hist, o_hist)

            # propose particles
            q_r_desired = torch.from_numpy(a_hist[contact_idx_new[0]-1, :])
            q_r_achieved = torch.from_numpy(q_r_hist[contact_idx_new[0], :])
            observation = torch.from_numpy(np.array([o_hist[contact_idx_new[0]]]))
            self.world.particles = self.propose_particles(q_r_desired[None, :], q_r_achieved[None, :], observation[None, :], self.world.num_particles, state_mins, state_maxs)
            self.world.particles = self.world.particles.squeeze(0)
            self.world.q_r_0 = q_r_hist[contact_idx_new[0], :]

            # plot belief
            self.world.plot_belief()


    def belief_update_loop(self, batch, num_particles, means, stds, state_mins, state_maxs, state_step_sizes):
        # get shapes and define parameters
        self.batch_size = batch['q_o'].shape[0]
        self.seq_len = batch['q_o'].shape[1]
        self.num_particles = num_particles

        # initialise particles --> samples particles randomly according to uniform distribution between state minimums and maximums
        initial_particles = [torch.rand(self.batch_size, self.num_particles, 1) for d in range(self.state_dim)] # generate tensors of shape [batch_size, num_particles, 1] with values between 0 and 1
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
                random_offset = torch.rand(self.batch_size) # generate a tensor of shape [batch_size] filled with random values between 0 and 1
                random_offset *= random_offset * (1 / num_resampled_float) # tensor has shape [batch_size] and is filled with random values between 0 and 1 / num_resampled
                markers = random_offset[:, None] + evenly_spaced_markers[None, :] # broadcast to get shape [batch_size, num_resampled]
                cum_probs = torch.cumsum(particle_probs, dim=1) # particle_probs has shape [batch_size, num_particles] --> take cum_sum along second dimension
                marker_matching = markers[:, :, None] < cum_probs[:, None, :] # broadcasting to create a boolean tensor [batch_size, num_resampled, num_particles], where entry is 'True' if marker < cum prob
                samples = torch.argmax(marker_matching.int(), dim=2).int() # for each batch, each particle to resample, extract index of the first 'True' along the last dimension of 'marker_matching'
                standard_particles = permute_batch(particles, samples) # pick and permute particle samples from 'particles' according to indices specified in 'samples' --> shape [batch_size, sample_size, state_dim]
                standard_particle_probs = torch.ones(self.batch_size, num_resampled) # initialise tensor to store resampled particle probabilities --> shape [batch_size, sample_size]
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
    def __init__(self, proposer_keep_ratio):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Dropout(p=proposer_keep_ratio),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid(), # sigmoid maps to between 0 and 1
        )

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
            nn.Linear(8, 32),
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

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes):
        std = 0.05
        sq_distance = compute_sq_distance(particle_list, batch['q_o'], state_step_sizes) # shape: [batch_size, seq_len, num_particles]
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        activations = particle_prob_list / torch.sqrt(std_tensor) * torch.exp(-sq_distance / (2.0 * std * 2.0)) # shape: [batch_size, seq_len, num_particles]
        loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, -1)))
        return loss
    