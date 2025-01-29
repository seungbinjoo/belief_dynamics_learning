import numpy as np
import tqdm
import math
import torch
import torch.nn as nn
import torch.utils
import matplotlib.pyplot as plt
from belief_dynamics_learning.belief_plotting_2d import *
from belief_dynamics_learning.world2d import *
from belief_dynamics_learning.dpf_utils import *
from belief_dynamics_learning.models import *

class DPF():
    """
    Apply differentiable particle filter (DPF) to current belief of the object pose (represented by particle set)
    to predict the belief at the next time step.
    Args:
        propose_ratio: ratio of proposed to resampled particles
        proposer_keep_ratio: dropout ratio for particle proposer network, used during training and inference to introduce randomness to particle proposer network
        min_obs_likelihood: lower limit for observation likelihood produced by the observation likelihood estimate
        world: object of class World2D
    """
    def __init__(self, propose_ratio, proposer_keep_ratio, min_obs_likelihood, num_particles, num_particles_test, world: World2D, xy_only=False, phi_dataset=True):
        # store hyperparameters
        self.propose_ratio = propose_ratio
        self.proposer_keep_ratio = proposer_keep_ratio
        self.min_obs_likelihood = min_obs_likelihood
        self.world = world
        
        # define more parameters
        if xy_only == False:
            self.state_dim = 3 # x, y, theta for the object pose
        else:
            self.state_dim = 2 # x, y for the object pose
        self.num_particles_float = num_particles
        self.num_particles = int(num_particles)
        self.num_particles_test_float = num_particles_test
        self.num_particles_test = int(num_particles_test)

        # build learnable networks: observation likelihood estimator, particle proposer
        self.build_networks(xy_only, phi_dataset)

        # OLE loss function
        self.OLE_loss_fn = loss_OLE(self)

    def build_networks(self, xy_only, phi_dataset):
        
        # device configuration --> use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        print('Using device:', self.device)

        # PARTICLE PROPOSER --> maps observations and robot poses to particles
        self.particle_proposer = ParticleProposer(self.proposer_keep_ratio, xy_only, phi_dataset).to(self.device)

        # OBSERVATION LIKELIHOOD ESTIMATOR --> maps observations and robot poses to probabilities
        self.obs_like_estimator = ObsLikelihoodEstimator(self.min_obs_likelihood).to(self.device)

        # NEGATIVE PROPOSER --> given a particle, decides whether to keep or discard particle from belief set
        self.negative_proposer = NegativeProposer().to(self.device)

    # compute observation likelihood (probabilities) for a particle set
    def measurement_update(self, phi, particles, means, stds):
        """
        For each particle in the particle-based belief, compute likelihood of the observation.
        Args:
            batch: for one timestep, batches of object pose, robot pose, object pose in robot frame, observation phi, closest point
            particles: for one timestep, shape [batch_size, num_particles, 3]
            means: dictionary {'q_o': [1, 1, 3], 'q_r': [1, 1, 2], ...}
            stds: dictionary {'q_o': [1, 1, 3], 'q_r': [1, 1, 2], ...}
        """
        # prepare input to the observation likelihood estimator network
        phi = phi.float().to(self.device) # shape [batch_size, 1]
        observation_input = torch.tile(phi[:, None, :], (1, particles.shape[1], 1)) # shape [batch_size, num_particles, 1]
        particle_input = self.transform_particles_as_input(particles, means, stds) # shape [batch_size, num_particles, 4]

        input = torch.cat((observation_input, particle_input), dim=-1) # shape [batch_size, num_particles, 5]
        input = input.view(-1, input.shape[-1]).float().to(self.device) # shape [batch_size * num_particles, 5]

        # for each particle, estimate the likelihood based on the observation
        obs_likelihood = self.obs_like_estimator(input) # pass particle set through observaton likelihood estimator network
        obs_likelihood = obs_likelihood.view(phi.shape[0], particles.shape[1])

        return obs_likelihood # shape [batch_size, num_particles]
    
    # normalise 'particles' tensor and split theta state into cos and sin states
    def transform_particles_as_input(self, particles, means, stds):
        
        # ensure means and stds are torch tensors
        means_q_o_r = torch.tensor(means['q_o_r'], device=self.device)
        stds_q_o_r = torch.tensor(stds['q_o_r'], device=self.device)

        # to avoid in place operations which interferes with gradient flow computation?
        particles = particles.clone()

        input = torch.cat((
            (particles[:, :, :2] - means_q_o_r[None, None, :2]) / stds_q_o_r[None, None, :2],
            torch.cos(particles[:, :, 2:3]),
            torch.sin(particles[:, :, 2:3])),
            dim=-1)

        return input
    
    def propose_particles(self, phi, num_particles, state_mins, state_maxs, xy_only):
        
        # output dimension of particle proposer
        if xy_only == False:
            output_dim_pp = 4
        else:
            output_dim_pp = 2

        input_pp = phi # [batch_size, 1]
        duplicated_input_pp = torch.tile(input_pp[:, None, :], (1, num_particles, 1))
        duplicated_input_pp = duplicated_input_pp.view(-1, 1).float().to(self.device) # shape [batch_size * num_particles, 1]
        
        proposed_particles = self.particle_proposer(duplicated_input_pp) # shape [batch_size * num_particles, output_dim_pp]
        proposed_particles = proposed_particles.view(phi.shape[0], num_particles, output_dim_pp) # shape [batch_size, num_particles, output_dim_pp]
        
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
        
        return proposed_particles

    # dpf training process that takes into account resampling when contact events occur
    # note: requires contact_only to be set to False
    def fit(self, data, split_ratio, batch_size, seq_len, num_epochs_pp, num_epochs_ole, num_epochs_np, learning_rate, learning_rate_ole, num_particles, xy_only, phi_dataset, num_contacts_desired, threshold_np):
        
        def get_batch_contacts(batch, num_contacts_desired):
            '''
            Args: batch = {'q_o': [batch_size, 1, 3],
                           'q_r': [batch_size, seq_len, state_dim]
                           ... }
            Output: batch_contacts = {'q_o': [batch_size, 1, 3],
                                      'q_r': [batch_size, num_contacts_desired, state_dim]
                                      ... }
            '''
            # initialise dictionary where batch_contacts will be stored
            batch_contacts = {'q_o': torch.zeros(batch_size, 1, 3),
                              'q_r': torch.zeros(batch_size, num_contacts_desired, 2),
                              'q_o_r': torch.zeros(batch_size, num_contacts_desired, 2),
                              'o': torch.zeros(batch_size, num_contacts_desired, 1),
                              'phi': torch.zeros(batch_size, num_contacts_desired, 1),
                              'cp': torch.zeros(batch_size, num_contacts_desired, 2)}
            
            keys = ['q_r', 'q_o_r', 'o', 'phi', 'cp'] # no need to extract contact events for 'q_o'

            # initialise matrix that will store the indices of contacts
            contact_idx = torch.zeros(batch_size, num_contacts_desired)

            for traj in range(batch_size):
                # extract indices of the contact events
                indices = torch.where(batch['o'][traj, :, 0] == 1)[0]

                # number of indices extracted is determined by num_contacts_desired
                indices = indices[:num_contacts_desired]
                
                # fill in values in batch_contacts using extracted indices
                for key in keys:
                    batch_contacts[key][traj, :, :] = batch[key][traj, indices, :]

                batch_contacts['q_o'] = batch['q_o']
            
            return batch_contacts

        # split data into training and validation set
        train_data, val_data = split_data(data, split_ratio)

        train_dataset = TrajectoriesDataset(train_data, seq_len)
        val_dataset = TrajectoriesDataset(val_data, seq_len)

        # create dataloaders for training and validation set
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # compute some statistics about training data
        means, stds, q_o_r_maxs, q_o_r_mins, q_r_step_sizes, q_r_maxs, q_r_mins = compute_statistics(train_data, phi_dataset)

        # TRAINING THE PARTICLE PROPOSER
        # optimiser and loss
        optimiser_e2e = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
        loss_fn = loss_fn_e2e()

        # initialise variables needed in the training loop
        epoch = 0
        train_loss_list = np.zeros((num_epochs_pp,))
        val_loss_list = np.zeros((num_epochs_pp,))

        # training loop for particle proposer
        print("TRAINING PARTICLE PROPOSER:")
        while epoch < num_epochs_pp:
            # print the epoch number every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs_pp}")

            # training loop: for each epoch, go through multiple batches --> e.g. q_o batch has dimensions [batch_size, seq_len, 3]
            for i, batch in enumerate(train_dataloader):
                # load to gpu 
                batch = {k: v.float().to(self.device) for k, v in batch.items()}

                # convert batch such that it only contains data from contact event timesteps
                batch_contacts = get_batch_contacts(batch, num_contacts_desired) # 'q_o': [batch_size, 1, 3], 'q_r': [batch_size, num_contacts_desired, 2]

                # sequence length is essentially now the number of contact events as this is the number of times we will be applying belief update loop
                seq_len = num_contacts_desired
                
                # initialisation
                particle_list = torch.zeros([batch_size, seq_len, num_particles, self.state_dim], dtype=torch.float64, device=self.device)
                particles = torch.zeros(batch_size, num_particles, self.state_dim)
                phi = (torch.rand(batch_size, 1) * 2 * math.pi) - math.pi # sample random phi from between -pi and pi
                state_mins = torch.from_numpy(q_o_r_mins).to(self.device)
                state_maxs = torch.from_numpy(q_o_r_maxs).to(self.device)
                particles = self.propose_particles(phi, num_particles, state_mins, state_maxs, xy_only) # [batch_size, num_particles, state_dim]
                particles = particles.squeeze(0)
                particle_list[:, 0, :, :] = particles

                particle_prob_list = torch.zeros([batch_size, seq_len, num_particles], dtype=torch.float64, device=self.device)
                particle_probs = torch.zeros(batch_size, num_particles)
                particle_probs = self.measurement_update(phi, particle_list[:, 0, :, :], means, stds) # [batch_size, num_particles] = [20, 100]
                particle_probs = particle_probs.squeeze(0)
                particle_probs = particle_probs / torch.sum(particle_probs, dim=1, keepdim=True) # normalise
                particle_prob_list[:, 0, :] = particle_probs

                # convert particles to world frame
                # particles[:, :, :2] = particles[:, :, :2] + batch_contacts['q_r'][:, 0:1, :2]
                # particle_list[:, 0, :, :] = particles

                # for each time step, i.e. each contact event
                for t in range(seq_len):
                    # determine number of particles to propose and number of particles to resample, based on propose ratio
                    # propose_ratio is ratio of proposed to resampled particles --> this follows an exponential function (gamma)^(t-1)
                    num_proposed_float = torch.round(torch.tensor((self.propose_ratio ** int(t+1))) * self.num_particles_float)
                    num_proposed = num_proposed_float.int()
                    num_resampled_float = self.num_particles_float - num_proposed_float
                    num_resampled = num_resampled_float.int()

                    # as long as propose ratio is less than 1.0, execute resampling and measurement update
                    if self.propose_ratio < 1.0:
                        # stop gradients flowing through resampling step
                        with torch.no_grad():
                            # simple resampling --> for batches
                            resampled_indices = np.zeros((batch_size, num_resampled.item()))

                            for b in range(batch_size):
                                # get the particle probabilities for the current batch item
                                current_probs = particle_probs[b, :]
                                
                                # resample indices for this batch item
                                indices = np.random.choice(np.arange(num_particles), torch.Tensor.numpy(num_resampled), p=torch.Tensor.numpy(current_probs))
                                # append the resampled indices for this batch item
                                resampled_indices[b, :] = indices

                            # resampled particles for batch
                            resampled_particles = torch.zeros(batch_size, num_resampled, 3)

                            for b in range(batch_size):
                                resampled_particles[b, :, :] = particles[b, resampled_indices[b, :], :]

                        # get phi
                        phi = batch_contacts['phi'][:, t, :] # [batch_size, 1]

                        # measurement update
                        resampled_particles[:, :, :2] = resampled_particles[:, :, :2] - batch_contacts['q_r'][:, t:t+1, :] # convert particles to robot frame
                        resampled_particle_probs = self.measurement_update(phi, resampled_particles, means, stds)
                        resampled_particle_probs = resampled_particle_probs.squeeze(0)
                        resampled_particle_probs = resampled_particle_probs / torch.sum(resampled_particle_probs, dim=1, keepdim=True)

                    # as long as propose ratio is greater than 0.0, execute particle proposing step
                    if self.propose_ratio > 0.0:
                        
                        # proposed particles
                        proposed_particles = self.propose_particles(phi, num_proposed, state_mins, state_maxs, xy_only)
                        proposed_particles = proposed_particles.squeeze(0)
                        proposed_particle_probs = torch.ones(batch_size, num_proposed) / num_proposed

                    # combine standard particles (particles that were resampled in the beginning of the loop and went through measurement update) with proposed particles
                    if self.propose_ratio == 1.0:
                        
                        # then all of the proposed particles are used in the new particle set
                        # particles[:, :, :2] = proposed_particles[:, :, :2] + batch_contacts['q_r'][:, t:t+1, :] # convert particles to world frame
                        particle_probs = proposed_particle_probs

                    # if propose_ratio is 0
                    elif self.propose_ratio == 0.0:
                        
                        # then all of the resampled particles are used in the new particle set
                        # particles[:, :, :2] = resampled_particles[:, :, :2] + batch_contacts['q_r'][:, t:t+1, :] # convert particles to world frame
                        particle_probs = resampled_particle_probs

                    # otherwise, the new particle set is a combination of resampled (standard) particles and proposed particles --> propose ratio = ratio of proposed to resampled particles
                    else:
                        
                        # prepare to combine resampled particles and proposed particle probabilities
                        resampled_particle_probs = resampled_particle_probs * (num_resampled_float / self.num_particles_float)
                        proposed_particle_probs = proposed_particle_probs * (num_proposed_float / self.num_particles_float)

                        # combine resampled and proposed particles
                        particles = torch.cat([resampled_particles, proposed_particles], dim=1) # concat in num_particles dimension [batch_size, num_particles, state_dim]
                        # particles[:, :, :2] = particles[:, :, :2] + batch_contacts['q_r'][:, t:t+1, :] # convert particles to world frame
                        particle_probs = torch.cat([resampled_particle_probs, proposed_particle_probs], dim=1) # concat in num_particles dimension [batch_size, num_particles]

                    # normalise probabilites
                    particle_probs = particle_probs / torch.sum(particle_probs, dim=1, keepdim=True)

                    # add particles and particle probabilities from this timestep to the list
                    particle_list[:, t, :, :] = particles
                    particle_prob_list[:, t, :] = particle_probs

                # disable ole network and just have the particle_prob_list be uniform probability every sequence
                particle_prob_list = torch.ones([batch_size, seq_len, num_particles], dtype=torch.float64) / num_particles

                # experiment: compute loss only for final timestep
                batch_contacts['q_o_r'] = batch_contacts['q_o_r'][:, -2:-1, :]

                # compute loss --> experiment: compute loss only using final contact event timestep particle set
                train_loss = loss_fn(batch_contacts, particle_list[:, -2:-1, :, :], particle_prob_list[:, -2:-1, :], q_r_step_sizes, self.world, xy_only)

                # backward and optimize
                optimiser_e2e.zero_grad()
                train_loss.backward()
                optimiser_e2e.step()

            # track training loss at each epoch
            train_loss_list[epoch] = train_loss
            
            # validation loop
            with torch.no_grad():
                # training loop: for each epoch, go through multiple batches --> e.g. q_o batch has dimensions [batch_size, seq_len, 3]
                for i, batch in enumerate(val_dataloader):
                    # load to gpu 
                    batch = {k: v.float().to(self.device) for k, v in batch.items()}

                    # convert batch such that it only contains data from contact event timesteps
                    batch_contacts = get_batch_contacts(batch, num_contacts_desired) # 'q_o': [batch_size, 1, 3], 'q_r': [batch_size, num_contacts_desired, 2]

                    # sequence length is essentially now the number of contact events as this is the number of times we will be applying belief update loop
                    seq_len = num_contacts_desired
                    
                    # initialisation
                    particle_list = torch.zeros([batch_size, seq_len, num_particles, self.state_dim], dtype=torch.float64, device=self.device)
                    particles = torch.zeros(batch_size, num_particles, self.state_dim)
                    phi = (torch.rand(batch_size, 1) * 2 * math.pi) - math.pi # sample random phi from between -pi and pi
                    state_mins = torch.from_numpy(q_o_r_mins).to(self.device)
                    state_maxs = torch.from_numpy(q_o_r_maxs).to(self.device)
                    particles = self.propose_particles(phi, num_particles, state_mins, state_maxs, xy_only) # [batch_size, num_particles, state_dim]
                    particles = particles.squeeze(0)
                    particle_list[:, 0, :, :] = particles

                    particle_prob_list = torch.zeros([batch_size, seq_len, num_particles], dtype=torch.float64, device=self.device)
                    particle_probs = torch.zeros(batch_size, num_particles)
                    particle_probs = self.measurement_update(phi, particle_list[:, 0, :, :], means, stds) # [batch_size, num_particles] = [20, 100]
                    particle_probs = particle_probs.squeeze(0)
                    particle_probs = particle_probs / torch.sum(particle_probs, dim=1, keepdim=True) # normalise
                    particle_prob_list[:, 0, :] = particle_probs

                    # convert particles to world frame
                    particles[:, :, :2] = particles[:, :, :2] + batch_contacts['q_r'][:, 0:1, :2]
                    particle_list[:, 0, :, :] = particles

                    # for each time step, i.e. each contact event
                    for t in range(seq_len):
                        # determine number of particles to propose and number of particles to resample, based on propose ratio
                        # propose_ratio is ratio of proposed to resampled particles --> this follows an exponential function (gamma)^(t-1)
                        num_proposed_float = torch.round(torch.tensor((self.propose_ratio ** int(t+1))) * self.num_particles_float)
                        num_proposed = num_proposed_float.int()
                        num_resampled_float = self.num_particles_float - num_proposed_float
                        num_resampled = num_resampled_float.int()

                        # as long as propose ratio is less than 1.0, execute resampling and measurement update
                        if self.propose_ratio < 1.0:
                            # stop gradients flowing through resampling step
                            with torch.no_grad():
                                # simple resampling --> for batches
                                resampled_indices = np.zeros((batch_size, num_resampled.item()))

                                for b in range(batch_size):
                                    # get the particle probabilities for the current batch item
                                    current_probs = particle_probs[b, :]
                                    
                                    # resample indices for this batch item
                                    indices = np.random.choice(np.arange(num_particles), torch.Tensor.numpy(num_resampled), p=torch.Tensor.numpy(current_probs))
                                    # append the resampled indices for this batch item
                                    resampled_indices[b, :] = indices

                                # resampled particles for batch
                                resampled_particles = torch.zeros(batch_size, num_resampled, 3)

                                for b in range(batch_size):
                                    resampled_particles[b, :, :] = particles[b, resampled_indices[b, :], :]

                            # get phi
                            phi = batch_contacts['phi'][:, t, :] # [batch_size, 1]

                            # measurement update
                            resampled_particles[:, :, :2] = resampled_particles[:, :, :2] - batch_contacts['q_r'][:, t:t+1, :] # convert particles to robot frame
                            resampled_particle_probs = self.measurement_update(phi, resampled_particles, means, stds)
                            resampled_particle_probs = resampled_particle_probs.squeeze(0)
                            resampled_particle_probs = resampled_particle_probs / torch.sum(resampled_particle_probs, dim=1, keepdim=True)

                        # as long as propose ratio is greater than 0.0, execute particle proposing step
                        if self.propose_ratio > 0.0:
                            
                            # proposed particles
                            proposed_particles = self.propose_particles(phi, num_proposed, state_mins, state_maxs, xy_only)
                            proposed_particles = proposed_particles.squeeze(0)
                            proposed_particle_probs = torch.ones(batch_size, num_proposed) / num_proposed

                        # combine standard particles (particles that were resampled in the beginning of the loop and went through measurement update) with proposed particles
                        if self.propose_ratio == 1.0:
                            
                            # then all of the proposed particles are used in the new particle set
                            # particles[:, :, :2] = proposed_particles[:, :, :2] + batch_contacts['q_r'][:, t:t+1, :] # convert particles to world frame
                            particle_probs = proposed_particle_probs

                        # if propose_ratio is 0
                        elif self.propose_ratio == 0.0:
                            
                            # then all of the resampled particles are used in the new particle set
                            # particles[:, :, :2] = resampled_particles[:, :, :2] + batch_contacts['q_r'][:, t:t+1, :] # convert particles to world frame
                            particle_probs = resampled_particle_probs

                        # otherwise, the new particle set is a combination of resampled (standard) particles and proposed particles --> propose ratio = ratio of proposed to resampled particles
                        else:
                            
                            # prepare to combine resampled particles and proposed particle probabilities
                            resampled_particle_probs *= (num_resampled_float / self.num_particles_float)
                            proposed_particle_probs *= (num_proposed_float / self.num_particles_float)

                            # combine resampled and proposed particles
                            particles = torch.cat([resampled_particles, proposed_particles], dim=1) # concat in num_particles dimension [batch_size, num_particles, state_dim]
                            # particles[:, :, :2] = particles[:, :, :2] + batch_contacts['q_r'][:, t:t+1, :] # convert particles to world frame
                            particle_probs = torch.cat([resampled_particle_probs, proposed_particle_probs], dim=1) # concat in num_particles dimension [batch_size, num_particles]

                        # normalise probabilites
                        particle_probs /= torch.sum(particle_probs, dim=1, keepdim=True)

                        # add particles and particle probabilities from this timestep to the list
                        particle_list[:, t, :, :] = particles
                        particle_prob_list[:, t, :] = particle_probs

                    # disable ole network and just have the particle_prob_list be uniform probability every sequence
                    particle_prob_list = torch.ones([batch_size, seq_len, num_particles], dtype=torch.float64) / num_particles

                    # convert particle_list to robot frame
                    particle_list
                    # compute loss
                    val_loss = loss_fn(batch_contacts, particle_list, particle_prob_list, q_r_step_sizes, self.world, xy_only)

                # track training loss at each epoch
                val_loss_list[epoch] = val_loss

            # increment epoch
            epoch += 1

        # plot training and validation loss
        epochs = range(1, num_epochs_pp+1)
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(epochs, train_loss_list, label='Training loss')
        axs[0].plot(epochs, val_loss_list, label='Validation loss')

        # labels and axes
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss over epochs for particle proposer')
        axs[0].legend()

        # TRAINING THE OBSERVATION LIKELIHOOD ESTIMATOR
        optimiser_OLE = torch.optim.Adam(self.obs_like_estimator.parameters(), lr=learning_rate_ole)
        loss_fn = self.OLE_loss_fn

        # initialise variables needed in the training loop
        epoch = 0
        train_loss_list_OLE = np.zeros((num_epochs_ole,))
        val_loss_list_OLE = np.zeros((num_epochs_ole,))

        # training loop for OLE
        print("TRAINING OBSERVATION LIKELIHOOD ESTIMATOR:")
        while epoch < num_epochs_ole:
            # print the epoch number every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs_ole}")

            # training loop: for each epoch, go through multiple batches --> e.g. q_o batch has dimensions [batch_size, seq_len, 3]
            for i, batch in enumerate(train_dataloader):
                # load to gpu 
                batch = {k: v.float().to(self.device) for k, v in batch.items()}

                # convert batch such that it only contains data from contact event timesteps
                batch_contacts_ole = get_batch_contacts(batch, num_contacts_desired) # 'q_o': [batch_size, 1, 3], 'q_r': [batch_size, num_contacts_desired, 2]

                # compute loss
                train_loss_OLE = loss_fn(batch_contacts_ole, means, stds, self.world.obj_dims, self.world.r_robot)

                # backward and optimize
                optimiser_OLE.zero_grad()
                train_loss_OLE.backward()
                optimiser_OLE.step()

            # track training loss at each epoch
            train_loss_list_OLE[epoch] = train_loss_OLE

            # validation loop
            with torch.no_grad():
                val_loss_OLE = 0.0
                for i, batch in enumerate(val_dataloader):
                    # load to gpu 
                    batch = {k: v.float().to(self.device) for k, v in batch.items()}

                    # convert batch such that it only contains data from contact event timesteps
                    batch_contacts_ole = get_batch_contacts(batch, num_contacts_desired) # 'q_o': [batch_size, 1, 3], 'q_r': [batch_size, num_contacts_desired, 2]

                    # sequence length is essentially now the number of contact events as this is the number of times we will be applying belief update loop
                    seq_len = num_contacts_desired

                    # compute validation loss
                    val_loss_OLE = loss_fn(batch_contacts_ole, means, stds, self.world.obj_dims, self.world.r_robot)

                val_loss_list_OLE[epoch] = val_loss_OLE

            # increment epoch
            epoch += 1

        # plot training and validation loss
        epochs = range(1, num_epochs_ole+1)
        axs[1].plot(epochs, train_loss_list_OLE, label='Training loss')
        axs[1].plot(epochs, val_loss_list_OLE, label='Validation loss')

        # labels and axes
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Loss over epochs for observation likelihood estimator')
        axs[1].legend()
        fig.tight_layout()

    # in this method, training loop does not resample --> all the particles in the loop are from the particle proposer (no resampling from the previous particle set)
    # training here is based on one-timestep data
    # only particle proposer is trained here
    def fit_wo_resampling(self, data, split_ratio, batch_size, seq_len, num_epochs_pp, num_epochs_ole, learning_rate, num_particles, xy_only, phi_dataset):
        
        # split data into training and validation set
        train_data, val_data = split_data(data, split_ratio)

        train_dataset = TrajectoriesDataset(train_data, seq_len)
        val_dataset = TrajectoriesDataset(val_data, seq_len)

        # create dataloaders for training and validation set
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # compute some statistics about training data
        means, stds, q_o_r_maxs, q_o_r_mins, q_r_step_sizes, q_r_maxs, q_r_mins = compute_statistics(train_data, phi_dataset)

        # TRAINING THE PARTICLE PROPOSER
        # optimiser and loss
        optimiser_e2e = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)
        loss_fn = loss_fn_e2e()

        # initialise variables needed in the training loop
        epoch = 0
        train_loss_list = np.zeros((num_epochs_pp,))
        val_loss_list = np.zeros((num_epochs_pp,))

        print('TRAINING PARTICLE PROPOSER')
        # training loop for particle proposer
        while epoch < num_epochs_pp:
            # print the epoch number every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs_pp}")

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
                            'q_o_r': batch['q_o_r'][:, t, :],
                            'phi': batch['phi'][:, t, :]}

                    # data for one timestep                
                    q_o = batch_t['q_o'].squeeze(1) # shape [batch_size, 3]
                    q_r_achieved = batch_t['q_r'].squeeze(1) # shape [batch_size, 2]
                    q_o_r = batch_t['q_o_r'].squeeze(1) # shape [batch_size, 2]
                    phi = batch_t['phi'] # shape [batch_size, 1]
                    
                    # propose particles
                    proposed_particles = self.propose_particles(phi, num_particles, q_o_r_mins, q_o_r_maxs, xy_only) # shape [batch_size, num_particles, 3]
                    particle_list[:, t, :, :] = proposed_particles

                # disable ole network and just have the particle_prob_list be uniform probability every sequence
                particle_prob_list = torch.ones([batch_size, seq_len, num_particles], dtype=torch.float64) / num_particles

                # compute loss
                train_loss = loss_fn(batch, particle_list, particle_prob_list, q_r_step_sizes, self.world, xy_only)

                # backward and optimize
                optimiser_e2e.zero_grad()
                train_loss.backward()
                optimiser_e2e.step()

            # track training loss at each epoch
            train_loss_list[epoch] = train_loss

            # validation loop
            with torch.no_grad():
                val_loss = 0.0
                for i, batch in enumerate(val_dataloader):
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
                                'q_o_r': batch['q_o_r'][:, t, :],
                                'phi': batch['phi'][:, t, :]}

                        # data for one timestep                
                        q_o = batch_t['q_o'].squeeze(1) # shape [batch_size, 3]
                        q_r_achieved = batch_t['q_r'].squeeze(1) # shape [batch_size, 2]
                        q_o_r = batch_t['q_o_r'].squeeze(1) # shape [batch_size, 2]
                        phi = batch_t['phi'] # shape [batch_size, 1]
                        
                        # propose particles
                        proposed_particles = self.propose_particles(phi, num_particles, q_o_r_mins, q_o_r_maxs, xy_only) # shape [batch_size, num_particles, 3]
                        particle_list[:, t, :, :] = proposed_particles

                    # disable ole network and just have the particle_prob_list be uniform probability every sequence
                    particle_prob_list = torch.ones([batch_size, seq_len, num_particles], dtype=torch.float64) / num_particles

                    # compute loss
                    val_loss = loss_fn(batch, particle_list, particle_prob_list, q_r_step_sizes, self.world, xy_only)

                val_loss_list[epoch] = val_loss

            # increment epoch
            epoch += 1

        # plot training and validation loss
        epochs = range(1, num_epochs_pp+1)
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(epochs, train_loss_list, label='Training loss')
        axs[0].plot(epochs, val_loss_list, label='Validation loss')

        # labels and axes
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss over epochs for particle proposer')
        axs[0].legend()

        # return most recent batch of proposed particles (for GMM plotting)
        return q_o.detach().numpy(), particle_list.detach().numpy(), particle_prob_list.detach().numpy()

    # testing particle update given an observation --> single timestep belief update performance testing
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

            # extract indices for contact events
            contact_idx_new = np.where(o_hist==1)[0]

            # plot sample trajectory
            self.world.plot_rollout(q_r_hist, o_hist)

            # for the first contact event, calculate phi observation
            d, closest_point = dist_to_object(q_r_hist[contact_idx_new[0], :], self.world.qo_gt, self.world.obj_dims, self.world.r_robot)
            self.world.q_r_0 = q_r_hist[contact_idx_new[0], :]
            closest_point = closest_point - self.world.q_r_0
            observation_axis = closest_point
            fixed_axis = [0, 1] # vertical axis in robot frame
            phi, phi_deg = angle_between_vectors(observation_axis, fixed_axis)
            phi_noise_sigma = 0.1
            phi = torch.from_numpy(np.array([phi])) + np.random.normal(loc=0.0, scale=phi_noise_sigma, size=1)

            # using phi observation from the first contact event, propose particles
            q_r_desired = torch.from_numpy(a_hist[contact_idx_new[0]-1, :])
            q_r_achieved = torch.from_numpy(q_r_hist[contact_idx_new[0], :])
            print('Desired robot position during contact event:', q_r_desired[None, :])
            print('Achieved robot position during contact event:', q_r_achieved[None, :])
            print('Phi during contact event:', phi[None, :])
            state_mins = torch.from_numpy(state_mins).to(self.device)
            state_maxs = torch.from_numpy(state_maxs).to(self.device)
            self.world.particles = self.propose_particles(phi[None, :], self.world.num_particles, state_mins, state_maxs, xy_only) # shape [1, num_particles, 3]
            
            # store particles and robot pose in 2D world object
            self.world.particles = self.world.particles.squeeze(0) # [num_particles, 3]
            self.world.q_r_0 = q_r_hist[contact_idx_new[0], :]

            # use particles to perform measurement update
            self.world.weights = self.measurement_update(phi[None, :], self.world.particles[None, :, :], means, stds) # [batch_size, num_particles] = [1, 100]
            self.world.weights = self.world.weights.squeeze() # [num_particles]
            self.world.weights = self.world.weights / torch.sum(self.world.weights) # normalise

            # get particle object position in world frame
            self.world.particles[:, :2] = self.world.particles[:, :2] + q_r_achieved[None, :]

            # plot belief
            self.world.plot_belief_with_obs(q_r_desired, q_r_achieved)

            # plot visualisation for particle weights
            self.plot_particle_weights(self.world.particles, self.world.weights, q_r_achieved, s=5)

            # test observation likelihood estimator by plotting observation likelihood surface
            num_particles_per_axis = 100
            particles = torch.zeros(num_particles_per_axis ** 2, 3)
            particles[:, -1] = torch.rand(num_particles_per_axis ** 2) * (2*torch.pi) - torch.pi # let particles have random angle
            particles = particles.view(num_particles_per_axis, num_particles_per_axis, 3)
            x_axis_ticks = torch.linspace(-0.5, 0.5, num_particles_per_axis) - self.world.q_r_0[0]
            y_axis_ticks = torch.linspace(0.5, 1.5, num_particles_per_axis) - self.world.q_r_0[1]
            for i in range(num_particles_per_axis):
                for j in range(num_particles_per_axis):
                    particles[i, j, :2] = torch.tensor([x_axis_ticks[i], y_axis_ticks[j]])
            particles = particles.view(num_particles_per_axis ** 2, 3)
            weights = self.measurement_update(phi[None, :], particles[None, :, :], means, stds)
            weights = weights / torch.sum(weights)
            particles[:, :2] = particles[:, :2] + self.world.q_r_0[None, :]
            self.plot_particle_weights(particles, weights, q_r_achieved, s=10)

    # testing particle update and resampling with non-contact-only trajectories
    # that is trajectories can have timesteps that contain or don't contain contact observations
    # note: every time there is a contact event, resampling is executed
    def test_resampling_non_contact_only(self, qo_gt, q_r_hist, o_hist, phi_hist, state_mins, state_maxs, means, stds, xy_only, compute_metrics):
        
        def get_phi(q_r_hist, contact_idx, n):
            # for the first contact event, calculate phi observation
            d, closest_point = dist_to_object(q_r_hist[contact_idx[n], :], self.world.qo_gt, self.world.obj_dims, self.world.r_robot)
            self.world.q_r_0 = q_r_hist[contact_idx[n], :]
            closest_point = closest_point - self.world.q_r_0
            observation_axis = closest_point
            fixed_axis = [0, 1] # vertical axis in robot frame
            phi, phi_deg = angle_between_vectors(observation_axis, fixed_axis)
            phi = torch.from_numpy(np.array([phi]))
            phi_noise_sigma = 0.1
            phi = phi + np.random.normal(loc=0.0, scale=phi_noise_sigma, size=1)
            return phi

        with torch.no_grad():
            # USE TEST DATASET
            self.world.qo_gt = qo_gt
            observation_count = int(np.sum(o_hist))

            # only keep bits of sequence around positive contact measurements --> truncate trajectory
            contact_idx = np.where(o_hist==1)[0]
            min_idx = np.min(contact_idx)
            max_idx = np.max(contact_idx)
            keep_idx = [min_idx, max_idx]
            q_r_hist = q_r_hist[keep_idx[0]:keep_idx[1]+1]
            # a_hist = a_hist[keep_idx[0]:keep_idx[1]+1]
            o_hist = o_hist[keep_idx[0]:keep_idx[1]+1]
            phi_hist = phi_hist[keep_idx[0]:keep_idx[1]+1]
            
            # extract indices for contact events for truncated trajectory
            contact_idx = np.where(o_hist==1)[0]

            # slice phi_hist so that it only contains contact events
            phi_hist = phi_hist[contact_idx, :]

            # plot sample trajectory
            if compute_metrics == False:
                self.world.plot_rollout(q_r_hist, o_hist)
            else:
                pass

            # initialisation
            particle_list = torch.zeros(observation_count, self.num_particles_test, self.state_dim)
            particles = torch.zeros(self.num_particles_test, self.state_dim)
            # phi = get_phi(q_r_hist, contact_idx, n=0)
            phi = torch.from_numpy(phi_hist[0, :])
            state_mins = torch.from_numpy(state_mins).to(self.device)
            state_maxs = torch.from_numpy(state_maxs).to(self.device)
            particles = self.propose_particles(phi[None, :], self.num_particles_test, state_mins, state_maxs, xy_only)
            particles = particles.squeeze(0)
            particle_list[0, :, :] = particles

            particle_prob_list = torch.zeros(observation_count, self.num_particles_test)
            particle_probs = torch.zeros(self.num_particles_test)
            particle_probs[None, :] = self.measurement_update(phi[None, :], particle_list[0:1, :, :], means, stds) # [batch_size, num_particles_test] = [1, 1000]
            particle_probs = particle_probs.squeeze(0)
            particle_probs = particle_probs / torch.sum(particle_probs) # normalise
            particle_prob_list[0, :] = particle_probs

            # convert particles to world frame
            particles[:, :2] = particles[:, :2] + q_r_hist[contact_idx[0], :]
            particle_list[0, :, :] = particles

            # BELIEF UPDATE LOOP
            for i in range(1, observation_count):
                
                # determine number of particles to propose and number of particles to resample, based on propose ratio
                # propose_ratio is ratio of proposed to resampled particles --> this follows an exponential function (gamma)^(t-1)
                num_proposed_float = torch.round(torch.tensor((self.propose_ratio ** int(i))) * self.num_particles_test_float)
                num_proposed = num_proposed_float.int()
                num_resampled_float = self.num_particles_test_float - num_proposed_float
                num_resampled = num_resampled_float.int()

                # as long as propose ratio is less than 1.0, execute resampling and measurement update
                if self.propose_ratio < 1.0:
                    # simple resampling
                    resampled_indices = np.random.choice(np.arange(self.num_particles_test), torch.Tensor.numpy(num_resampled), p=torch.Tensor.numpy(particle_probs))
                    resampled_particles = particles[resampled_indices, :]

                    # get phi
                    # phi = get_phi(q_r_hist, contact_idx, n=i)
                    phi = torch.from_numpy(phi_hist[i, :])

                    # measurement update
                    resampled_particles[:, :2] = resampled_particles[:, :2] - q_r_hist[contact_idx[i], :] # convert particles to robot frame
                    resampled_particle_probs = self.measurement_update(phi[None, :], resampled_particles[None, :, :], means, stds)
                    resampled_particle_probs = resampled_particle_probs.squeeze(0)
                    resampled_particle_probs = resampled_particle_probs / torch.sum(resampled_particle_probs)

                # as long as propose ratio is greater than 0.0, execute particle proposing step
                if self.propose_ratio > 0.0:
                    
                    # proposed particles
                    proposed_particles = self.propose_particles(phi[None, :], num_proposed, state_mins, state_maxs, xy_only)
                    proposed_particles = proposed_particles.squeeze(0)
                    proposed_particle_probs = torch.ones(num_proposed) / num_proposed

                # combine standard particles (particles that were resampled in the beginning of the loop and went through measurement update) with proposed particles
                if self.propose_ratio == 1.0:
                    
                    # then all of the proposed particles are used in the new particle set
                    particles[:, :2] = proposed_particles[:, :2] + q_r_hist[contact_idx[i], :] # convert particles to world frame
                    particle_probs = proposed_particle_probs

                # if propose_ratio is 0
                elif self.propose_ratio == 0.0:
                    
                    # then all of the resampled particles are used in the new particle set
                    particles[:, :2] = resampled_particles[:, :2] + q_r_hist[contact_idx[i], :] # convert particles to world frame
                    particle_probs = resampled_particle_probs

                # otherwise, the new particle set is a combination of resampled (standard) particles and proposed particles --> propose ratio = ratio of proposed to resampled particles
                else:
                    
                    # prepare to combine resampled particles and proposed particle probabilities
                    resampled_particle_probs *= (num_resampled_float / self.num_particles_test_float)
                    proposed_particle_probs *= (num_proposed_float / self.num_particles_test_float)

                    # combine resampled and proposed particles
                    particles = torch.cat([resampled_particles, proposed_particles], dim=0)
                    particles[:, :2] = particles[:, :2] + q_r_hist[contact_idx[i], :] # convert particles to world frame
                    particle_probs = torch.cat([resampled_particle_probs, proposed_particle_probs], dim=0)

                # normalise probabilites
                particle_probs /= torch.sum(particle_probs, dim=0, keepdim=True)

                # add particles and particle probabilities from this timestep to the list
                particle_list[i, :, :] = particles
                particle_prob_list[i, :] = particle_probs

            if compute_metrics == False:
                # plot belief update
                cols = 3
                rows = (np.ceil(observation_count / cols)).astype(int)
                fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))

                # flatten the axs array to make indexing easier
                axs = axs.flatten()

                for i in range(observation_count):
                    
                    # track these values so that we can differentiate when plotting proposed particles vs resampled particles
                    num_proposed_float = torch.round(torch.tensor((self.propose_ratio ** int(i))) * self.num_particles_test_float)
                    num_proposed = num_proposed_float.int()
                    num_resampled_float = self.num_particles_test_float - num_proposed_float
                    num_resampled = num_resampled_float.int()
                    
                    # for plotting intermediate trajectory --> portion of truncated trajectory to plot
                    if i > 0:
                        q_r_hist_plot = q_r_hist[contact_idx[i-1]:contact_idx[i], :]
                        # a_hist_plot = a_hist[contact_idx[i-1]:contact_idx[i], :]
                        o_hist_plot = o_hist[contact_idx[i-1]:contact_idx[i]]

                    # get robot position at particlular contact event
                    self.world.q_r_0 = q_r_hist[contact_idx[i]]

                    # plotting
                    axs[i].set_aspect('equal')
                    axs[i].set_xlim(self.world.bounds[0])
                    axs[i].set_ylim(self.world.bounds[1])
                    axs[i].set_title("Contact event " + str(i+1))
                    plot_robot(axs[i], self.world.q_r_0, self.world.r_robot, color=robot_color)
                    plot_object(axs[i], self.world.qo_gt, self.world.obj_dims, color=gt_color)

                    # proportion of the total particles that we want to actually plot
                    ratio_to_plot = 0.2

                    if i == 0:
                        plot_object_belief(axs[i], particle_list[i, :int(np.round(self.num_particles_test*ratio_to_plot)), :], particle_prob_list[i, :int(np.round(self.num_particles_test*ratio_to_plot))], self.world.obj_dims, particle_color='blue')
                    else:
                        plot_object_belief(axs[i], particle_list[i, num_proposed:num_proposed+(int(np.round(num_proposed*ratio_to_plot))), :], particle_prob_list[i, num_proposed:num_proposed+(int(np.round(num_proposed*ratio_to_plot)))], self.world.obj_dims, particle_color='blue')
                        plot_object_belief(axs[i], particle_list[i, 0:int(np.round(num_resampled*ratio_to_plot)), :], particle_prob_list[i, 0:int(np.round(num_resampled*ratio_to_plot))], self.world.obj_dims, particle_color='green')

                    # plotting intermediate trajectory
                    if i > 0:
                        axs[i].plot(q_r_hist_plot[:, 0], q_r_hist_plot[:, 1], 'o-', color=robot_color, alpha=0.5)

                    # for last contact event / plot
                    if i == (observation_count - 1):
                        # plot the particle with the highest probability
                        max_idx = torch.argmax(particle_prob_list[i, :])
                        max_value = particle_prob_list[i, max_idx:max_idx+1]
                        max_prob_particle = particle_list[i, max_idx:max_idx+1, :]
                        plot_object_belief(axs[i], max_prob_particle, torch.tensor([1]), self.world.obj_dims, particle_color='orange')
                        
                        # take the particle set and average x, y, theta across the particle set --> plot the average
                        avg_particle = torch.mean(particle_list[i, :, :], dim=0)
                        plot_object_belief(axs[i], avg_particle[None, :], torch.tensor([1]), self.world.obj_dims, particle_color='yellow')

                # remove any unused subplots
                for j in range(observation_count, rows * cols):
                    fig.delaxes(axs[j])
                
                fig.tight_layout()
            
            else:
                pass

        if compute_metrics == True:
            # consider final timestep in belief update loop
            i = observation_count - 1

            # obtain ground truth object state
            q_o = self.world.qo_gt

            # obtain particle with max probability
            max_idx = torch.argmax(particle_prob_list[i, :])
            max_value = particle_prob_list[i, max_idx:max_idx+1]
            max_prob_particle = particle_list[i, max_idx:max_idx+1, :] # particle is in world frame
            max_prob_particle = max_prob_particle.squeeze(0)
            max_prob_particle[-1] = max_prob_particle[-1] % 3.14159 # make sure angle state is between 0 and pi

            # squared error based on particle with max probability
            max_particle_squared_error_xy, max_particle_squared_error_theta = compute_squared_error(q_o, max_prob_particle)

            # obtain average particle of the particle set
            avg_particle = torch.mean(particle_list[i, :, :], dim=0)
            avg_particle = avg_particle.squeeze(0) # particle is in world frame
            avg_particle[-1] = avg_particle[-1] % 3.14159 # make sure angle state is between 0 and pi

            # squared error based on average particle
            avg_particle_squared_error_xy, avg_particle_squared_error_theta = compute_squared_error(q_o, avg_particle)

            return max_particle_squared_error_xy, max_particle_squared_error_theta, avg_particle_squared_error_xy, avg_particle_squared_error_theta
        
        else:
            pass
    
    # tesing particle update and resampling with trajectories that are not contact-only
    # every time there is a contact event, resampling is executed
    # here, we test also with negative proposing during steps that have no contact events
    def test_with_NP(self, qo_gt, q_r_hist, o_hist, phi_hist, state_mins, state_maxs, means, stds, xy_only, compute_metrics, threshold_np, margin_np):
        with torch.no_grad():
            
            # sample new object and robot pose
            self.world.sample_gt_object_pose()
            self.world.sample_robot_pose()

            # GENERATE TRAJECTORY
            # perform rollouts of length 100, until a trajectory with desired number of observations is found
            # observation_count = 0
            # desired_observation_count = 10 # desired minimum
            # while observation_count < desired_observation_count:
            #     q_r_hist, o_hist, a_hist = self.world.rollout(1000, self.world.qo_gt.copy())
            #     observation_count = np.sum(o_hist)

            # USE TEST DATASET
            self.world.qo_gt = qo_gt
            observation_count = int(np.sum(o_hist))

            # only keep bits of sequence around positive contact measurements --> truncate trajectory
            contact_idx = np.where(o_hist==1)[0]
            min_idx = np.min(contact_idx)
            max_idx = np.max(contact_idx)
            keep_idx = [min_idx, max_idx]
            q_r_hist = q_r_hist[keep_idx[0]:keep_idx[1]+1]
            # a_hist = a_hist[keep_idx[0]:keep_idx[1]+1]
            o_hist = o_hist[keep_idx[0]:keep_idx[1]+1]
            phi_hist = phi_hist[keep_idx[0]:keep_idx[1]+1]
            
            # extract indices for contact events for truncated trajectory
            contact_idx = np.where(o_hist==1)[0]

            # slice phi_hist so that it only contains contact events
            phi_hist = phi_hist[contact_idx, :]

            # plot sample trajectory
            if compute_metrics == False:
                self.world.plot_rollout(q_r_hist, o_hist)
            else:
                pass

            # compute length of the subsequence, which was extracted
            sub_seq_len = int(q_r_hist.shape[0])

            # initialisation
            particle_np_hist = torch.zeros(1, 3)
            particle_list = torch.zeros(sub_seq_len, self.num_particles_test, self.state_dim)
            particles = torch.zeros(self.num_particles_test, self.state_dim)
            # phi = get_phi(q_r_hist, contact_idx, n=0)
            phi = torch.from_numpy(phi_hist[0, :])
            state_mins = torch.from_numpy(state_mins).to(self.device)
            state_maxs = torch.from_numpy(state_maxs).to(self.device)
            particles = self.propose_particles(phi[None, :], self.num_particles_test, state_mins, state_maxs, xy_only)
            particles = particles.squeeze(0)
            particle_list[0, :, :] = particles

            particle_prob_list = torch.zeros(sub_seq_len, self.num_particles_test)
            particle_probs = torch.zeros(self.num_particles_test)
            particle_probs[None, :] = self.measurement_update(phi[None, :], particle_list[0:1, :, :], means, stds) # [batch_size, num_particles_test] = [1, 1000]
            particle_probs = particle_probs.squeeze(0)
            particle_probs = particle_probs / torch.sum(particle_probs) # normalise
            particle_prob_list[0, :] = particle_probs

            # convert particles to world frame
            particles[:, :2] = particles[:, :2] + q_r_hist[contact_idx[0], :]
            particle_list[0, :, :] = particles
            
            # keep track of contact event number
            t_contact = 0

            # for each time step
            for t in range(sub_seq_len):
                # if there is contact, use particle proposer and measurement update
                if o_hist[t] == 1:
                    # make sure 'particles' and 'particle_probs' is updated to the most recent particle set
                    if t != 0:
                        particles = particle_list[t-1, ...]
                        particle_probs = particle_prob_list[t-1, ...]

                    # determine number of particles to propose and number of particles to resample, based on propose ratio
                    # propose_ratio is ratio of proposed to resampled particles --> this follows an exponential function (gamma)^(t-1)
                    num_proposed_float = torch.round(torch.tensor((self.propose_ratio ** int(t_contact+1))) * self.num_particles_test_float)
                    num_proposed = num_proposed_float.int()
                    num_resampled_float = self.num_particles_test_float - num_proposed_float
                    num_resampled = num_resampled_float.int()

                    # as long as propose ratio is less than 1.0, execute resampling and measurement update
                    if self.propose_ratio < 1.0:
                        # simple resampling
                        resampled_indices = np.random.choice(np.arange(self.num_particles_test), torch.Tensor.numpy(num_resampled), p=torch.Tensor.numpy(particle_probs))
                        resampled_particles = particles[resampled_indices, :]

                        # get phi
                        # phi = get_phi(q_r_hist, contact_idx, n=t_contact)
                        phi = torch.from_numpy(phi_hist[int(t_contact), :])

                        # measurement update
                        resampled_particles[:, :2] = resampled_particles[:, :2] - q_r_hist[contact_idx[t_contact], :] # convert particles to robot frame
                        resampled_particle_probs = self.measurement_update(phi[None, :], resampled_particles[None, :, :], means, stds)
                        resampled_particle_probs = resampled_particle_probs.squeeze(0)
                        resampled_particle_probs = resampled_particle_probs / torch.sum(resampled_particle_probs)

                    # as long as propose ratio is greater than 0.0, execute particle proposing step
                    if self.propose_ratio > 0.0:
                        
                        # proposed particles
                        proposed_particles = self.propose_particles(phi[None, :], num_proposed, state_mins, state_maxs, xy_only)
                        proposed_particles = proposed_particles.squeeze(0)
                        proposed_particle_probs = torch.ones(num_proposed) / num_proposed

                    # combine standard particles (particles that were resampled in the beginning of the loop and went through measurement update) with proposed particles
                    if self.propose_ratio == 1.0:
                        
                        # then all of the proposed particles are used in the new particle set
                        particles[:, :2] = proposed_particles[:, :2] + q_r_hist[contact_idx[t_contact], :] # convert particles to world frame
                        particle_probs = proposed_particle_probs

                    # if propose_ratio is 0
                    elif self.propose_ratio == 0.0:
                        
                        # then all of the resampled particles are used in the new particle set
                        particles[:, :2] = resampled_particles[:, :2] + q_r_hist[contact_idx[t_contact], :] # convert particles to world frame
                        particle_probs = resampled_particle_probs

                    # otherwise, the new particle set is a combination of resampled (standard) particles and proposed particles --> propose ratio = ratio of proposed to resampled particles
                    else:
                        
                        # prepare to combine resampled particles and proposed particle probabilities
                        resampled_particle_probs *= (num_resampled_float / self.num_particles_test_float)
                        proposed_particle_probs *= (num_proposed_float / self.num_particles_test_float)

                        # combine resampled and proposed particles
                        particles = torch.cat([resampled_particles, proposed_particles], dim=0)
                        particles[:, :2] = particles[:, :2] + q_r_hist[contact_idx[t_contact], :] # convert particles to world frame
                        particle_probs = torch.cat([resampled_particle_probs, proposed_particle_probs], dim=0)

                    # normalise probabilites
                    particle_probs /= torch.sum(particle_probs, dim=0, keepdim=True)

                    # add particles and particle probabilities from this timestep to the list
                    particle_list[t, :, :] = particles
                    particle_prob_list[t, :] = particle_probs

                    # increment contact event number
                    t_contact = t_contact + 1

                # if there is no contact, perform negative proposing (hard-coded)
                else:
                    # only activate negative proposer if robot is sufficiently close to the particle set
                    q_o_particles = particle_list[t-1, :, :2] # particle pose in world frame
                    q_r = q_r_hist[t, None, :] # robot position at time t
                    q_o_r = q_o_particles - q_r # [num_particles, 2]
                    x_dist_min = torch.min(q_o_r[:, 0])
                    y_dist_min = torch.min(q_o_r[:, 1])
                    dist_min = torch.sqrt((x_dist_min ** 2.0) + (y_dist_min ** 2.0)).item()

                    if dist_min < np.sqrt((self.world.obj_dims[0] ** 2.0) + (self.world.obj_dims[1] ** 2.0)).item() + self.world.r_robot + 0.05:
                    
                        # for every particle in the particle set, decide whether to keep or eradicate
                        for p in range(self.num_particles_test):
                            # check that particle and robot are in intersection
                            q_o_particle = particle_list[t-1, p, :]
                            d, _ = dist_to_object(q_r_hist[t], torch.Tensor.numpy(q_o_particle), self.world.obj_dims, self.world.r_robot)

                            if d < 0 - margin_np:

                                # sample number between 0 and 1 from uniform distribution
                                n = np.random.uniform(0, 1)

                                # if n is larger than some threshold --> get rid of particle
                                if n > threshold_np:
                                    # keep record particle negative proposing history, i.e. particles that are eradicated
                                    eradicated_particle = particle_list[t, p, :]
                                    particle_np_hist = torch.cat((particle_np_hist, eradicated_particle[None, :]), dim=0)

                                    # exclude current particle
                                    particle_list_exclude = torch.cat((particle_list[t-1, :p, :], particle_list[t-1, p+1:, :]), dim=0)
                                    particle_prob_list_exclude = torch.cat((particle_prob_list[t-1, :p], particle_prob_list[t-1, p+1:]), dim=0)

                                    # # 1. new particle is average from this particle set
                                    # new_particle = torch.mean(particle_list_exclude, dim=0)
                                    # new_particle = new_particle[None, :]
                                    # new_particle_prob = torch.mean(particle_prob_list_exclude, dim=0)
                                    # new_particle_prob = new_particle_prob[None]

                                    # 2. new particle is proposed from the most recent contact event information
                                    # phi = get_phi(q_r_hist, contact_idx, n=t_contact)
                                    phi = torch.from_numpy(phi_hist[int(t_contact), :])
                                    new_particle = self.propose_particles(phi[None, :], 1, state_mins, state_maxs, xy_only)
                                    new_particle = new_particle.squeeze(0)
                                    new_particle[:, :2] = new_particle[:, :2] + q_r_hist[t, :]
                                    # set new particle prob
                                    new_particle_prob = torch.tensor([1]) / self.num_particles_test

                                    # insert new particle into particle set
                                    particle_list[t, :, :] = torch.cat((particle_list_exclude[:p, :], new_particle, particle_list_exclude[p:, :]), dim=0)
                                    particle_prob_list[t, :] = torch.cat((particle_prob_list_exclude[:p], new_particle_prob, particle_prob_list_exclude[p:]), dim=0)

                                    # normalise probabilities
                                    particle_prob_list[t, :] = particle_prob_list[t, :] / torch.sum(particle_prob_list[t, :])

                                # if n is below threshold, do nothing
                                else:
                                    pass
                            else:
                                particle_list[t, ...] = particle_list[t-1, ...]
                                particle_prob_list[t, ...] = particle_prob_list[t-1, ...]
                    else:
                        # current particle set is the same as the previous timestep's particle set
                        particle_list[t, ...] = particle_list[t-1, ...]
                        particle_prob_list[t, ...] = particle_prob_list[t-1, ...]

            # if we dont have to compute metrics, show the particle belief update plots
            if compute_metrics == False:
                # plot belief update
                cols = 3
                rows = (np.ceil(observation_count / cols)).astype(int)
                fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))

                # flatten the axs array to make indexing easier
                axs = axs.flatten()

                for i in range(observation_count):
                    
                    # track these values so that we can differentiate when plotting proposed particles vs resampled particles
                    num_proposed_float = torch.round(torch.tensor((self.propose_ratio ** int(i))) * self.num_particles_test_float)
                    num_proposed = num_proposed_float.int()
                    num_resampled_float = self.num_particles_test_float - num_proposed_float
                    num_resampled = num_resampled_float.int()
                    
                    # for plotting intermediate trajectory --> portion of truncated trajectory to plot
                    if i > 0:
                        q_r_hist_plot = q_r_hist[contact_idx[i-1]:contact_idx[i], :]
                        # a_hist_plot = a_hist[contact_idx[i-1]:contact_idx[i], :]
                        o_hist_plot = o_hist[contact_idx[i-1]:contact_idx[i]]

                    # get robot position at particlular contact event
                    self.world.q_r_0 = q_r_hist[contact_idx[i]]

                    # plotting
                    axs[i].set_aspect('equal')
                    axs[i].set_xlim(self.world.bounds[0])
                    axs[i].set_ylim(self.world.bounds[1])
                    axs[i].set_title("Contact event " + str(i+1), fontsize=16)
                    # bounds = np.array([[-0.3, 0.2], [0.9, 1.4]])
                    # axs[i].set_xlim(bounds[0])
                    # axs[i].set_ylim(bounds[1])
                    plot_robot(axs[i], self.world.q_r_0, self.world.r_robot, color=robot_color)
                    plot_object(axs[i], self.world.qo_gt, self.world.obj_dims, color=gt_color)

                    # proportion of the total particles that we want to actually plot
                    ratio_to_plot = 0.2
                    if i == 0:
                        plot_object_belief(axs[i], particle_list[contact_idx[i], :int(np.round(self.num_particles_test*ratio_to_plot)), :], particle_prob_list[contact_idx[i], :int(np.round(self.num_particles_test*ratio_to_plot))], self.world.obj_dims, particle_color='blue')
                    else:
                        plot_object_belief(axs[i], particle_list[contact_idx[i], num_proposed:num_proposed+(int(np.round(num_proposed*ratio_to_plot))), :], particle_prob_list[contact_idx[i], num_proposed:num_proposed+(int(np.round(num_proposed*ratio_to_plot)))], self.world.obj_dims, particle_color='blue')
                        plot_object_belief(axs[i], particle_list[contact_idx[i], 0:int(np.round(num_resampled*ratio_to_plot)), :], particle_prob_list[contact_idx[i], 0:int(np.round(num_resampled*ratio_to_plot))], self.world.obj_dims, particle_color='green')

                    # plotting intermediate trajectory
                    if i > 0:
                        axs[i].plot(q_r_hist_plot[:, 0], q_r_hist_plot[:, 1], 'o-', color=robot_color, alpha=0.5)

                    # for last contact event / plot
                    if i == (observation_count - 1):
                        # plot the particle with the highest probability
                        max_idx = torch.argmax(particle_prob_list[contact_idx[i], :])
                        max_value = particle_prob_list[contact_idx[i], max_idx:max_idx+1]
                        max_prob_particle = particle_list[contact_idx[i], max_idx:max_idx+1, :]
                        # plot_object_belief(axs[i], max_prob_particle, torch.tensor([1]), self.world.obj_dims, particle_color='orange')
                        
                        # take the particle set and average x, y, theta across the particle set --> plot the average
                        avg_particle = torch.mean(particle_list[contact_idx[i], :, :], dim=0)
                        plot_object_belief(axs[i], avg_particle[None, :], torch.tensor([1]), self.world.obj_dims, particle_color='yellow')

                        # plot particle which have been eradicated (plot with very low opacity)
                        num_eradicated_plot = 10
                        # plot_object_belief(axs[i], particle_np_hist[-num_eradicated_plot:-1, :], torch.ones(num_eradicated_plot)/100000, self.world.obj_dims, particle_color='black')
                        print('Total number of particles eradicated throughout trajectory: ', particle_np_hist.shape[0])

                # remove any unused subplots
                for j in range(observation_count, rows * cols):
                    fig.delaxes(axs[j])
                
                fig.tight_layout()
            
            # if we do have to compute metrics
            else:
                # consider final timestep in belief update loop
                # obtain ground truth object state
                q_o = self.world.qo_gt

                # obtain particle with max probability
                max_idx = torch.argmax(particle_prob_list[t, :])
                max_value = particle_prob_list[t, max_idx:max_idx+1]
                max_prob_particle = particle_list[t, max_idx:max_idx+1, :] # particle is in world frame
                max_prob_particle = max_prob_particle.squeeze(0)
                max_prob_particle[-1] = max_prob_particle[-1] % 3.14159 # make sure angle state is between 0 and pi

                # squared error based on particle with max probability
                max_particle_squared_error_xy, max_particle_squared_error_theta = compute_squared_error(q_o, max_prob_particle)

                # obtain average particle of the particle set
                avg_particle = torch.mean(particle_list[t, :, :], dim=0)
                avg_particle = avg_particle.squeeze(0) # particle is in world frame
                avg_particle[-1] = avg_particle[-1] % 3.14159 # make sure angle state is between 0 and pi

                # squared error based on average particle
                avg_particle_squared_error_xy, avg_particle_squared_error_theta = compute_squared_error(q_o, avg_particle)

                return max_particle_squared_error_xy, max_particle_squared_error_theta, avg_particle_squared_error_xy, avg_particle_squared_error_theta
    
    def compute_metrics(self, data, state_mins, state_maxs, means, stds, xy_only, compute_metrics, num_test_samples, threshold_np, margin_np, activate_np):
        # initialise vectors to store squared errors
        MSE_max_particle_xy = torch.zeros(num_test_samples)
        MSE_max_particle_theta = torch.zeros(num_test_samples)
        MSE_avg_particle_xy = torch.zeros(num_test_samples)
        MSE_avg_particle_theta = torch.zeros(num_test_samples)

        # generate test samples and record squared errors
        i = 0
        for i in tqdm.tqdm(range(num_test_samples)):
            qo_gt = data['q_o'][i, 0, :]
            q_r_hist = data['q_r'][i, :, :]
            o_hist = data['o'][i, :, :]
            phi_hist = data['phi'][i, :, :]
            if activate_np == False:
                max_particle_squared_error_xy, max_particle_squared_error_theta, avg_particle_squared_error_xy, avg_particle_squared_error_theta = self.test_resampling_non_contact_only(qo_gt, q_r_hist, o_hist, phi_hist, state_mins, state_maxs, means, stds, xy_only, compute_metrics)
            else:
                max_particle_squared_error_xy, max_particle_squared_error_theta, avg_particle_squared_error_xy, avg_particle_squared_error_theta = self.test_with_NP(qo_gt, q_r_hist, o_hist, phi_hist, state_mins, state_maxs, means, stds, xy_only, compute_metrics, threshold_np, margin_np)
            MSE_max_particle_xy[i] = max_particle_squared_error_xy
            MSE_max_particle_theta[i] = max_particle_squared_error_theta
            MSE_avg_particle_xy[i] = avg_particle_squared_error_xy
            MSE_avg_particle_theta[i] = avg_particle_squared_error_theta

        MSE_max_particle_xy = torch.mean(MSE_max_particle_xy)
        MSE_max_particle_theta = torch.mean(MSE_max_particle_theta)
        MSE_avg_particle_xy = torch.mean(MSE_avg_particle_xy)
        MSE_avg_particle_theta = torch.mean(MSE_avg_particle_theta)

        print('xy MSE for particle with max probability: ', MSE_max_particle_xy.item())
        print('theta MSE for particle with max probability: ', MSE_max_particle_theta.item())
        print('xy MSE for average particle from the final belief: ', MSE_avg_particle_xy.item())
        print('theta MSE for average particle from the final belief: ', MSE_avg_particle_theta.item())

        return MSE_max_particle_xy, MSE_max_particle_theta, MSE_avg_particle_xy, MSE_avg_particle_theta

    # visualisation of particles and their weights
    def plot_particle_weights(self, particles, weights, q_r_achieved, s):
        plt.figure(figsize=(7.5, 7.5), dpi=100)
        scatter = plt.scatter(particles[:, 0], particles[:, 1], c=weights, cmap='viridis', s=s)
        plt.plot(q_r_achieved[0], q_r_achieved[1], 'x', c='red', label='Robot position')
        plt.xlim([-0.5, 0.5])
        plt.ylim([0.5, 1.5])
        # Ensure the aspect ratio is equal, making x and y scales the same
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(scatter, label='Weight intensity')
        plt.legend()

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
        seq_len = particle_list.shape[1]
        t = np.random.randint(0, seq_len)

        # randomly select particles to plot
        num_particles = particle_list.shape[2] # number of particles we wish to plot
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
        fig, axs = plt.subplots(len(keys)*2, figsize=(18, 36), dpi=100)
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

                # label the ground truth point with its value
                ax.text(closest_x[keys[i]], gt_y_sum, f'{gt_y_sum:.2f}', fontsize=10, color='tab:red', verticalalignment='bottom', horizontalalignment='right')
                
                # plot limits
                ax.set_xlim([ranges[keys[i]][0], ranges[keys[i]][1]])  # set x-axis limits based on ranges
                ax.set_ylim([0, np.max(y_sum[keys[i]]) * 1.1])
                ax.set_xlabel('state', fontsize=16)
                ax.set_yticks(np.linspace(0, np.max(y_sum[keys[i]]) * 1.1, 5))  # create 5 evenly spaced ticks
                ax.grid(True)
                ax.legend(fontsize=16)
            
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
                ax.legend(fontsize=16)
            
        plt.tight_layout()

# LOSS FUNCTIONS
# class that defines end to end training loss function
class loss_fn_e2e(nn.Module):
    def __init__(self):
        super(loss_fn_e2e, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes, world, xy_only):
        # parameters and inputs
        std = 0.01
        particle_std = 0.01
        seq_len = particle_list.shape[1]
        # print('particle_list', particle_list[0, 0, :10, :])
        # print('batch_q_o_r:', batch['q_o_r'][0, 0, :])
        
        # squared distance computation
        sq_distance = compute_sq_distance(particle_list, torch.cat((batch['q_o_r'], torch.tile(batch['q_o'][:, :, 2:3], (1, seq_len, 1))), dim=-1), state_step_sizes, xy_only) # shape: [batch_size, seq_len, num_particles]
        # print('sq_distance:', sq_distance[0, 0, :10])
        
        # gaussian computation
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        # print('particle_prob_list', particle_prob_list[0, 0, :10])
        activations = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=-1)))
        # print('activations:', activations[0, 0, :10])
        # print('sum of activations:', torch.sum(activations, dim=-1))
        # print('sum of activations plus 1e-16:', 1e-16 + torch.sum(activations, dim=-1))
        # print('negative log:', -torch.log(1e-16 + torch.sum(activations, dim=-1)))
        # print('loss:', loss)

        return loss
    
# individual loss for the observation likelihood estimator (OLE)
class loss_OLE(nn.Module):
    def __init__(self, dpf_instance):
        super(loss_OLE, self).__init__()
        self.dpf_instance = dpf_instance

    def forward(self, batch, means, stds, obj_dims, r_robot):
        # note: should only one time step be taken here??
        seq_len =  batch['phi'].shape[1]
        batch_size = batch['phi'].shape[0]
        loss = 0.0

        for t in range(seq_len):
            phi = batch['phi'][:, t, :] # [batch_size, 1]
            q_o_r = torch.cat((batch['q_o_r'][:, t, :], batch['q_o'][:, 0, 2:3]), dim=-1) # [batch_size, 3]

            # take the true object states and treat them as particles --> num_particles = batch_size
            test_particles = torch.tile(q_o_r[None , :, :], (batch_size, 1, 1))

            # apply observation likelihood estimator (OLE) for all pairs of observations and states in that batch
            OLE_out = self.dpf_instance.measurement_update(phi, test_particles, means, stds) # [batch_size, num_particles] = [batch_size, batch_size]

            # maximise probability at true states, minimise probability at all other states
            true_state_probs = torch.diag(OLE_out) # [batch_size]
            other_state_probs = OLE_out - torch.diag(true_state_probs) # [batch_size, batch_size]
            loss_at_t = torch.sum(-torch.log(true_state_probs)) / (batch_size) + torch.sum(-torch.log(1.0 - other_state_probs)) / (batch_size * (batch_size - 1)) # normalise the loss such that (correct -> 1, incorrect -> 0)
            loss += loss_at_t
        
        return loss
    
class loss_fn_NP(nn.Module):
    def __init__(self):
        super(loss_fn_NP, self).__init__()

    def forward(self, batch, particle_list, particle_prob_list, state_step_sizes, world, xy_only):
        # parameters and inputs
        std = 0.02
        particle_std = 0.02
        seq_len = particle_list.shape[1]
        
        # squared distance computation
        sq_distance = compute_sq_distance_angle(particle_list, torch.cat((batch['q_o_r'], torch.tile(batch['q_o'][:, :, 2:3], (1, seq_len, 1))), dim=-1), state_step_sizes, xy_only) # shape: [batch_size, seq_len, num_particles]
        
        # gaussian computation
        std_tensor = torch.tensor(2.0 * np.pi * (std ** 2.0))
        activations = (particle_prob_list / torch.sqrt(std_tensor)) * torch.exp(-sq_distance / (2.0 * (particle_std ** 2.0))) # shape: [batch_size, seq_len, num_particles]
        loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=-1)))

        return loss