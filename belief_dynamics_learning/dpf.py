import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from belief_dynamics_learning.belief_plotting_2d import *
from belief_dynamics_learning.world2d import *

class DPF():

    def __init__(self, propose_ratio, proposer_keep_ratio, min_obs_likelihood, world: World2D):
        """
        Apply differentiable particle filter (DPF) to current belief of the object pose (represented by particle set)
        to predict the belief at the next time step.
        Args:
            propose_ratio: 
            proposer_keep_ratio: dropout ratio for proposer network
            min_obs_likelihood:
            world: object of class World2D
        """
        # store hyperparameters
        self.propose_ratio = propose_ratio
        self.proposer_keep_ratio = proposer_keep_ratio
        self.min_obs_likelihood = min_obs_likelihood
        self.world = world
        
        # other useful parameters
        self.state_dim = 3

        # build learnable networks: observation likelihood estimator, particle proposer
        self.build_networks()

    def build_networks(self):

        # PARTICLE PROPOSER --> maps observations and robot poses to particles
        self.particle_proposer = ParticleProposer(self.proposer_keep_ratio)

        # OBSERVATION LIKELIHOOD ESTIMATOR --> maps observations and robot poses to probabilities
        self.obs_like_estimator = ObsLikelihoodEstimator(self.min_obs_likelihood)

    def measurement_update(self, observation, q_r, particles):
        """
        For each particle in the particle-based belief, compute likelihood of the observation.
        Args:
            observation:
            q_r:
            particles: 
        """

        # prepare input to the observation likelihood estimator network
        q_r_input = torch.tile(torch.tensor(q_r), (self.world.num_particles,))
        observation_input = torch.tile(torch.tensor(observation), (self.world.num_particles,))
        particles_input = torch.tensor(self.transform_particles_as_input(particles))
        input = torch.cat((q_r_input, observation_input, particles_input), dim=1)

        # for each particle, estimate the likelihood based on the observation
        with torch.no_grad(): # turn of gradient computation for model evaluation
            self.obs_like_estimator.eval() # set model to evaluation mode (e.g. deactivates any dropout layers)
            obs_likelihood = self.obs_like_estimator(input) # pass input particle set through the observaton likelihood estimator network

        return obs_likelihood
    
    def transform_particles_as_input(particles): # 'particles' has dimensions [num_particles, 3] --> normalise particles matrix (along first dimension)
        mean = np.mean(particles, axis=0)
        std = np.std(particles, axis=0)
        normalised_particles = (particles - mean) / std
        return normalised_particles
    
    # def train_models(self, learning_rate, batch_size, num_epochs):
        
    #     # define optimiser for observation likelihood estimator
    #     optimiser_ole = torch.optim.Adam(self.obs_like_estimator.parameters(), lr=learning_rate)

    #     # define optimiser for particle proposer network
    #     optimiser_pp = torch.optim.Adam(self.particle_proposer.parameters(), lr=learning_rate)

    #     # define loss --> for end-to-end training
    #     sq_distance = compute_sq_distance()
    #     activations = (particle_weights / torch.sqrt(2 * np.pi * std **2)) * torch.exp(-sq_distance / (2.0 * particle_std **2 ))
    #     loss = torch.mean(-torch.log(1e-16 + activations))

class ParticleProposer(nn.Module):

    # initialise neural network architecture
    def __init__(self, proposer_keep_ratio):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(p=proposer_keep_ratio),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Tanh(),
        )

    # define the forward pass of the network (how input data x moves through network layers)
    def forward(self, x):
        x = self.linear_stack(x)
        return x
    
class ObsLikelihoodEstimator(nn.Module):

    # initialise neural network architecture
    def __init__(self, min_obs_likelihood):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(6, 32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.min_obs_likelihood = min_obs_likelihood
    
    # define the forward pass of the network (how input data x moves through network layers)
    def forward(self, x):
        x = self.linear_stack(x)
        x = x * (1 - self.min_obs_likelihood) + self.min_obs_likelihood # ensures that all probabilities outputted are higher than minimum observation likelihood (a design parameter)
        return x