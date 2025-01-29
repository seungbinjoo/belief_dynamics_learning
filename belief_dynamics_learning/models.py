import torch
import math
import torch.nn as nn

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
            'q_o_r': self.data['q_o_r'][traj_idx, start_idx:end_idx, :],
            'o': self.data['o'][traj_idx, start_idx:end_idx, :],
            'phi': self.data['phi'][traj_idx, start_idx:end_idx, :],
            'cp': self.data['cp'][traj_idx, start_idx:end_idx, :]
        }
        return sample

# class that defines particle proposer network
class ParticleProposer(nn.Module):

    # initialise neural network architecture
    def __init__(self, proposer_keep_ratio, xy_only, phi_dataset):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(p=proposer_keep_ratio),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Tanh(), # tanh maps to between -1 and 1
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
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
        self.min_obs_likelihood = min_obs_likelihood
    
    # define the forward pass of the network (how input data x moves through network layers)
    def forward(self, x):
        x = self.linear_stack(x)
        x = x * (1 - self.min_obs_likelihood) + self.min_obs_likelihood # ensures that all probabilities outputted are higher than minimum observation likelihood (a design parameter)
        return x
    
# class that defines orientation predictor network
class NegativeProposer(nn.Module):

    # initialise neural network architecture
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(), # sigmoid maps to probability between 0 and 1
        )

    # define the forward pass of the network (how input data x moves through network layers)
    def forward(self, x):
        x = self.linear_stack(x)
        return x