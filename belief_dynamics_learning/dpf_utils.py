import numpy as np
import math

# take raw data and make a dictionary which holds all the data
def organise_data(raw_data, num_sequences, num_steps_per_sequence):
    
    # initialise dictionary values
    q_o = np.zeros((num_sequences, 1, 3), dtype=float) # object pose
    q_r = np.zeros((num_sequences, num_steps_per_sequence, 2), dtype=float) # robot pose
    o = np.zeros((num_sequences, num_steps_per_sequence, 1), dtype=float) # observation
    a = np.zeros((num_sequences, num_steps_per_sequence, 2), dtype=float) # action

    data = {'q_o': q_o,
            'q_r': q_r,
            'o': o,
            'a': a}

    # store contents of raw data file into q_r, o, a
    for i, trajectory in enumerate(raw_data):
        q_o_traj, q_r_traj, o_traj, a_traj = trajectory
        
        # store object pose for all trajectories
        q_o_traj = np.expand_dims(q_o_traj, axis=(0, 1))
        data['q_o'][i, :, :] = q_o_traj

        # store robot pose histories for all trajectories
        q_r_traj = np.expand_dims(q_r_traj[:num_steps_per_sequence, :], axis=0)
        data['q_r'][i, :, :] = q_r_traj

        # store observation histories for all trajectories
        o_traj = np.expand_dims(o_traj[:num_steps_per_sequence], axis=(0,2))
        data['o'][i, :, :] = o_traj

        # store action histories for all trajectories
        a_traj = np.expand_dims(a_traj, axis=0)
        data['a'][i, :, :] = a_traj

    # modify actions such that 'a' represents the current desired robot position --> shift every element in 'a' forward by one timestep
    num_timesteps = data['a'].shape[1]
    data['a'][:, 1:, :] = data['a'][:, :num_timesteps-1, :]

    return data

# split data into training / validation set according to split_ratio
def split_data(data, split_ratio):
    
    # keys for dictionary which contains data
    keys = ['q_o', 'q_r', 'o', 'a']

    # number of trajectories in the data
    num_trajectories = data['q_o'].shape[0]

    train_data = {key: data[key][:math.floor(num_trajectories*split_ratio), :, :] for key in keys}
    val_data = {key: data[key][math.floor(num_trajectories*split_ratio):, :, :] for key in keys}

    return train_data, val_data

# compute some statistics of the training data
def compute_statistics(data):
    
    means = dict()
    stds = dict()
    q_o_maxs = []
    q_o_mins = []
    q_r_step_sizes = []
    q_r_maxs = []
    q_r_mins = []

    keys = ['q_o', 'q_r', 'o', 'a']

    for key in keys:
        # compute means
        means[key] = np.mean(data[key], axis=(0, 1))

        # compute stds
        stds[key] = np.std(data[key], axis=(0, 1))

    # compute object pose maximums
    q_o_maxs = np.max(data['q_o'], axis=(0, 1))
    q_o_mins = np.min(data['q_o'], axis=(0, 1))

    # compute robot pose maximums
    q_r_maxs = np.max(data['q_r'], axis=(0, 1))
    q_r_mins = np.min(data['q_r'], axis=(0, 1))

    # compute average robot pose step sizes (q_r_step_sizes)
    for i in range(2):
        steps = np.reshape(data['q_r'][:, 1:, i] - data['q_r'][:, :-1, i], [-1])
        q_r_step_sizes.append(np.mean(abs(steps)))

    return means, stds, q_o_maxs, q_o_mins, q_r_step_sizes, q_r_maxs, q_r_mins

# compute squared distance between particle list and the object poses in the batch --> note: scale each dimension by dividing by the step sizes (for a sensible metric across state dimensions)
def compute_sq_distance(particle_list, batch_q_o, q_r_step_sizes):
    
    # compute some parameters
    batch_size = batch_q_o.shape[0]
    seq_len = batch_q_o.shape[1]
    num_particles = particle_list.shape[2]
    state_dim = particle_list.shape[-1]
    
    # add dimension to tensor containing batch of object poses
    batch_q_o = batch_q_o[:, :, None, :]
    assert batch_q_o.shape == (batch_size, seq_len, 1, state_dim)

    # compute squared distance
    result = 0.0
    for i in range(state_dim-1):
        # compute difference
        diff = particle_list[..., i] - batch_q_o[..., i]
        # wrap angle for theta
        if i == 2:
            diff = wrap_angle(diff)
        # add up scaled squared distance
        result += (diff / q_r_step_sizes[i]) ** 2
    return result

# method for keeping angles between 0 and 2*pi
def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi))