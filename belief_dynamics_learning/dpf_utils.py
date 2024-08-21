import numpy as np
import math
import torch
import torch.nn as nn
from belief_dynamics_learning.world2d import *

# take raw data and make a dictionary which holds all the data
def organise_data(raw_data, num_sequences, num_steps_per_sequence, contact_only=True, xy_only=False):

    # initialise dictionary values
    q_o = np.zeros((num_sequences, 1, 3), dtype=float) # object pose
    q_r = np.zeros((num_sequences, num_steps_per_sequence, 2), dtype=float) # robot pose
    q_o_r = np.zeros((num_sequences, num_steps_per_sequence, 2), dtype=float) # object pose in robot frame
    phi = np.zeros((num_sequences, num_steps_per_sequence, 1), dtype=float) # phi observation
    cp = np.zeros((num_sequences, num_steps_per_sequence, 2), dtype=float) # closest point

    data = {'q_o': q_o,
            'q_r': q_r,
            'q_o_r': q_o_r,
            'phi': phi,
            'cp': cp}
    
    # store contents of raw data file into q_o, q_r, phi
    for i, trajectory in enumerate(raw_data):
        q_o_traj, q_r_traj, phi_traj, cp_traj = trajectory
        
        # store object pose for all trajectories
        data['q_o'][i, :, :] = q_o_traj[None, :]

        # store robot pose histories for all trajectories
        q_r_traj = np.expand_dims(q_r_traj[:num_steps_per_sequence, :], axis=0)
        data['q_r'][i, :, :] = q_r_traj

        # store object pose in robot frame for all trajectories
        data['q_o_r'] = data['q_o'][:, :, :2] - data['q_r']

        # store observation histories for all trajectories
        # note: initially, phi_traj has shape () --> scalar np array
        data['phi'][i, :, :] = np.array(phi_traj)[None, None]

        # store closest point data --> won't be used in training though
        data['cp'][i, :, :] = np.array(cp_traj)[None, :]

    # if we desire trajectories with only contacts, for q_r, o, a, only take data from the single index with contact
    if contact_only == True:
        data['q_r'] = np.expand_dims(data['q_r'][:, 1, :], axis=1)
        data['q_o_r'] = np.expand_dims(data['q_o_r'][:, 1, :], axis=1)
    
    # if we only want xy states, get rid of angles from the dataset
    if xy_only == True:
        data['q_o'] = data['q_o'][:, :, :2]

    return data

def noisify_data(data):
    data = data
    return data

def check_errors_data(data, obj_dims, r_robot, xy_only):
    """
    Args:
        'data' is a dictionary containing...
        Object pose data: (500, 1, 3) or (500, 1, 2), depending on 'xy_only'
        Robot pose data: (500, 1, 2)
        Observation data: (500, 1, 1)
        Action data: (500, 1, 2)
    Returns:
        errors_in_data: that is, the data but only the trajectories that have the error where the object intersects with the robot
    """
    q_r = torch.from_numpy(data['q_r']) # [num_sequences, 1, 2]
    q_o = torch.from_numpy(data['q_o'][:, :, None, :]) # [num_sequences, 1, 1, state_dim]

    # for all trajectories in data, calculate distance between gt object and robot
    d = dist_to_object_torch(q_r, q_o, obj_dims, r_robot, xy_only) # [num_sequences, seq_len, num_particles] = [500, 1, 1]
    d = torch.squeeze(d) # [500]

    # select indices where distance is less than zero --> hence there is intersection
    error_indices = torch.where(d < 0)[0]

    if error_indices.shape[0] < 1:
        print('No errors (intersections between robot and object) found in the dataset.')
        return
    else:
        data_errors_only = {'q_o': data['q_o'][error_indices, :, :],
                            'q_r': data['q_r'][error_indices, :, :],
                            'o': data['o'][error_indices, :, :],
                            'a': data['a'][error_indices, :, :]}

        return data_errors_only

# split data into training / validation set according to split_ratio
def split_data(data, split_ratio):
    
    # keys for dictionary which contains data
    keys = ['q_o', 'q_r', 'q_o_r', 'phi', 'cp']

    # number of trajectories in the data
    num_trajectories = data['q_o'].shape[0]

    train_data = {key: data[key][:math.floor(num_trajectories*split_ratio), :, :] for key in keys}
    val_data = {key: data[key][math.floor(num_trajectories*split_ratio):, :, :] for key in keys}

    return train_data, val_data

# compute some statistics of the training data
def compute_statistics(data, phi_dataset):
    
    means = dict()
    stds = dict()
    q_o_r_maxs = []
    q_o_r_mins = []
    q_r_step_sizes = []
    q_r_maxs = []
    q_r_mins = []

    if phi_dataset == False:
        keys = ['q_o', 'q_r', 'o', 'a']
    else:
        keys = ['q_o', 'q_r', 'phi', 'q_o_r', 'cp']

    for key in keys:
        # compute means
        means[key] = np.mean(data[key], axis=(0, 1))

        # compute stds
        stds[key] = np.std(data[key], axis=(0, 1))

    # compute object pose maximums
    q_o_r_maxs = np.max(data['q_o_r'], axis=(0, 1))
    q_o_r_mins = np.min(data['q_o_r'], axis=(0, 1))

    # compute robot pose maximums
    q_r_maxs = np.max(data['q_r'], axis=(0, 1))
    q_r_mins = np.min(data['q_r'], axis=(0, 1))

    # compute average robot pose step sizes (q_r_step_sizes)
    for i in range(2):
        steps = np.reshape(data['q_r'][:, 1:, i] - data['q_r'][:, :-1, i], [-1])
        q_r_step_sizes.append(np.mean(abs(steps)))

    return means, stds, q_o_r_maxs, q_o_r_mins, q_r_step_sizes, q_r_maxs, q_r_mins

# compute squared distance between particle list and the object poses in the batch --> note: scale each dimension by dividing by the step sizes (for a sensible metric across state dimensions)
def compute_sq_distance(particle_list, batch_q_o, q_r_step_sizes, xy_only):
    
    # compute some parameters
    batch_size = batch_q_o.shape[0]
    seq_len = batch_q_o.shape[1]
    num_particles = particle_list.shape[2]
    state_dim = batch_q_o.shape[-1]

    if xy_only == True:
        assert state_dim == 2
    
    # add dimension to tensor containing batch of object poses
    batch_q_o = batch_q_o[:, :, None, :]
    assert batch_q_o.shape == (batch_size, seq_len, 1, state_dim)

    result = 0.0
    scalings = [1, 1, 10]

    # compute squared distance
    for i in range(state_dim):
        # compute difference
        diff = particle_list[..., i] - batch_q_o[..., i] # 'diff' has shape [batch_size, seq_len, num_particles] --> note: '...' represents as many ':' as needed to cover all the dimensions

        # wrap angle for theta
        if i == 2:
            diff = wrap_angle(diff)
            # print('diff theta:', diff/scalings[i])
        else:
            # print('diff xy:', diff/scalings[i])
            pass
        
        # add up scaled squared distance
        result += (diff/scalings[i]) ** 2
    return result

def compute_sq_distance_other(batch_q_o, num_states_other, buffer):
    """
    Args:
        batch_q_o: [batch_size, 1, state_dim]
        num_states_other: []
        buffer: []
    Returns:
        sq_distance_other: [batch_size, seq_len, num_states_other, num_states_other, num_states_other]
    """
    # x, y, theta should have shapes [batch_size, 1, num_states_other]
    x_ratio = (batch_q_o[:, :, 0] - (-0.5)) / (0.5 - (-0.5))
    num_states_other_left = torch.floor(num_states_other * x_ratio) # [batch_size, 1]
    num_states_other_right = num_states_other - num_states_other_left # [batch_size, 1]

    x = torch.zeros(batch_q_o.shape[0], batch_q_o.shape[1], num_states_other)
    for i in range(batch_q_o.shape[0]):
        x_left = torch.linspace(-0.5, batch_q_o[i, 0, 0].item() - buffer, int(num_states_other_left[i, 0].item()))
        x_right = torch.linspace(batch_q_o[i, 0, 0].item() + buffer, 0.5, int(num_states_other_right[i, 0].item()))
        x[i, 0, :] = torch.cat((x_left, x_right))

    y_ratio = (batch_q_o[:, :, 1] - (0.5)) / (1.5 - (0.5))
    num_states_other_left = torch.floor(num_states_other * y_ratio) # [batch_size, 1]
    num_states_other_right = num_states_other - num_states_other_left # [batch_size, 1]

    y = torch.zeros(batch_q_o.shape[0], batch_q_o.shape[1], num_states_other)
    for i in range(batch_q_o.shape[0]):
        y_left = torch.linspace(0.5, batch_q_o[i, 0, 1].item() - buffer, int(num_states_other_left[i, 0].item()))
        y_right = torch.linspace(batch_q_o[i, 0, 1].item() + buffer, 1.5, int(num_states_other_right[i, 0].item()))
        y[i, 0, :] = torch.cat((y_left, y_right))

    scalings = [1, 1]
    diff_x = (x - batch_q_o[:, :, 0:1]) / scalings[0] # [batch_size, 1, num_states_other]
    diff_y = (y - batch_q_o[:, :, 1:2]) / scalings[1] # [batch_size, 1, num_states_other]
    # diff_theta = (theta - batch_q_o[:, :, 2:3]) / scalings[2] # [batch_size, 1, num_states_other]

    result_x = diff_x ** 2.0 # [batch_size, 1, num_states_other]
    result_y = diff_y ** 2.0 # [batch_size, 1, num_states_other]
    # result_theta = diff_theta ** 2.0 # [batch_size, 1, num_states_other]

    sq_distance_other = torch.zeros(batch_q_o.shape[0], batch_q_o.shape[1], num_states_other, num_states_other)
    # sq_distance_other = torch.zeros(batch_q_o.shape[0], batch_q_o.shape[1], num_states_other, num_states_other, num_states_other)

    # for i in range(num_states_other):
    #     for j in range(num_states_other):
    #         for k in range(num_states_other):
    #             sq_distance_other[:, :, i, j, k] = result_x[:, :, i] + result_y[:, :, j] + result_theta[:, :, k]

    for i in range(num_states_other):
        for j in range(num_states_other):
            sq_distance_other[:, :, i, j] = result_x[:, :, i] + result_y[:, :, j]

    return sq_distance_other

# method for keeping angles between -pi and pi
def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

# angle between two vectors --> between -pi and pi
def angle_between_vectors(u, v):
    # convert inputs to numpy arrays
    u = np.array(u)
    v = np.array(v)
    
    # calculate the dot product
    dot_product = np.dot(u, v)
    
    # calculate the magnitudes of the vectors
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # compute the angle in radians between 0 and pi
    cos_theta = dot_product / (norm_u * norm_v)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # calculate the 2D cross product (equivalent to the z-component of the 3D cross product)
    cross_product_z = u[0] * v[1] - u[1] * v[0]
    
    # determine the sign of the angle
    if cross_product_z < 0:
        angle_rad = -angle_rad
    
    # convert angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_rad, angle_deg