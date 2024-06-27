import os
import numpy as np 
import pickle
import tqdm
from belief_dynamics_learning.world2d import World2D, dist_to_object

root = os.path.dirname(os.path.abspath(__file__))

def generate_dataset(world, num_sequences, num_steps_per_sequence):
    """
    Given an instance of the World2D class, generate a dataset of ``num_sequences``
    sequences of (observation, action, state) tuples. 
    An observation is a binary contact measurement. 
    An action is the desired robot position. 
    A state is the true state of the world, including the robot and object positions. 
    """
    data = []
    for i in tqdm.tqdm(range(num_sequences)):
        # sample a new ground truth pose 
        world.sample_gt_object_pose()
        # sample a new robot pose
        world.sample_robot_pose()
        # check for initial contact, resample robot pose if in contact
        d, _ = dist_to_object(world.q_r_0, world.qo_gt, world.obj_dims, world.r_robot)
        while d < 0: 
            world.sample_robot_pose()
            d, _ = dist_to_object(world.q_r_0, world.qo_gt, world.obj_dims, world.r_robot)
        # sample a sequence of (s,a,o) tuples
        q_r_hist, o_hist, a_hist = world.rollout(num_steps_per_sequence, world.qo_gt.copy())
        # only keep bits of sequence around positive contact measurements 
        contact_idx = np.where(o_hist==1)[0]
        if len(contact_idx) == 0: 
            continue
        else: 
            min_idx = np.min(contact_idx)
            max_idx = np.max(contact_idx)
            buffer_lb = np.min([min_idx, 10])
            buffer_ub = np.min([len(o_hist)-max_idx, 10])
            keep_idx = [min_idx-buffer_lb, max_idx+buffer_ub]
            q_r_hist = q_r_hist[keep_idx[0]:keep_idx[1]]
            o_hist = o_hist[keep_idx[0]:keep_idx[1]]
            a_hist = a_hist[keep_idx[0]:keep_idx[1]]
        # save the sequence to a dataset
        data.append((q_r_hist, o_hist, a_hist))
    

if __name__ == "__main__":
    num_particles = 100
    dt=0.1
    # object properties 
    dim_object = np.array([0.067, 0.14])
    qo_gt = np.array([0, 1, 0]) # ground truth object pose
    sigma_pos = 0.06 # standard deviation of position noise
    # robot properties
    qr_0 = np.array([0, 0.7])
    r_robot = 0.05
    # create a world instance
    world = World2D(num_particles, qo_gt, dim_object, sigma_pos, 
                    qr_0, r_robot, dt)
    # generate a dataset
    data = generate_dataset(world, 100, 100)
    # save the dataset to a file
    save_path = root + '../data/data.pkl'
    with open('data.pkl', 'wb') as f: 
        pickle.dump(save_path, f)