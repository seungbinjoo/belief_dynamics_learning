import os
import numpy as np 
import pickle
import tqdm
from belief_dynamics_learning.world2d import World2D, dist_to_object
from belief_dynamics_learning.dpf_utils import angle_between_vectors

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
        data.append((world.qo_gt, q_r_hist, o_hist, a_hist))
    return data

# data generator that creates trajectories of length 2, with 1 observation --> we ensure that the observation is not at the first time step
def generate_dataset_contact_only(world, num_sequences, num_steps_per_sequence=1):
    data = []
    for i in tqdm.tqdm(range(num_sequences)):
        observation_count = 0
        # perform rollouts of length 2, until a trajectory with 1 or more observations is found
        while observation_count < 1:
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
            observation_count = np.sum(o_hist)
            # if first time step has observation, skip
            if o_hist[0] == 0:
                continue
        data.append((world.qo_gt, q_r_hist, o_hist, a_hist))
    return data

# data generation for contact-only trajectories, where the observation is phi
def generate_dataset_phi(world, num_sequences, num_steps_per_sequence=3):
    data = []
    for i in tqdm.tqdm(range(num_sequences)):
        
        observation_count = 0
        # perform rollouts until a trajectory with 1 or more observations is found
        while observation_count < 3:
            # sample a new ground truth pose 
            world.sample_gt_object_pose()
            # sample a new robot pose
            world.sample_robot_pose()
            # check for initial contact, resample robot pose if in contact
            d, closest_point = dist_to_object(world.q_r_0, world.qo_gt, world.obj_dims, world.r_robot)
            closest_point = closest_point - world.q_r_0
            while d < 0: 
                world.sample_robot_pose()
                d, closest_point = dist_to_object(world.q_r_0, world.qo_gt, world.obj_dims, world.r_robot)
                closest_point = closest_point - world.q_r_0
            
            # sample a sequence of (s,a,o) tuples
            q_r_hist, o_hist, a_hist = world.rollout(num_steps_per_sequence, world.qo_gt.copy())
            observation_count = np.sum(o_hist)

            # if first time step has observation, skip
            # if o_hist[0] == 0:
            #     continue

            # compute observation phi
            fixed_axis = [0, 1] # vertical axis in robot frame
            phi_hist = []
            closest_point_hist = []
            for t in range(num_steps_per_sequence):
                _, closest_point = dist_to_object(q_r_hist[t], world.qo_gt, world.obj_dims, world.r_robot)
                closest_point = closest_point - q_r_hist[t]
                observation_axis = closest_point
                phi, phi_deg = angle_between_vectors(observation_axis, fixed_axis)
                phi_hist.append(phi)
                closest_point_hist.append(closest_point)

        # dataset: q_o, q_r, phi
        data.append((world.qo_gt, q_r_hist, phi_hist, closest_point_hist))

    return data

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
    data = generate_dataset_phi(world, 100, 3)
    # save the dataset to a file
    save_path = os.path.join(root, '../data/phi_data_min_3_contacts_seq_len_3_test_100_trajectories.pkl')
    with open(save_path, 'wb') as f: 
        pickle.dump(data, f)