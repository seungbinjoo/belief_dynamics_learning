import numpy as np 
from belief_dynamics_learning.world2d import World2D, dist_to_object


def generate_dataset(world, num_sequences, num_steps_per_sequence):
    """
    Given an instance of the World2D class, generate a dataset of ``num_sequences``
    sequences of (observation, action, state) tuples. 
    An observation is a binary contact measurement. 
    An action is the desired robot position. 
    A state is the true state of the world, including the robot and object positions. 
    """
    for i in range(num_sequences):
        # sample a new ground truth pose 
        world.sample_gt_object_pose()
        # sample a new robot pose
        world.sample_robot_pose()
        # check for initial contact, resample robot pose if in contact
        d, _ = dist_to_object(world.q_r_0, world.q_o_gt, world.obj_dims, world.r_robot)
        while d < 0: 
            world.sample_robot_pose()
            d, _ = dist_to_object(world.q_r_0, world.q_o_gt, world.obj_dims, world.r_robot)
        # sample a sequence of (s,a,o) tuples
        q_r_hist, o_hist, a_hist = world.rollout(num_steps_per_sequence, world.qo_gt.copy())
        #TODO: only keep bits of sequence around positive contact measurements 
        contact_idx = np.where(o_hist==1)[0]
        if len(contact_idx) == 0: 
            continue
        
