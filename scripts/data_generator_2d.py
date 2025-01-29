import os
import numpy as np 
import pickle
import tqdm
from belief_dynamics_learning.world2d import World2D, dist_to_object
from belief_dynamics_learning.dpf_utils import angle_between_vectors
from scipy.spatial import ConvexHull, Delaunay, qhull

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

def generate_dataset_contact_only(world, num_sequences, num_steps_per_sequence=1):
    """
    Modification of generate_dataset() method such that generated trajectories
    are of length 2, and have a contact observation at the second timestep
    """
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
def generate_dataset_phi(world, num_sequences, num_steps_per_sequence, num_contacts=10):
    """
    Modification of generate_dataset() method such that generated trajectories
    have contact observation at every timestep (number of contacts given by 'num_contacts').
    Contact observation is angle 'phi', which is the angle between the fixed_axis (vertical axis)
    and observation_axis (axis connecting closest point on object and robot center)
    """
    data = []
    for i in tqdm.tqdm(range(num_sequences)):
        
        observation_count = 0
        # perform rollouts until a trajectory with 10 or more observations is found
        while observation_count < num_contacts:
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
        data.append((world.qo_gt, q_r_hist, o_hist, phi_hist, closest_point_hist))

    return data

# generate datasets where the robot trajectories are optimal in the sense that convex hull of robot positions during contact encompasses ground truth object
def generate_dataset_phi_opt_traj(world, num_sequences, num_steps_per_sequence, num_contacts=10):
    """
    Modification of generate_dataset() method such that generated trajectories have
    contact observation at every timestep (number of contacts given by 'num_contacts').
    Other generate_dataset() methods are based on a random walk. Here, "optimal"
    trajectories are hard-coded / generated. Given the generated trajectory, if the
    convex hull of robot positions does not fully encompass the object, then the trajectory
    is discarded. This way, all trajectories contain contact observations around the object.
    """
    # given q_o, compute coordinates of the corners of the object
    def get_corners(q_o, dim_object):
        x, y, theta = q_o
        width, height = dim_object

        # Calculate half dimensions
        half_width = width / 2
        half_height = height / 2

        # Define the four corners relative to the center (before rotation)
        corners = np.array([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])

        # Rotation matrix for angle theta
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Rotate each corner and translate to (x, y)
        rotated_corners = (rotation_matrix @ corners.T).T
        translated_corners = rotated_corners + np.array([x, y])

        return translated_corners

    def is_within_convex_hull(corners, robot_contact_pos):
        # Create a convex hull for the points in robot_contact_pos
        # Add a small jitter to points to prevent precision errors
        jitter = 1e-10 * np.random.randn(*robot_contact_pos.shape)
        robot_contact_pos_jittered = robot_contact_pos + jitter

        try:
            # Create a convex hull with the 'QJ' option to joggle input
            hull = ConvexHull(robot_contact_pos_jittered, qhull_options='QJ')
            delaunay = Delaunay(robot_contact_pos_jittered[hull.vertices], qhull_options='QJ')
            
            # Check if all points in corners are inside the convex hull
            is_inside = delaunay.find_simplex(corners) >= 0
            return np.all(is_inside)
        
        except qhull.QhullError as e:
            print("Convex hull error:", e)
            return False

    data = []
    dim_object = np.array([0.067, 0.14])
    for i in tqdm.tqdm(range(num_sequences)):
        
        observation_count = 0
        opt_traj_check = False
        # perform rollouts until a trajectory with 10 or more observations is found
        while observation_count < num_contacts or opt_traj_check == False:
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

            # at least three points needed in robot_contact_pos to make a simplex for convex hull code to work
            if observation_count > num_contacts:
                # check that trajectory is "optimal"
                corners = get_corners(world.qo_gt.copy(), dim_object)
                contact_idx = np.where(o_hist==1)[0]
                robot_contact_pos = q_r_hist[contact_idx, :]
                opt_traj_check = is_within_convex_hull(corners, robot_contact_pos)
                print(opt_traj_check)

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
        data.append((world.qo_gt, q_r_hist, o_hist, phi_hist, closest_point_hist))

    return data

# generate dataset
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
    data = generate_dataset_phi_opt_traj(world, 10, 1000)
    # save the dataset to a file
    save_path = os.path.join(root, '../data/name_of_dataset.pkl')
    with open(save_path, 'wb') as f: 
        pickle.dump(data, f)