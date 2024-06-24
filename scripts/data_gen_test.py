import sys
sys.path.append('../distance-py/build')
import distancepy

import matplotlib.pyplot as plt 
import mujoco
import numpy as np
import quaternion
from tqdm import tqdm

from belief_dynamics_learning.robot import Robot
from belief_dynamics_learning.trajectory_sampling import TrajectorySampler
from belief_dynamics_learning.mujoco_utils import sim_and_show_candidate_traj
from belief_dynamics_learning.mujoco_utils import compute_gt_rollout


def compute_dist_and_contact_point(gt_rollout, q_traj, system, box_dims): 
    q_robot_poses = np.concatenate([q_traj, np.zeros((q_traj.shape[0], 2))], axis=1)
    object_poses = gt_rollout[:, :, system.model.nq-7:system.model.nq] # (1, num_time_steps, 7) 

    object_poses_list = [object_poses[:, i, :] for i in range(object_poses.shape[1])] # list of length num_time_steps with shape (num_particles, 7)
    q_robot_poses_list = [q_robot_poses[i, :] for i in range(q_robot_poses.shape[0])] # list of length num_time_steps with shape (num_particles, 9)
    print(len(q_robot_poses_list), len(object_poses_list))
    print(q_robot_poses_list[0].shape, object_poses_list[0].shape)

    dists, closest_points = distancepy.computeRobotObjectDistancesAndPoints(
                q_robot_poses_list, object_poses_list, box_dims) 
    print(np.array(dists).shape)
    print(np.array(closest_points).shape)
    return dists, closest_points 

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("../mujoco_franka_emika_panda/scene_box.xml")
    box_dims = np.array([0.067, 0.14, 0.095])
    pos_obj_init = np.array([0.5, 0.0, 0.0475]) 
    orient_obj_init = quaternion.as_float_array(quaternion.from_rotation_vector(np.array([0, 0, 0])))
    pose_obj_init = np.concatenate((pos_obj_init, orient_obj_init))
    q_home = np.array([0, 0.7, 0, -1.57079, 0, 1.57079+0.7, 0.7853, 0.04])
    q_start = np.array([-0.0113, -0.2576, -0.0366, -2.6312, -0.0391, 2.3982, 0.7856])
    q_start_9 = np.concatenate([q_start, [0, 0]])
    dt_control = 0.01 
    dt_mujoco = 0.01 
    model.opt.timestep = dt_mujoco
    num_particles = 10
    system = Robot(model, q_start_9[:8], pose_obj_init, num_particles=num_particles)
    system.EE_name = 'grasp_frame'
    system.launch_viewer(right_ui=False)

    xd = pos_obj_init.copy()
    TrajSampler = TrajectorySampler()
    q_traj_list = TrajSampler.generate_trajs(q_start, system, xd.reshape(1, -1))
    q_traj = q_traj_list[0]
    print(q_traj.shape)
    # sim_and_show_candidate_traj(system, q_traj)
    gt_rollout = compute_gt_rollout(system, q_traj, pose_obj_init) 
    print(gt_rollout.shape)

    dists, closest_pointss = compute_dist_and_contact_point(gt_rollout, q_traj, system, box_dims)