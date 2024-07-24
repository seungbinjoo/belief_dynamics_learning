import numpy as np
import time 
import torch
from belief_dynamics_learning.belief_plotting_2d import *

# colors 
robot_color = 'darkslategrey'
gt_color = 'lightpink'

# binary measurement model
P_c = 0.78 # probability of contact (true positive)
P_n = 0.9 # probability of no contact (true negative)


def dist_to_object(q_r, q_o, dim_object, r_robot):
    """
    Compute the shortest distance between the robot and the object.
    Args:
        q_r: robot pose [3]
        q_o: object pose [3]
        dim_object: object dimensions [2] - assuming rectangle
        r_robot: robot radius
    Returns:   
        d: shortest distance
        closest_point: closest point on the object to the robot 
                       (in the robot frame)
    """
    # Compute robot position in object frame
    sin_o = np.sin(q_o[2])
    cos_o = np.cos(q_o[2])
    R = np.array([[cos_o, sin_o], [-sin_o, cos_o]])
    q_r_o = R @ (q_r[:2] - q_o[:2])
    # Clip robot to rectangle
    q_r_o_clipped = np.clip(q_r_o, -dim_object/2, dim_object/2)
    # Compute distance
    d = np.linalg.norm(q_r_o - q_r_o_clipped) - r_robot
    # Convert the closest point back to the robot frame
    R_inv = np.linalg.inv(R)
    closest_point = R_inv @ q_r_o_clipped + q_o[:2]
    return d, closest_point

def dist_to_object_batch(q_r, q_o, dim_object, r_robot):
    """
    Compute the shortest distance between the robot and the object in batch. 
    Args: 
        q_r: robot pose [3]
        q_o: object poses [N, 3]
        dim_object: object dimensions [2]
        r_robot: robot radius
    Returns:
        d: shortest distances for each possible object pose to 
           given robot pose [N]
    """
    # Compute robot position in object frame
    sin_o = np.sin(q_o[:, 2])
    cos_o = np.cos(q_o[:, 2])
    R = np.array([[cos_o, sin_o], [-sin_o, cos_o]]).transpose(2, 0, 1) # [N, 2, 2]
    q_r_o = np.einsum('ijk,ik->ij', R, q_r[:2] - q_o[:, :2]) # [N, 2]
    # Clip robot to rectangle
    q_r_o_clipped = np.clip(q_r_o, -dim_object/2, dim_object/2) # [N, 2]
    # Compute distance
    d = np.linalg.norm(q_r_o - q_r_o_clipped, axis=1) - r_robot # [N]
    return d

def dist_to_object_torch(q_r, q_o, dim_object, r_robot):
    """
    Compute the shortest distance between the robot and the object.
    Args:
        q_r: robot pose [3] (tensor)
        q_o: object pose [3] (tensor)
        dim_object: object dimensions [2] (tensor) - assuming rectangle
        r_robot: robot radius (scalar)
    Returns:   
        d: shortest distance (tensor)
        closest_point: closest point on the object to the robot 
                       (in the robot frame) (tensor)
    """
    # print('robot pose:', q_r)
    # print('proposed particle = predicted object pose:', q_o)
    # Compute robot position in object frame
    sin_o = torch.sin(q_o[2])
    cos_o = torch.cos(q_o[2])
    R = torch.tensor([[cos_o, sin_o], [-sin_o, cos_o]])
    q_r_o = torch.matmul(R, (q_r[:2] - q_o[:2]))
    
    # Clip robot to rectangle
    q_r_o_clipped = torch.clamp(q_r_o, torch.from_numpy(-dim_object/2), torch.from_numpy(dim_object/2))
    
    # Compute distance
    d = torch.norm(q_r_o - q_r_o_clipped) - r_robot
    # print('distance:', d)
    
    # Convert the closest point back to the robot frame
    # R_inv = torch.linalg.inv(R)
    # closest_point = torch.matmul(R_inv, q_r_o_clipped) + q_o[:2]
    
    # return d, closest_point
    return d

def dist_to_object_batch_torch(q_r, q_o, dim_object, r_robot):
    """
    Compute the shortest distance between the robot and the object in batch. 
    Args: 
        q_r: robot pose [3] (tensor)
        q_o: object poses [N, 3] (tensor)
        dim_object: object dimensions [2] (tensor)
        r_robot: robot radius (scalar)
    Returns:
        d: shortest distances for each possible object pose to 
           given robot pose [N] (tensor)
    """
    # Compute robot position in object frame
    sin_o = torch.sin(q_o[:, 2])
    cos_o = torch.cos(q_o[:, 2])
    R = torch.stack([torch.stack([cos_o, sin_o], dim=1), torch.stack([-sin_o, cos_o], dim=1)], dim=1)  # [N, 2, 2]
    q_r_o = torch.einsum('nij,nj->ni', R, q_r[:2] - q_o[:, :2])  # [N, 2]
    
    # Clip robot to rectangle
    q_r_o_clipped = torch.clamp(q_r_o, -dim_object / 2, dim_object / 2)  # [N, 2]
    
    # Compute distance
    d = torch.norm(q_r_o - q_r_o_clipped, dim=1) - r_robot  # [N]
    
    return d

class World2D:
    def __init__(self, num_particles, gt_obj_pose, 
                 obj_dims, sigma_pos, q_r_0, r_robot, dt):
        self.num_particles = num_particles
        self.qo_gt = gt_obj_pose
        self.obj_dims = obj_dims
        self.q_r_0 = q_r_0
        self.r_robot = r_robot
        self.particles = self.generate_particles(sigma_pos)
        self.weights = np.ones(num_particles) / num_particles
        self.dt = dt
        self.bounds = np.array([[-0.5, 0.5], [0.5, 1.5]])
        # reduce bounds for sampling
        self.sample_bounds = self.bounds.copy()
        self.sample_bounds[:,0] += 0.1
        self.sample_bounds[:,1] -= 0.1
        # random walk motion model
        self.A = np.array([[1., 0., self.dt, 0.], 
                           [0., 1., 0., self.dt], 
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        self.B = np.array([[0.5 * self.dt**2, 0, self.dt, 0], 
                           [0, 0.5 * self.dt**2, 0, self.dt]]).T
        
    def sample_gt_object_pose(self):
        """
        Sample a new ground truth object pose. 
        """
        qo_gt_new = np.zeros(3)
        qo_gt_new[:2] = np.random.uniform(self.sample_bounds[:, 0], self.sample_bounds[:, 1])
        qo_gt_new[2] = np.random.uniform(-np.pi, np.pi) # used to be: (0, np.pi)
        self.qo_gt = qo_gt_new
        # print("New ground truth pose: ", self.qo_gt)

    def sample_robot_pose(self):
        """
        Sample a new robot pose. 
        """
        qr_new = np.zeros(2)
        qr_new = np.random.uniform(self.sample_bounds[:, 0], self.sample_bounds[:, 1])
        self.q_r_0 = qr_new
        # print("New robot pose: ", self.q_r_0)

    def generate_particles(self, sigma_pos):
        """
        Particles are represented as 3D vectors (x, y, theta), 
        where theta is the yaw angle. We sample the particles
        from a Gaussian distribution around the ground truth object pose.
        The orientation is sampled uniformly from [0, pi].
        """
        particles = np.zeros((self.num_particles, 3))
        sigma_qo_pos = np.diag([sigma_pos, sigma_pos])**2
        # sample positions from a Gaussian distribution (for now)
        particles[:, :2] = np.random.multivariate_normal(self.qo_gt[:2], sigma_qo_pos, self.num_particles)
        # sample yaw angles uniformly 
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        return particles
    
    def plot_belief(self):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        plot_robot(ax, self.q_r_0, self.r_robot, color=robot_color)
        plot_object(ax, self.qo_gt, self.obj_dims, color=gt_color)
        plot_object_belief(ax, self.particles, self.weights, self.obj_dims)
        plt.show()
    
    def plot_rollout(self, q_r_hist, o_hist):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.plot(q_r_hist[:, 0], q_r_hist[:, 1], 'o-', color=robot_color, alpha=0.5)
        contact_points = np.where(o_hist == 1)[0]
        for idx in contact_points:
            ax.add_patch(plt.Circle(q_r_hist[idx], self.r_robot, color='red', alpha=0.5))
        # ax.scatter(q_r_hist[contact_points, 0], q_r_hist[contact_points, 1], color='red', s=10)
        plot_object(ax, self.qo_gt, self.obj_dims, color=gt_color)
        plt.show()

    def plot_single_step(self, q_r_des, q_r_actual, q_o): 
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        plot_robot(ax, q_r_actual, self.r_robot, color=robot_color)
        plot_robot(ax, q_r_des, self.r_robot, color='r', alpha=0.5)
        plot_object(ax, q_o, self.obj_dims, color=gt_color, zorder=0)
        # build legend 
        rectangle_patch0 = patches.Rectangle((0, 0), self.obj_dims[0], self.obj_dims[1], 
                                            color=gt_color, label='Ground truth')
        # circle_patch = patches.Circle((0, 0), r_robot, color=robot_color, label='Robot')
        circle_patch0 = Line2D([0], [0], marker='o', color=robot_color, 
                            markerfacecolor=robot_color, markersize=10)
        circle_patch1 = Line2D([0], [0], marker='o', color='r', 
                            markerfacecolor='r', markersize=10, alpha=0.5)

        ax.legend(handles=[circle_patch0, circle_patch1, rectangle_patch0],
                    labels=[r'$q_{r,real}$', r'$q_{r,des}$', 'Ground truth (GT)'],
                    loc='upper left', fontsize=14)
        plt.show()

    def step(self, x, q_o, sigma_a=0.5):
        """
        Sample a random walk motion model for the robot.
        x_t = x_{t-1} + v_{t-1} * dt + 1/2 * a_t * dt^2 
        Given the one-step dynamics, we also check for contact with the object, either ground truth or from a particle.
        Args:
            x: robot state [4] (x, y, vx, vy)
            q_o: object pose [3]
            sigma_a: standard deviation of the acceleration noise
        Returns:    
            x_new: new robot state [4]
            o: contact observation (1 if in contact, 0 otherwise)
        """
        mean_a = np.zeros(2)
        a = np.random.normal(mean_a, sigma_a)
        x_new = np.dot(self.A, x) + np.dot(self.B, a)
        
        # check for contact
        d, _ = dist_to_object(x_new[:2], q_o, self.obj_dims, self.r_robot) 
        # check if within bounds
        within_bounds = (self.bounds[0, :] <= x_new[:2]) & (x_new[:2] <= self.bounds[1, :])

        # Enforce bounds if necessary
        if not within_bounds.all():
            # Reflect velocity if we hit the bounds
            x_new[2:] *= -1

        q_des = x_new[:2].copy()
        # if in contact, deflect the robot
        if d < 0:
            o = 1 # contact observation
            # deflect velocity
            x_new[2:] = -x_new[2:]
            # Clip robot to rectangle surface
            R_object = np.array([[np.cos(q_o[2]), -np.sin(q_o[2])], [np.sin(q_o[2]), np.cos(q_o[2])]])
            q_r_o = R_object.T @ (x_new[:2] - q_o[:2])
            # find closest point on object surface
            q_r_o_normalized = q_r_o / (self.obj_dims/2) # 1 if on the surface
            if np.abs(q_r_o_normalized[0]) > 1 and np.abs(q_r_o_normalized[1]) > 1:
                # Closest point is on the corner
                q_o_closest = np.sign(q_r_o_normalized)
            elif np.abs(q_r_o_normalized[0]) > np.abs(q_r_o_normalized[1]):
                # move to x border
                q_o_closest = np.array([np.sign(q_r_o_normalized[0]), q_r_o_normalized[1]])
            else:
                # move to y border
                q_o_closest = np.array([q_r_o_normalized[0], np.sign(q_r_o_normalized[1])])
            q_o_closest *= self.obj_dims/2
            dq_r_o_new = q_r_o - q_o_closest
            dq_r_o_new *= self.r_robot / np.linalg.norm(dq_r_o_new)
            q_r_o_new = q_o_closest + dq_r_o_new
            x_new[:2] = np.dot(R_object, q_r_o_new) + q_o[:2]
            # self.plot_single_step(q_des, x_new[:2], q_o)
        else: 
            o = 0
        # clip velocity
        x_new[2:] = np.clip(x_new[2:], -0.5, 0.5)
        return x_new, o, q_des 

    def rollout(self, num_steps, q_o):
        """
        Simulate the robot and generate observations for a 
        given number of time steps. The object is assumed to be static. 
        Args: 
            num_steps: number of time steps to simulate
            q_o: object pose [3] - this can be either the ground truth pose
                 or the pose from a particle 
        """
        q_r = self.q_r_0
        q_r_hist = [q_r]
        o_hist = [0]
        action_hist = []
        v0_mean = np.zeros(2)
        # sample initial vel 
        v0 = np.random.normal(v0_mean, 0.1)
        x = np.concatenate([q_r, v0])
        for t in range(num_steps):
            x,o, q_des = self.step(x, q_o)
            action_hist.append(q_des)
            q_r_hist.append(x[:2])
            o_hist.append(o)
        return np.array(q_r_hist), np.array(o_hist), np.array(action_hist)
    

if __name__ == '__main__': 
    num_particles = 1000

    # object properties 
    dim_object = np.array([0.067, 0.14])
    qo_gt = np.array([0, 1, np.pi/4]) # ground truth object pose
    sigma_pos = 0.06 # standard deviation of position noise

    # robot properties
    qr_0 = np.array([0, 0.7])
    r_robot = 0.05

    world = World2D(num_particles, qo_gt, dim_object, sigma_pos, qr_0, r_robot)
    world.plot_belief()

