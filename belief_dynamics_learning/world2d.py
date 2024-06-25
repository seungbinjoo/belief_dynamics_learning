import numpy as np
import time 
from belief_dynamics_learning.belief_plotting_2d import *

# colors 
robot_color = 'darkslategrey'
gt_color = 'lightpink'

# binary measurement model
P_c = 0.78 # probability of contact (true positive)
P_n = 0.9 # probability of no contact (true negative)


class World2D:
    def __init__(self, num_particles, gt_obj_pose, 
                 obj_dims, sigma_pos, q_r_0, r_robot):
        self.num_particles = num_particles
        self.qo_gt = gt_obj_pose
        self.obj_dims = obj_dims
        self.q_r_0 = q_r_0
        self.r_robot = r_robot
        self.particles = self.generate_particles(sigma_pos)
        self.weights = np.ones(num_particles) / num_particles

    def generate_particles(self, sigma_pos):
        """
        Particles are represented as 3D vectors (x, y, theta), 
        where theta is the yaw angle.
        """
        particles = np.zeros((self.num_particles, 3))
        sigma_qo_pos = np.diag([sigma_pos, sigma_pos])**2
        # sample positions from a Gaussian distribution (for now)
        particles[:, :2] = np.random.multivariate_normal(self.qo_gt[:2], sigma_qo_pos, self.num_particles)
        # sample yaw angles uniformly 
        particles[:, 2] = np.random.uniform(0, np.pi, self.num_particles)
        return particles
    
    def plot_belief(self):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([0.6, 1.6])
        plot_robot(ax, self.q_r_0, self.r_robot, color=robot_color)
        plot_object(ax, self.qo_gt, self.obj_dims, color=gt_color)
        plot_object_belief(ax, self.particles, self.weights, self.obj_dims)
        plt.show()


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

