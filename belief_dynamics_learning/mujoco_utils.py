import time 
import mujoco
import numpy as np
import quaternion
import mediapy as media
from mujoco import rollout


def reset_mujoco(system, obj_pose, robot_pos=None, bot=None):
    mujoco.mj_resetData(system.model, system.data)
    system.data.ctrl[7:-1] = np.zeros(7)
    if robot_pos is not None:
      if robot_pos.shape[0] == 7:
        robot_pos = np.concatenate([robot_pos, np.array([0, 0])])
      q_pos = np.concatenate([robot_pos, obj_pose])
      mujoco.mju_copy(system.data.qpos, q_pos) 
    elif bot is not None:
      system.data.qpos[:7] = bot.get_q()
      system.data.ctrl[:7] = bot.get_q()
    else: 
        print('WARNING: you need to provide either robot_pos or bot to reset \
              mujoco to a robot config')
        return 
    system.data.qvel = np.zeros(system.model.nv)
    mujoco.mj_forward(system.model, system.data)
    system.data.qacc = np.zeros(system.model.nv)

def construct_mj_ctrl_from_traj(system, q_traj):
    dq_traj = np.concatenate((np.zeros((1,7)), np.diff(q_traj, axis=0)/system.model.opt.timestep))
    q_ctrl = np.concatenate([q_traj, np.zeros((q_traj.shape[0], 1)), dq_traj], axis=1)
    return q_ctrl

def compute_gt_rollout(system, q_traj, initial_obj_pose):
    q_ctrl = construct_mj_ctrl_from_traj(system, q_traj)
    qv_qa_vec = np.zeros(system.model.nv + system.model.na)
    initial_state = np.concatenate((
        [0], q_traj[0], np.zeros(2), 
        initial_obj_pose,
        qv_qa_vec
        ))
    rollout_ctrl = np.tile(q_ctrl, (1, 1))
    rollout_states, _ = rollout.rollout(system.model, system.data, 
                                        initial_state=initial_state, 
                                        control=rollout_ctrl)
    return rollout_states[:, :, 1:]

def compute_particle_rollouts(system, candidates):
    rollouts = []
    for i in range(len(candidates)):
        q_traj = candidates[i]     
        system.particles[:, :7] = q_traj[0]
        q_ctrl = construct_mj_ctrl_from_traj(system, q_traj)
        initial_state = np.concatenate((np.zeros((system.num_particles+1, 1)), system.particles), axis=1)
        mj_ctrl = np.tile(q_ctrl, (system.num_particles+1,1, 1))
        rollout_states, _ = rollout.rollout(system.model, system.data, 
                                            initial_state=initial_state, 
                                            control=mj_ctrl)
        rollouts.append(rollout_states[:, :, 1:])
    return rollouts

def sync_robot_motion_with_mujoco(system, q_ref, dq_ref, dt_control, dt_mj):
    num_physics_updates = int(dt_control/dt_mj)
    for i in range(q_ref.shape[0]):
        system.data.ctrl[:7] = q_ref[i]
        system.data.ctrl[7:-1] = dq_ref[i]
        for _ in range(num_physics_updates):
            mujoco.mj_step(system.model, system.data)    
            time.sleep(dt_mj)
        time.sleep(dt_control)
        system.viewer.sync() 

def add_trajectory_to_scene(system, q_traj, scene=None, num_ee_points=20, start_idx=None, color=[1, 1, 0, 1]):
    if scene is None:
        scene = system.viewer.user_scn
        scene.ngeom = system.num_particles + num_ee_points + 1 # +1 for workspace bounds, num_ee_points from candidates
    if start_idx is None:   
        start_idx = scene.ngeom - 1
        scene.ngeom += q_traj.shape[0]
    for i in range(q_traj.shape[0]):
        x,_ = system.fk_pose(q_traj[i])

        mujoco.mjv_initGeom(
            scene.geoms[start_idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=x, 
            mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            rgba=color,
        )
        start_idx += 1
    if scene is None:
        system.viewer.sync()

def add_box_gt_to_scene(system, obj_dims, T_base2box, scene=None, start_idx=None):
    if scene is None:
        scene = system.viewer.user_scn 
    box_size = obj_dims/2 # using half size for mujoco
    if start_idx is None:
        start_idx = scene.ngeom
        scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[start_idx],
        type=mujoco.mjtGeom.mjGEOM_BOX, 
        size=box_size,  
        pos=T_base2box[:3,3], 
        mat=T_base2box[:3,:3].reshape((9,)),
        rgba = [1, 0.4, 0.6, 1], # pink
    )

def add_box_particle_belief_to_scene(system, particle_states, weights,
                                     box_size, scene=None, start_idx=None, 
                                     alpha=None):
    if scene is None: 
        scene = system.viewer.user_scn 
        # scene.ngeom = 0
    if start_idx is None:
        start_idx = scene.ngeom
    if scene.ngeom <= 1: 
        scene.ngeom += particle_states.shape[0] 
    
    box_size = box_size/2 # using half size for mujoco
    T_box2mjbox = np.eye(4)
    T_box2mjbox[:3,3] = np.array([0,0,0])
    num_geoms = start_idx #1 #scene.ngeom
    # w = np.concatenate(([1.], weights))
    w = weights
    for i in range(len(weights)): # exclude the non-contact reference particle 
        p = particle_states[i].copy()
        T_world2box = np.eye(4)
        T_world2box[:3,3] = p[system.model.nq-7:system.model.nq-4]
        quat = p[system.model.nq-4:system.model.nq]
        quat = quaternion.from_float_array(quat)
        rot = quaternion.as_rotation_matrix(quat)
        T_world2box[:3,:3] = rot
        T_world2mjbox = T_world2box @ T_box2mjbox
        if alpha is not None:
            c = [1, 0, 0, alpha]
        elif w[i] < 1e-3: # particles that died out will be orange 
            c = [1, 0.5, 0, 0.05]
        elif w[i] == 1: 
            # this is gt particle 
            c = [1, 0.4, 0.6, 1] # pink
        else: # particles that are alive will be blue
            c = [0, 0, 1., 1.]
            c[-1] = np.min([1, 5.0 * w[i]])
        idx = num_geoms + i
        mujoco.mjv_initGeom(
            scene.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_BOX, 
            size=box_size,  
            pos=T_world2mjbox[:3,3], 
            mat=T_world2mjbox[:3,:3].flatten(),
            rgba=c,  
        )
        # num_geoms += 1

def add_grasp_particle_to_scene(system, particle, box_size, scene=None, start_idx=None): 
    if scene is None:
        scene = system.viewer.user_scn 
    if start_idx is None:
        start_idx = scene.ngeom
        scene.ngeom += 1
    box_size = box_size/2 # using half size for mujoco
    T_box2mjbox = np.eye(4)
    T_box2mjbox[:3,3] = np.array([0,0,0])
    p = particle.copy()
    T_world2box = np.eye(4)
    T_world2box[:3,3] = p[:3]
    quat = quaternion.from_float_array(p[3:])
    rot = quaternion.as_rotation_matrix(quat)
    T_world2box[:3,:3] = rot
    T_world2mjbox = T_world2box @ T_box2mjbox
    mujoco.mjv_initGeom(
        scene.geoms[start_idx],
        type=mujoco.mjtGeom.mjGEOM_BOX, 
        size=box_size,  
        pos=T_world2mjbox[:3,3], 
        mat=T_world2mjbox[:3,:3].flatten(),
        rgba=[1, 1, 0, 1], # yellow  
    )

def sim_and_show_candidate_traj(system, q_traj, particle_rollout=False, 
                                obj_type='box', obj_dims=None): 
    # add_box_particle_belief_to_scene(system, system.particles[:-1], system.weights, obj_dims)
    add_trajectory_to_scene(system, q_traj)
    q0 = q_traj[0]
    mujoco.mju_copy(system.data.qpos, np.concatenate(
        [q0, np.zeros(2), system.particles[-1, system.model.nq-7:system.model.nq]]))
    mujoco.mj_forward(system.model, system.data)
    dq_traj = np.concatenate((np.zeros((1,7)), np.diff(q_traj, axis=0)/system.model.opt.timestep))
    q_ctrl = np.concatenate([q_traj, np.zeros((q_traj.shape[0], 1)), dq_traj], axis=1)
    if particle_rollout:
        rollout_ctrl = np.tile(q_ctrl, (system.num_particles, 1))
        rollout_particles, _ = mujoco.rollout.rollout(system.model, system.data, 
                                                      initial_state=system.particles[:-1].copy(), 
                                                      ctrl=rollout_ctrl)
        rollout_particles = rollout_particles.reshape(
            (system.num_particles, -1, system.model.nq+system.model.nv+system.model.na))
    for i in range(1, q_ctrl.shape[0]):
        system.data.ctrl = q_ctrl[i]
        mujoco.mj_step(system.model, system.data)
        if particle_rollout: 
            add_box_particle_belief_to_scene(system, rollout_particles[:, i], 
                                             system.weights, obj_dims)
        system.viewer.sync()
        time.sleep(system.model.opt.timestep)

