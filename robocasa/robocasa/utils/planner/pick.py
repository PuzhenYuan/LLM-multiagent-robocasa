from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored

def pick_up_planner(env, obs, extra_para):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    if not hasattr(env, 'init_eef_pos'):
        env.init_eef_pos = None
    if not hasattr(env, 'init_eef_mat'):
        env.init_eef_mat = None
    if not hasattr(env, 'task_stage'): # set task_stage = 0 whenever a task is completed
        env.task_stage = 0
    
    object_keys = env.objects.keys()
    if extra_para in object_keys:
        obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
        obj = env.objects[obj_str] # rot
        obj.pos = obs[obj_str + '_pos'] # pos
    else:
        action = CU.create_action()
        info = {'end_control': True, 'arm_need_reset': False}
        print(colored(f'Failure: there is no {extra_para} in the environment!', 'red'))
        return action, info
    
    end_control = False
    
    # move the gripper to the object position
    if env.task_stage == 0:
        
        if env.init_eef_pos is None:
            env.init_eef_pos = obs['robot0_eef_pos']
        if env.init_eef_mat is None:
            env.init_eef_mat = T.quat2mat(obs['robot0_eef_quat'])
        
        # position control
        base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
        eef_pos = obs['robot0_eef_pos']
        target_pos = obs[obj_str + '_pos']
        
        action_pos = CU.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
        action_pos = CU.map_action(action_pos, base_ori) # 3-dim
        
        # orientation control
        eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
        obj_wrt_world_mat = T.quat2mat(obs[obj_str + '_quat']) 
        # close to np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]]), so only extract z axis rotation
        theta_z = np.arctan2(obj_wrt_world_mat[1, 0], obj_wrt_world_mat[0, 0])
        desired_obj_wrt_world_mat = np.array(
            [
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ]
        )
        target_obj_wrt_eef_mat = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) @ np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
        target_eef_wrt_world_mat = desired_obj_wrt_world_mat @ target_obj_wrt_eef_mat.T
        error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
        error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
        
        action_axisangle = CU.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
        action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim

        action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False)
        
        if np.linalg.norm(eef_pos - target_pos) < 0.01:
            env.task_stage += 1
            CU.pid_eef_pos_ctlr.reset()
            CU.pid_eef_axisangle_ctlr.reset()
            env.timmer = 10
    
    # close the gripper until contact
    elif env.task_stage == 1:
        action = CU.create_action(grasp=True)
        obj_contact = env.check_contact(obj, env.robots[0].gripper["right"])
        if obj_contact:
            env.timmer -= 1
        if env.timmer == 0:
            env.task_stage += 1
    
    # reset arm position using control sequence
    elif env.task_stage == 2:
        
        # position control
        base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
        eef_pos = obs['robot0_eef_pos']
        target_pos = env.init_eef_pos
        
        action_pos = CU.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
        action_pos = CU.map_action(action_pos, base_ori) # 3-dim
        
        # orientation control
        eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
        target_eef_wrt_world_mat = env.init_eef_mat
        error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
        error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
        
        action_axisangle = CU.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
        action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim
        
        action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=True)
        
        if np.linalg.norm(eef_pos - target_pos) < 0.01:
            # reset all planner related infomation 
            env.task_stage = 0
            env.init_eef_pos = None
            env.init_eef_mat = None
            CU.pid_eef_pos_ctlr.reset()
            CU.pid_eef_axisangle_ctlr.reset()
            end_control = True
    
    info = {'end_control': end_control, 'arm_need_reset': False}
    return action, info