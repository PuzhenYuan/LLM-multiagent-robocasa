from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored
from copy import deepcopy


class PickUpPlanner:
    def __init__(self, env, obs, extra_para, id=0):
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        self.pid_base_pos_ctlr = deepcopy(CU.pid_base_pos_ctlr)
        self.pid_base_ori_ctlr = deepcopy(CU.pid_base_ori_ctlr)
        self.pid_eef_pos_ctlr = deepcopy(CU.pid_eef_pos_ctlr)
        self.pid_eef_axisangle_ctlr = deepcopy(CU.pid_eef_axisangle_ctlr)
        
        self.task_stage = 0
        self.id = id
        
        object_keys = env.objects.keys()
        if extra_para in object_keys:
            self.obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
            obj = env.objects[self.obj_str] # rot
            obj.pos = obs[self.obj_str + '_pos'] # pos
        else:
            raise ValueError(f'there is no {extra_para} in the environment!')
        
        self.init_eef_pos = obs[f'robot{self.id}_eef_pos']
        self.init_eef_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
        self.obj_pos = obs[self.obj_str + '_pos']
        self.above_obj_pos = self.obj_pos + np.array([0, 0, 0.15]) # move above the object first
        self.above_obj_pos[2] = np.clip(self.above_obj_pos[2], 1, 1.1) # too high or low will cause collision with cabinet and sink, respectively

    def get_control(self, env=None, obs=None):
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        end_control = False
        obj = env.objects[self.obj_str]
    
        # move the gripper above the object position
        if self.task_stage == 0:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.above_obj_pos # move above the object
            
            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
            obj_wrt_world_mat = T.quat2mat(obs[self.obj_str + '_quat']) 
            # close to np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]]), so only extract z-axis rotation
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
            
            action_axisangle = self.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim

            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos_ctlr.reset()
                self.pid_eef_axisangle_ctlr.reset()
        
        # move the gripper to the object position
        elif self.task_stage == 1:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.obj_pos + np.array([0, 0, -0.0075])
            
            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
            obj_wrt_world_mat = T.quat2mat(obs[self.obj_str + '_quat']) 
            # close to np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]]), so only extract z-axis rotation
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
            
            action_axisangle = self.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim

            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos_ctlr.reset()
                self.pid_eef_axisangle_ctlr.reset()
                self.timmer = 10
        
        # close the gripper until contact
        elif self.task_stage == 2:
            action = CU.create_action(grasp=True, id=self.id)
            obj_contact = env.check_contact(obj, env.robots[self.id].gripper["right"])
            if obj_contact:
                self.timmer -= 1
            if self.timmer == 0:
                self.task_stage += 1
        
        # move the gripper above the object position
        elif self.task_stage == 3:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.above_obj_pos # move above the object
            
            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = self.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=True, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos_ctlr.reset()
                self.pid_eef_axisangle_ctlr.reset()
        
        # reset arm position using control sequence
        elif self.task_stage == 4:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.init_eef_pos
            
            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = self.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=True, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                # reset all planner related infomation 
                self.task_stage = 0
                self.init_eef_pos = None
                self.init_eef_mat = None
                self.pid_eef_pos_ctlr.reset()
                self.pid_eef_axisangle_ctlr.reset()
                end_control = True
        
        info = {'end_control': end_control, 'arm_need_reset': False}
        return action, info