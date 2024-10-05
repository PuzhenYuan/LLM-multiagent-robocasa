from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored
from copy import deepcopy


class PlaceToPlanner:
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
        
        self.init_eef_pos = obs[f'robot{self.id}_eef_pos']
        self.init_eef_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
        
        # get object and fixture keys
        object_keys = env.objects.keys()
        fixtures = list(env.fixtures.values())
        fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
        valid_target_fxtr_classes = [
            cls for cls in fxtr_classes if fxtr_classes.count(cls) == 1 and cls in [
                "CoffeeMachine", "Toaster", "Stove", "Stovetop", "OpenCabinet",
                "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
            ]
        ]
        fixture_keys = [fxtr.lower() for fxtr in valid_target_fxtr_classes]
        
        # if extra_para is object keys, place to object surface
        if extra_para in object_keys:
            self.obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
            obj = env.objects[self.obj_str] # rot
            obj.pos = obs[self.obj_str + '_pos'] # pos
            # calculate placement position
            self.placement_pos = obs[self.obj_str + '_pos'] + np.array([0, 0, 0.05])
            self.above_placement_pos = self.placement_pos + np.array([0, 0, 0.10]) # move above the object first
        
        # if extra_para is fixture keys, place to fixture surface
        elif extra_para in fixture_keys:
            self.fxtr_str = extra_para
            for fxtr in fixtures:
                if type(fxtr).__name__.lower() == self.fxtr_str:
                    self.fixture = fxtr
            # calculate placement position
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            delta_pos_wrt_body =  np.array([-0.08, 0, 0.05]) # place near agent
            delta_pos_wrt_world = delta_pos_wrt_body @ \
            np.array( # rotate delta position based on base orientation
                [
                    [np.cos(base_ori), np.sin(base_ori), 0], 
                    [-np.sin(base_ori), np.cos(base_ori), 0], 
                    [0, 0, 1]
                ]
            )
            self.placement_pos = self.fixture.pos + delta_pos_wrt_world
            self.above_placement_pos = self.placement_pos + np.array([0, 0, 0.10]) # move above the fixture first
        
        else:
            raise ValueError(f'there is no {extra_para} in the environment!')
    
    def get_control(self, env=None, obs=None):
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        end_control = False
        
        # move the gripper to the above placement position
        if self.task_stage == 0:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.above_placement_pos
            
            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, grasp=True, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos_ctlr.reset()
        
        # move the gripper to the placement position
        if self.task_stage == 1:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.placement_pos
            
            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, grasp=True, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos_ctlr.reset()
                self.timmer = 10
        
        # open the gripper until not contact
        elif self.task_stage == 2:
            action = CU.create_action(grasp=False, id=self.id)
            self.timmer -= 1
            if self.timmer == 0:
                self.task_stage += 1
        
        # reset arm position to above placement position using control sequence
        elif self.task_stage == 3:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            eef_pos = obs[f'robot{self.id}_eef_pos']
            target_pos = self.above_placement_pos # move above the object

            action_pos = self.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs[f'robot{self.id}_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = self.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01: 
                self.task_stage += 1
                self.pid_eef_pos_ctlr.reset()
                self.pid_eef_axisangle_ctlr.reset()
        
        # reset arm position to initial position using control sequence
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
            
            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False, id=self.id)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                # reset all planner related infomation 
                self.task_stage = 0
                self.init_eef_pos = None
                self.init_eef_mat = None
                self.pid_eef_pos_ctlr.reset()
                self.pid_eef_axisangle_ctlr.reset()
                end_control = True
        
        info = {'end_control': end_control, 'arm_need_reset': True}
        return action, info