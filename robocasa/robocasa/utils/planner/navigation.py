from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored
from copy import deepcopy


robot0_history = []
robot1_history = [] # TODO: more elegant way to handle this?

class NavigationPlanner:
    def __init__(self, env, obs, extra_para, id=0):
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        self.pid_eef_pos_ctlr = deepcopy(CU.pid_eef_pos_ctlr)
        self.pid_eef_axisangle_ctlr = deepcopy(CU.pid_eef_axisangle_ctlr)
        self.pid_base_pos_ctlr = deepcopy(CU.pid_base_pos_ctlr)
        self.pid_base_ori_ctlr = deepcopy(CU.pid_base_ori_ctlr)
        self.pid_base_height_ctlr = deepcopy(CU.pid_base_height_ctlr)
        
        self.task_stage = 0
        self.id = id
        
        global robot0_history
        global robot1_history
        
        # initialize history if history is empty
        if self.id == 0 and not robot0_history:
            robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos))
            robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos))
        elif self.id == 1 and not robot0_history:
            robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos))
            robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos))
        
        # get target fixture randomly
        if extra_para == None:
            
            fixtures = list(env.fixtures.values())
            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls for cls in fxtr_classes if fxtr_classes.count(cls) == 1 and cls in [
                    "CoffeeMachine", "Toaster", "Stove", "Stovetop", "OpenCabinet",
                    "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
                ]
            ]
            while True:
                self.target_fixture = env.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if fxtr_class not in valid_target_fxtr_classes:
                    continue
                break
                
            self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(self.target_fixture)
        
        # navigate back to the second to last position
        elif extra_para == 'back':
            if self.id == 0:
                self.target_pos, self.target_ori = robot0_history[-2]
                robot0_history.append((self.target_pos, self.target_ori))
            elif self.id == 1:
                self.target_pos, self.target_ori = robot1_history[-2]
                robot1_history.append((self.target_pos, self.target_ori))
        
        # navigate to the given fixture or object
        else:
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
            
            if extra_para in object_keys: # navigate to the given object
                obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
                obj = env.objects[obj_str] # rot
                obj.pos = obs[obj_str + '_pos'] # pos
                self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(obj)
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
                
            elif extra_para in fixture_keys: # navigate to the given fixture
                fixture_str = extra_para
                for fxtr in fixtures:
                    if type(fxtr).__name__.lower() == fixture_str:
                        fixture = fxtr
                self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(fixture)
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
            
            else:
                raise ValueError(f'there is no fixture or object {extra_para} in the environment!')
        
        # get base position and orientation
        base_pos = obs[f'robot{self.id}_base_pos'][:2]
        base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
        
        # get target position and orientation
        target_pos = self.target_pos[:2]
        target_ori = self.target_ori[2]
        
        # calculate middle point 1 and middle point 2
        self.init_pos = base_pos
        self.dist = 0.5
        self.middle1_pos = np.array([1.5, -1.5])
        self.middle1_pos[0] = base_pos[0] - self.dist * np.cos(base_ori)
        self.middle1_pos[1] = base_pos[1] - self.dist * np.sin(base_ori)
        self.middle2_pos = np.array([1.5, -1.5])
        self.middle2_pos[0] = target_pos[0] - self.dist * np.cos(target_ori)
        self.middle2_pos[1] = target_pos[1] - self.dist * np.sin(target_ori)
    
    def get_control(self, env=None, obs=None):
        """
        control method designed for navigation task
        """
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        end_control = False
    
        # get base position and orientation
        base_pos = obs[f'robot{self.id}_base_pos'][:2]
        base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
        
        # get target position and orientation
        target_pos = self.target_pos[:2]
        target_ori = self.target_ori[2]
        
        # move to middle position 1
        if self.task_stage == 0:
            action = self.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=self.middle1_pos)
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, id=self.id)
            if np.linalg.norm(self.middle1_pos - base_pos) <= 0.05:
                self.task_stage += 1
                self.pid_base_pos_ctlr.reset()
        
        # then turn the orientation
        elif self.task_stage == 1:
            tz = self.pid_base_ori_ctlr.compute(current_value=base_ori, target_value=target_ori)
            action = CU.map_action(tz, base_ori) # 1-dim
            action = CU.create_action(base_ori=action, joint="stable", id=self.id)
            if np.cos(target_ori - base_ori) >= 0.998:
                self.task_stage += 1
                self.pid_base_ori_ctlr.reset()
        
        # then move to middle position 2
        elif self.task_stage == 2:
            action = self.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=self.middle2_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, id=self.id)
            if np.linalg.norm(self.middle2_pos - base_pos) <= 0.05:
                self.task_stage += 1
                self.pid_base_pos_ctlr.reset()
            
        # finally move to target position
        elif self.task_stage == 3:
            action = self.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, id=self.id)
            if np.linalg.norm(target_pos - base_pos) <= 0.05:
                # reset all planner related infomation 
                self.task_stage = 0 
                self.init_pos = None
                self.middle1_pos = None
                self.middle2_pos = None
                self.pid_base_pos_ctlr.reset()
                end_control = True
        
        info = {'end_control': end_control, 'arm_need_reset': False}
        return action, info