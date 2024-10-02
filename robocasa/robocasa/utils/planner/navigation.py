from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored
from copy import deepcopy


class NavigationPlanner:
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
    
        # navigate to the given fixture or item
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
                
            elif extra_para in fixture_keys: # navigate to the given fixture
                fixture_str = extra_para
                for fxtr in fixtures:
                    if type(fxtr).__name__.lower() == fixture_str:
                        fixture = fxtr
                self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(fixture)
            
            else:
                raise ValueError(f'there is no {extra_para} in the environment!')
        
        # get base position and orientation
        base_pos = obs[f'robot{self.id}_base_pos'][:2]
        base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
        
        # get target position and orientation
        target_pos = self.target_pos[:2]
        target_ori = self.target_ori[2]
        
        # calculate middle point 1 and middle point 2
        self.init_pos = base_pos
        self.middle1_pos = np.array([1.5, -1.5])
        self.middle1_pos[0] = base_pos[0] - 0.25 * np.cos(base_ori)
        self.middle1_pos[1] = base_pos[1] - 0.25 * np.sin(base_ori)
        self.middle2_pos = np.array([1.5, -1.5])
        self.middle2_pos[0] = target_pos[0] - 0.25 * np.cos(target_ori)
        self.middle2_pos[1] = target_pos[1] - 0.25 * np.sin(target_ori)
    
    def get_control(self, env=None, obs=None):
        
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
            action = CU.create_action(base_ori=action, id=self.id)
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


def navigation_planner(env, obs, extra_para):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    if not hasattr(env, 'init_pos'):
        env.init_pos = None
    if not hasattr(env, 'middle1_pos'):
        env.middle1_pos = None
    if not hasattr(env, 'middle2_pos'):
        env.middle2_pos = None
    if not hasattr(env, 'task_stage'): # set task_stage = 0 whenever a task is completed
        env.task_stage = 0
    if not hasattr(env, 'has_chosen_target'):
        env.has_chosen_target = False # set env.has_chosen_target = False when a navigation task is completed

    # choose target position to navigate to if hasn't chosen target
    if not env.has_chosen_target:
        
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
                env.target_fixture = env.rng.choice(fixtures)
                fxtr_class = type(env.target_fixture).__name__
                if fxtr_class not in valid_target_fxtr_classes:
                    continue
                break
                
            env.target_pos, env.target_ori = env.compute_robot_base_placement_pose(env.target_fixture)
            env.has_chosen_target = True
    
        # navigate to the given fixture or item
        else:
            fixture_keys = env.fixtures.keys()
            object_keys = env.objects.keys()
            
            if extra_para in object_keys: # navigate to the given object
                obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
                obj = env.objects[obj_str] # rot
                obj.pos = obs[obj_str + '_pos'] # pos
                env.target_pos, env.target_ori = env.compute_robot_base_placement_pose(obj)
                env.has_chosen_target = True
                
            elif extra_para in fixture_keys: # navigate to the given fixture, not all fixtures are supported
                # TODO: examine the extra_para, follow the random choose codes, to make it robust and easier to use
                fixture_str = extra_para
                fixture = env.fixtures[fixture_str]
                env.target_pos, env.target_ori = env.compute_robot_base_placement_pose(fixture)
                env.has_chosen_target = True
            
            else:
                action = CU.create_action()
                info = {'end_control': True, 'arm_need_reset': False}
                print(colored(f'Failure: there is no {extra_para} in the environment!', 'red'))
                return action, info
    
    assert env.has_chosen_target
    end_control = False
    
    # get base position and orientation
    base_pos = obs['robot0_base_pos'][:2]
    base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
    
    # get target position and orientation
    target_pos = env.target_pos[:2]
    target_ori = env.target_ori[2]
    
    # calculate middle point 1 and middle point 2
    if env.init_pos is None:
        env.init_pos = base_pos
    if env.middle1_pos is None:
        env.middle1_pos = np.array([1.5, -1.5])
        env.middle1_pos[0] = base_pos[0] - 0.25 * np.cos(base_ori)
        env.middle1_pos[1] = base_pos[1] - 0.25 * np.sin(base_ori)
    if env.middle2_pos is None:
        env.middle2_pos = np.array([1.5, -1.5])
        env.middle2_pos[0] = target_pos[0] - 0.25 * np.cos(target_ori)
        env.middle2_pos[1] = target_pos[1] - 0.25 * np.sin(target_ori)
    
    # move to middle position 1
    if env.task_stage == 0:
        action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=env.middle1_pos)
        action = CU.map_action(action, base_ori) # 2-dim
        action = CU.create_action(base_pos=action)
        if np.linalg.norm(env.middle1_pos - base_pos) <= 0.05:
            env.task_stage = 1
            CU.pid_base_pos_ctlr.reset()
    
    # then turn the orientation
    elif env.task_stage == 1:
        tz = CU.pid_base_ori_ctlr.compute(current_value=base_ori, target_value=target_ori)
        action = CU.map_action(tz, base_ori) # 1-dim
        action = CU.create_action(base_ori=action)
        if np.cos(target_ori - base_ori) >= 0.998:
            env.task_stage += 1
            CU.pid_base_ori_ctlr.reset()
    
    # then move to middle position 2
    elif env.task_stage == 2:
        action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=env.middle2_pos) # ground coordinates
        action = CU.map_action(action, base_ori) # 2-dim
        action = CU.create_action(base_pos=action)
        if np.linalg.norm(env.middle2_pos - base_pos) <= 0.05:
            env.task_stage += 1
            CU.pid_base_pos_ctlr.reset()
        
    # finally move to target position
    elif env.task_stage == 3:
        action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
        action = CU.map_action(action, base_ori) # 2-dim
        action = CU.create_action(base_pos=action)
        if np.linalg.norm(target_pos - base_pos) <= 0.05:
            # reset all planner related infomation 
            env.has_chosen_target = False
            env.task_stage = 0 
            env.init_pos = None
            env.middle1_pos = None
            env.middle2_pos = None
            CU.pid_base_pos_ctlr.reset()
            end_control = True
    
    info = {'end_control': end_control, 'arm_need_reset': False}
    return action, info