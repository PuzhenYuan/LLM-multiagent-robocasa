from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored
from copy import deepcopy


class WaitPlanner:
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
        
        self.id = id
    
    def get_control(self, env=None, obs=None):
        """
        control method designed for waiting...
        """
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        action = CU.create_action(id=self.id)
        info = {'end_control': True, 'arm_need_reset': False}
        return action, info