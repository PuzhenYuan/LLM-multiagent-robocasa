from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def open_microwave_door_checker(env):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "microwave")
    door_state = env.microwave.get_door_state(env=env)
    
    success = True
    for joint_p in door_state.values():
        if joint_p < 0.90:
            success = False
            break
    end_control = success
    return end_control