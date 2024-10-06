import robocasa.utils.planner as planner
import robocasa.utils.checker as checker

import robocasa.utils.control_utils as CU
from collections import OrderedDict


def search_config(lang_command, controller_dict):
    """
    extract dict key and extra parameters from lang_command
    """
    find_key = False
    for key in controller_dict.keys():
        if lang_command.startswith(key):
            controller_config = controller_dict[key]
            remaining_str = lang_command.replace(key, "", 1).strip()
            find_key = True
            break
    if not find_key:
        raise ValueError("language command cannot match any controller")
    if remaining_str != "":
        extra_para = remaining_str
    else:
        extra_para = None
    return controller_config, extra_para


def get_recent_obs(ob_dict):
    """
    extract the recent and the newest observation from ob_dict
    """
    obs = {key: value[-1] for key, value in ob_dict.items()}
    return obs

controller_dict = OrderedDict() # order of the key is important

controller_dict['wait'] = {
    'type': 'planner',
    'planner': planner.WaitPlanner,
}

controller_dict['reset arm'] = {
    'type': 'planner',
    'planner': planner.ResetArmPlanner,
}

controller_dict['pick up'] = {
    'type': 'planner',
    'planner': planner.PickUpPlanner,
}

controller_dict['place to'] = {
    'type': 'planner',
    'planner': planner.PlaceToPlanner,
}

controller_dict['navigate to'] = {
    'type': 'planner',
    'planner': planner.NavigationPlanner,
}

controller_dict['navigation'] = {
    'type': 'planner',
    'planner': planner.NavigationPlanner,
}

controller_dict['open microwave door'] = {
    'type': 'policy',
    'ckpt_path': '/home/ypz/project/model_openpnp_autodl_epoch_700.pth',
    'env_lang': 'open the microwave door', 
    'checker': checker.open_microwave_door_checker,
}

controller_dict['pick the object from the counter and place it in the microwave'] = {
    'type': 'policy',
    'ckpt_path': '/home/ypz/project/model_openpnp_autodl_epoch_700.pth',
    'env_lang': 'pick the object from the counter and place it in the microwave',
    'checker': checker.open_microwave_door_checker, # TODO: better task design
}

# TODO: to avoid huge gpu memory usage, instore env_embedding instead of env_lang and discard encoder and discard encoder backbone

if __name__ == "__main__":
    print(controller_dict)
    print(search_config('pick up', controller_dict))