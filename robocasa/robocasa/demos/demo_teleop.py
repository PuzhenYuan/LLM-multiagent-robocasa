import argparse
import json
import time
from collections import OrderedDict
from termcolor import colored

import robosuite
from robosuite import load_controller_config
from robocasa.scripts.collect_demos import collect_human_trajectory
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper

from robocasa.models.arenas.layout_builder import STYLES

def choose_option(options, option_name, show_keys=False, default=None, default_message=None):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input("Choose an option 0 to {}, or any other key for default ({}): ".format(
            len(options) - 1,
            default_message,
        ))
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task (choose among 100+ tasks)")
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"])
    args = parser.parse_args()


    tasks = OrderedDict([
        ("PnPCounterToCab", "pick and place from counter to cabinet"),
        ("PnPCounterToSink", "pick and place from counter to sink"),
        ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
        ("PnPStoveToCounter", "pick and place from stove to counter"),
        ("OpenSingleDoor", "open cabinet or microwave door"),
        ("CloseDrawer", "close drawer"),
        ("TurnOnMicrowave", "turn on microwave"),
        ("TurnOnSinkFaucet", "turn on sink faucet"),
        ("TurnOnStove", "turn on stove"),
        ("ArrangeVegetables", "arrange vegetables on a cutting board"),
        ("MicrowaveThawing", "place frozen food in microwave for thawing"),
        ("RestockPantry", "restock cans in pantry"),
        ("PreSoakPan", "prepare pan for washing"),
        ("PrepareCoffee", "make coffee"),
        ("PnPCounterToStove", "pick and place from counter to stove"), # added
        ("NavigateKitchen", "navigation in the kitchen"), # added
        ("MultistepSteaming", "multistep steaming"), # added
        
        ("ArrangeItems", "arrange items in the kitchen, " + colored("self designed", "yellow")), # added, self designed
    ])

    styles = OrderedDict()
    for k in sorted(STYLES.keys()):
        styles[k] = STYLES[k]

    if args.task is None:
        args.task = choose_option(tasks, "task", default="PnPCounterToCab", show_keys=True)

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": "PandaMobile",
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "layout_ids": args.layout,
        "style_ids": args.style,
        "translucent_robot": True,
    }

    args.renderer = "mjviewer"
    
    print(colored(f"Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=(args.renderer != "mjviewer"),
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)
    
    # Wrap the environment with data collection wrapper
    import datetime
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y-%m-%d-%H-%M-%S')
    tmp_directory = "/tmp/{}".format(time_str)
    env = DataCollectionWrapper(env, tmp_directory)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
        )
    else:
        raise ValueError

    excluded_eps = []
    
    # collect demonstrations
    while True:
        ep_directory, discard_traj = collect_human_trajectory(
            env, device, "right", "single-arm-opposed", mirror_actions=True, render=(args.renderer != "mjviewer"),
            max_fr=30,
        )
        print()
        
        if discard_traj:
            continue
        
        keep = input('keep the trajectory? (y/n): ')
        if keep == 'y' and ep_directory is not None:
            
            import os
            from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5
            
            # excluded_eps.append(ep_directory.split('/')[-1])
            new_dir = os.path.join('/home/ypz/msclab/robocasa_space/robocasa/robocasa/models/assets/demonstrations_private', args.task, time_str)
            os.makedirs(new_dir)
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps)
            break
