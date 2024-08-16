import os
import argparse
import json
import time
from collections import OrderedDict
from termcolor import colored

import robosuite
from robosuite import load_controller_config
from robocasa.scripts.collect_demos import collect_human_trajectory, gather_demonstrations_as_hdf5
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper

import robocasa
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
        ("PnPCabToConuter", "pick and place from cabinet to counter"),
        ("PnPCounterToSink", "pick and place from counter to sink"),
        ("PnPSinkToCounter", "pick and place from sink to counter"),
        ("PnPCounterToMicrowave", "pick and place from counter to microwave"),
        ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
        ("PnPCounterToStove", "pick and place from counter to stove"),
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
        ("NavigateKitchen", "navigation in the kitchen"), # added
        ("MultistepSteaming", "multistep steaming"), # added
        
        # added, self designed
        ("ArrangeItems", "one agent arranges items in the kitchen, " + colored("one agent singletask", "yellow")), 
        ("TwoAgentArrange", "two agents arrange items in the kitchen, " + colored("two agents singletask", "yellow")),
        ("OpenMicrowavePnP", "open microwave door, pick the food from the counter and place it in the microwave, " + colored("one agent multitask", "yellow")),
    ])

    styles = OrderedDict()
    for k in sorted(STYLES.keys()):
        styles[k] = STYLES[k]

    if args.task is None:
        args.task = choose_option(tasks, "task", default="PnPCounterToCab", show_keys=True)

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": ["PandaMobile", "PandaMobile"] if args.task == 'TwoAgentArrange' else "PandaMobile", 
        # "PandaMobile", "VX300SMobile" are OK, while other robots may raise action space not compatible error
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "layout_ids": args.layout,
        "style_ids": args.style,
        "translucent_robot": True,
    }

    # args.renderer = "mjviewer" # pressing [ or ] to change all camera view
    args.renderer = "mujoco"
    
    print(colored(f"Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=(args.renderer != "mjviewer"),
        has_offscreen_renderer=False,
        render_camera="robot0_agentview_center", # important, which camera to be used, "robot0_frontview" by default
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
        try:
            ep_directory, discard_traj = collect_human_trajectory(
                env, device, "right", "single-arm-opposed", mirror_actions=True, render=(args.renderer != "mjviewer"),
                max_fr=30,
            )
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt, stop collecting trajectory')
            break
         
        if discard_traj: # reset env
            print('Reset the environment...')
            if ep_directory is not None:
                excluded_eps.append(ep_directory.split('/')[-1])
        else:
            time.sleep(1)
            keep = input('Keep the trajectory? (y/n): ')
            if keep == 'y' and ep_directory is not None: # save the trajectory
                print('Save the trajectory...')
            elif keep == 'n' and ep_directory is not None: # discard the trajectory
                print('Discard the trajectory...')
                excluded_eps.append(ep_directory.split('/')[-1])
            else:
                print('Invalid input, discard the trajectory by default')
                if ep_directory is not None:
                    excluded_eps.append(ep_directory.split('/')[-1])
        
        num_collected = len([name for name in os.listdir(tmp_directory)]) - len(excluded_eps)
        print('\nNumber of demonstrations collected: {}'.format(num_collected))
        finish = input('Finish the demonstration collection? (y/n): ')
        if finish == 'y':
            print('\nDemonstration collection finished. Start gathering demostration and save to hdf5...')
            this_directory = os.path.abspath(os.path.dirname(__file__))
            new_dir = os.path.join(robocasa.models.assets_root, "demonstrations_private", args.task, time_str)
            os.makedirs(new_dir)
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps)
            break
        else:
            print('Continue collecting demonstrations...\n')
            continue
            
            
