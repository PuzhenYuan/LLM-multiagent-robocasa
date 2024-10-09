import os
import time
import datetime
import shutil
import argparse
import json
import h5py
import imageio
import numpy as np
import traceback
from copy import deepcopy
from collections import OrderedDict
from termcolor import colored

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils # added
import robomimic.utils.env_utils as EnvUtils # added
import robomimic.utils.train_utils as TrainUtils # added
import robomimic.utils.log_utils as LogUtils # added

from robomimic.utils.dataset import SequenceDataset, R2D2Dataset, MetaDataset
from robomimic.envs.env_base import EnvBase, EnvType
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from tianshou.env import SubprocVectorEnv

import robosuite
from robosuite import load_controller_config

import robocasa.utils.control_utils as CU
import robocasa.utils.controller_dict as CD


def run_controlled_rollout_multitask_twoagent(
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        multiagent_config=None # to set observation util configs
    ):
    
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    ob_dict = env.reset()
    if render:
        env.render(mode="human")

    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = { k: False for k in env.is_success() } # success metrics

    end_step = None

    video_frames = []
    
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    horizon_reached = False
    task_i = -1
    
    # to render initial state
    for _ in range(1):
        ac0 = CU.create_action(grasp=False, id=0)
        ac1 = CU.create_action(grasp=False, id=1)
        ac = np.concatenate([ac0, ac1], axis=0)
        ob_dict, r, done, info = env.step(ac)
    
    # get all valid keys
    object_keys = env.env.env.objects.keys()
    fixtures = list(env.env.env.fixtures.values())
    fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
    valid_target_fxtr_classes = [
        cls for cls in fxtr_classes if fxtr_classes.count(cls) == 1 and cls in [
            "CoffeeMachine", "Toaster", "Stove", "Stovetop", "OpenCabinet",
            "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
        ]
    ]
    fixture_keys = [fxtr.lower() for fxtr in valid_target_fxtr_classes]
    
    # print available fixtures, objects, and commands
    print(colored("\nAvailable fixtures in env {}:".format(type(env.env.env).__name__), "yellow"))
    for key in fixture_keys:
        print(key)
    print(colored("\nAvailable objects in env {}:".format(type(env.env.env).__name__), "yellow"))
    for key in object_keys:
        print(key)
    print(colored("\nAvailable commands in controller dict:", "yellow"))
    for key in CD.controller_dict.keys():
        print(CD.controller_dict[key]["usage"])
    
    # start episode
    while True:
        task_i += 1
        end_control0 = False
        end_control1 = False
        
        # get language command and controller config for each agent
        print()
        
        try:
            lang_command0 = input(colored("Please enter agent0's command for task {}:\n".format(task_i), "yellow"))
            controller_config0, extra_para0 = CD.search_config(lang_command0, CD.controller_dict)
        except ValueError as e:
            print(colored("Error: agent0's {}".format(e), 'red'))
            continue
        try:
            lang_command1 = input(colored("Please enter agent1's command for task {}:\n".format(task_i), "yellow"))
            controller_config1, extra_para1 = CD.search_config(lang_command1, CD.controller_dict)
        except ValueError as e:
            print(colored("Error: agent1's {}".format(e), 'red'))
            continue
        
        # get policy or planner for each agent
        
        if controller_config0["type"] == "policy":
            env_lang0 = controller_config0["env_lang"]
            ckpt_path0 = controller_config0["ckpt_path"]
            policy0, ckpt_dict0 = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path0, device=device, verbose=False)
            ObsUtils.initialize_obs_utils_with_config(multiagent_config, verbose=False) # maintain obs_utils for all policies
            assert isinstance(policy0, RolloutPolicy)
            policy0.start_episode(lang=env_lang0)
            checker0 = controller_config0["checker"]
            
        elif controller_config0["type"] == "planner":
            obs = env.env.env.observation_spec()
            try:
                planner0 = controller_config0["planner"](env, obs, extra_para0, id=0)
            except ValueError as e:
                planner0 = None
                print(colored('Error: {}'.format(e), 'red'))
                continue
        
        if controller_config1["type"] == "policy":
            env_lang1 = controller_config1["env_lang"]
            ckpt_path1 = controller_config1["ckpt_path"]
            policy1, ckpt_dict1 = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path1, device=device, verbose=False)
            ObsUtils.initialize_obs_utils_with_config(multiagent_config, verbose=False) # maintain obs_utils for all policies
            assert isinstance(policy1, RolloutPolicy)
            policy1.start_episode(lang=env_lang1)
            checker1 = controller_config1["checker"]
            
        elif controller_config1["type"] == "planner":
            obs = env.env.env.observation_spec()
            try:
                planner1 = controller_config1["planner"](env, obs, extra_para1, id=1)
            except ValueError as e:
                planner1 = None
                print(colored('Error: {}'.format(e), 'red'))
                continue

        # start control loop
        for step_i in range(horizon): #LogUtils.tqdm(range(horizon)):
            
            # use policy or planner to get action
            
            if end_control0:
                ac0 = CU.create_action(id=0)
            else:
                if controller_config0["type"] == "policy":
                    # policy_ob = ob_dict
                    policy_ob0 = {key: value for key, value in ob_dict.items() if not key.startswith('robot1')}
                    ac0 = policy0(policy_ob0, goal_dict)
                    end_control0 = checker0(env)
                    arm_need_reset0 = True
                elif controller_config0["type"] == "planner":
                    obs = env.env.env.observation_spec()
                    ac0, control_info0 = planner0.get_control(env=env, obs=obs)
                    end_control0 = control_info0["end_control"]
                    arm_need_reset0 = control_info0["arm_need_reset"]
            
            if end_control1:
                ac1 = CU.create_action(id=1)
            else:
                if controller_config1["type"] == "policy":
                    # policy_ob = ob_dict
                    policy_ob1 = {key.replace('robot1', 'robot0'): value for key, value in ob_dict.items() if not key.startswith('robot0')}
                    ac1 = policy1(policy_ob1, goal_dict)
                    end_control1 = checker1(env)
                    arm_need_reset1 = True
                elif controller_config1["type"] == "planner":
                    obs = env.env.env.observation_spec()
                    ac1, control_info1 = planner1.get_control(env=env, obs=obs)
                    end_control1 = control_info1["end_control"]
                    arm_need_reset1 = control_info1["arm_need_reset"]
                
            # play action
            ac = np.concatenate([ac0, ac1], axis=0)
            ob_dict, r, done, info = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human") # can change camera here, or by default the first camera in camera list

            # compute reward
            rews.append(r)

            # cur_success_metrics = env.is_success()
            cur_success_metrics = info["is_success"]

            if success is None:
                success = deepcopy(cur_success_metrics)
            else:
                for k in success:
                    success[k] = success[k] | cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    frame = env.render(mode="rgb_array", height=512, width=512)
                    video_frames.append(frame)
                video_count += 1

            # break if done or success or end control
            if done or (terminate_on_success and success["task"]) or (end_control0 and end_control1):
                if end_step == None:
                    end_step = step_i
                else:
                    end_step += step_i
                
                if verbose:
                    print(colored("Success: agent0's task {} succeeded".format(task_i), 'green'))
                    print(colored("Success: agent1's task {} succeeded".format(task_i), 'green'))
                
                # delete policy or planner objects
                if controller_config0["type"] == "policy":
                    del policy0 # to avoid huge gpu memory usage
                    del checker0
                elif controller_config0["type"] == "planner":
                    del planner0
                if controller_config1["type"] == "policy":
                    del policy1 # to avoid huge gpu memory usage
                    del checker1
                elif controller_config1["type"] == "planner":
                    del planner1

                # reset arm to initial position and orientation
                if arm_need_reset0:
                    initial_qpos=(-0.01612974, -1.03446714, -0.02397936, -2.27550888, 0.03932365, 1.51639493, 0.69615947)
                    env.env.env.robots[0].set_robot_joint_positions(initial_qpos)
                if arm_need_reset1:
                    initial_qpos=(-0.01612974, -1.03446714, -0.02397936, -2.27550888, 0.03932365, 1.51639493, 0.69615947)
                    env.env.env.robots[1].set_robot_joint_positions(initial_qpos)
                if arm_need_reset0 or arm_need_reset1:
                    ac0 = CU.create_action(grasp=False, id=0)
                    ac1 = CU.create_action(grasp=False, id=1)
                    ac = np.concatenate([ac0, ac1], axis=0)
                    ob_dict, r, done, info = env.step(ac)
                break
            
            # break if horizon reached and task failed
            if step_i == horizon - 1:
                horizon_reached = True
                if end_step == None:
                    end_step = step_i
                else:
                    end_step += step_i
                    
                if verbose:
                    if not end_control0 and end_control1:
                        print(colored("Failure: horizon reached, agent0's task {} failed".format(task_i), 'red'))
                        print(colored("Success: agent1's task {} succeeded".format(task_i), 'green'))
                    elif not end_control1 and end_control0:
                        print(colored("Success: agent0's task {} succeeded".format(task_i), 'green'))
                        print(colored("Failure: horizon reached, agent1's task {} failed".format(task_i), 'red'))
                    elif not end_control0 and not end_control1:
                        print(colored("Failure: horizon reached, agent0's task {} failed".format(task_i), 'red'))
                        print(colored("Failure: horizon reached, agent1's task {} failed".format(task_i), 'red'))
                
                # delete policy or planner objects
                if controller_config0["type"] == "policy":
                    del policy0 # to avoid huge gpu memory usage
                    del checker0
                elif controller_config0["type"] == "planner":
                    del planner0
                if controller_config1["type"] == "policy":
                    del policy1 # to avoid huge gpu memory usage
                    del checker1
                elif controller_config1["type"] == "planner":
                    del planner1
                
                # reset arm to initial position and orientation
                if arm_need_reset0:
                    initial_qpos=(-0.01612974, -1.03446714, -0.02397936, -2.27550888, 0.03932365, 1.51639493, 0.69615947)
                    env.env.env.robots[0].set_robot_joint_positions(initial_qpos)
                if arm_need_reset1:
                    initial_qpos=(-0.01612974, -1.03446714, -0.02397936, -2.27550888, 0.03932365, 1.51639493, 0.69615947)
                    env.env.env.robots[1].set_robot_joint_positions(initial_qpos)
                if arm_need_reset0 or arm_need_reset1:
                    ac0 = CU.create_action(grasp=False, id=0)
                    ac1 = CU.create_action(grasp=False, id=1)
                    ac = np.concatenate([ac0, ac1], axis=0)
                    ob_dict, r, done, info = env.step(ac)
                break
            
        if success["task"] == True:
            if verbose:
                print(colored('All task success in a multitask rollout!!!', 'green'))
            break
        if done:
            if verbose:
                print('Done by some reasons')
            break
        if horizon_reached:
            continue # try again or change task

    # post process, write video, calculate returns, etc.
    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    end_step = end_step or step_i
    total_reward = np.sum(rews[:end_step + 1])
    
    results["Return"] = total_reward
    results["Horizon"] = end_step + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def run_controlled_multitask_twoagent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent[0] # should be a list
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict) # TODO: change config to handle multiagent observation
    
    # initialize obs_utils, related to ob_dict in env.step return
    multiagent_config = config
    robot0_lowdim_obs_list = config["observation"]["modalities"]["obs"]["low_dim"]
    robot1_lowdim_obs_list = [key.replace('robot0', 'robot1') for key in robot0_lowdim_obs_list]
    multiagent_config["observation"]["modalities"]["obs"]["low_dim"] = robot0_lowdim_obs_list + robot1_lowdim_obs_list
    robot0_rgb_obs_list = config["observation"]["modalities"]["obs"]["rgb"]
    robot1_rgb_obs_list = [key.replace('robot0', 'robot1') for key in robot0_rgb_obs_list]
    multiagent_config["observation"]["modalities"]["obs"]["rgb"] = robot0_rgb_obs_list + robot1_rgb_obs_list
    ObsUtils.initialize_obs_utils_with_config(multiagent_config, verbose=False) # use this to maintain obs_utils
    
    assert args.env.startswith("TwoAgent") # only support two agent tasks
    
    # args.renderer = "mujoco" # off-screen render, and write to video
    # args.renderer = "mjviewer" # on-screen render
    
    # initialize env_kwargs, maybe need more keys, if raise error, refer to multi_teleop_test.py
    env_kwargs = {
        "env_name": args.env,
        "robots": ["PandaMobile", "PandaMobile"], # "PandaMobile", "VX300SMobile" are OK, while other robots may raise action space not compatible error
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "layout_ids": None,
        "style_ids": None,
        "has_renderer": (args.renderer != "mjviewer"),
        "has_offscreen_renderer": False,
        "render_camera": None, # "robot0_agentview_center", # important, which camera to be used, "robot0_frontview" by default
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": 20,
        "renderer": args.renderer,
        "camera_names": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"] + \
                        ["robot1_agentview_left", "robot1_agentview_right", "robot1_eye_in_hand"],
        "camera_heights": 128,
        "camera_widths": 128,
        "translucent_robot": True,
        "layout_ids": -2, # added for navigation
    }
    
    # create environment from args.env
    env = EnvUtils.create_env(
        env_type=EnvType.ROBOSUITE_TYPE,
        render=args.render, 
        render_offscreen=(args.video_path is None), 
        use_image_obs=write_video,
        env_lang=None, # None by default, and will assign to env._ep_lang_str if not none, refer to env_robosuite.py
        **env_kwargs,
    )
    
    # handle environment wrappers
    from robomimic.envs.wrappers import FrameStackWrapper
    frame_stack = 10 # from auto gen config file
    env = FrameStackWrapper(env, num_frames=frame_stack)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # can choose to use logger
    data_logger = None

    # -------------------------------------------- copied from TrainUtils.rollout_with_stats() --------------------------------------------- #
    ### passing some parameters ###
    envs = [env]
    horizon = args.horizon
    use_goals = False
    num_episodes = args.n_rollouts
    render = args.render
    video_dir = args.video_path if write_video else None
    epoch = 4 # epoch number can be assigned manually
    video_skip = args.video_skip
    terminate_on_success = True
    del_envs_after_rollouts = True
    verbose = args.verbose
    ### end of passing parameters ###

    all_rollout_logs = OrderedDict()

    if isinstance(horizon, list):
        horizon_list = horizon
    else:
        horizon_list = [horizon]

    for env, horizon in zip(envs, horizon_list):

        env_name = env.name

        if video_dir is not None:
            # video is written per env
            video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
            video_path = os.path.join(video_dir, "{}_{}{}".format(env_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), video_str))
            video_writer = imageio.get_writer(video_path, fps=20)
            
        env_video_writer = None
        if write_video:
            print("video writes to " + video_path)
            env_video_writer = imageio.get_writer(video_path, fps=20)

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env_name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            if verbose:
                print("\nStarting episode {}...".format(ep_i + 1))
            try:
                rollout_info = run_controlled_rollout_multitask_twoagent(
                    env=env,
                    horizon=horizon,
                    render=render,
                    use_goals=use_goals,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                    verbose=verbose,
                    multiagent_config=multiagent_config # to set observation util configs
                )
            except Exception as e:
                print("Rollout exception at episode number {}!".format(ep_i))
                print(traceback.format_exc())
                break
            
            rollout_info["time"] = time.time() - rollout_timestamp

            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                # print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        if len(rollout_logs) > 0:
            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
            all_rollout_logs[env_name] = rollout_logs_mean
        else:
            all_rollout_logs[env_name] = {"Time_Episode": -1, "Return": -1, "Success_Rate": -1, "time": -1}

        if del_envs_after_rollouts:
            # delete the environment after use
            del env

        if data_logger is not None:
            # summarize results from rollouts to tensorboard and terminal
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                else:
                    data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            print('Env: {}'.format(env_name))
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()
    
    # -------------------------------------------- copied from TrainUtils.rollout_with_stats() --------------------------------------------- #
    
    print('\nAll rollout logs:')
    print(all_rollout_logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        nargs='+',
        default=['/home/ypz/project/model_wash4task_epoch_150.pth', '/home/ypz/project/model_steam4task_epoch_350.pth'],
        # required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=500, # None by default
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default='TwoAgentWashPnPSteam', # None
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/ypz/msclab/robocasa_space/test/",
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["robot0_frontview"], # ["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=2, # None by default
        help="(optional) set seed for rollouts",
    )
    
    # for debug use
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True, # None by default
        help="(optional) debug for rollouts",
    )
    
    # manually assign environment language for 
    parser.add_argument(
        "--env_lang",
        type=str,
        default=None, # None by default, should be like "agent0 task0_lang, agent1 task1_lang, ..."
        help="(optional) set language condition",
    )
    
    # set renderer: "mujoco" or "mjviewer"
    # "mujoco": off-screen render with fixed camera, and write to video
    # "mjviewer": on-screen render with more function, can toggle camera
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        choices=["mujoco", "mjviewer"],
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    args.render == False
    run_controlled_multitask_twoagent(args)
