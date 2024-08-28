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

def batchify_obs(obs_list):
    """
    TODO: add comments
    """
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    
    return obs


def run_rollout_multitask_twoagent(
        policy_list, 
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        env_lang=None # "task0_lang, task1_lang, ..."
    ):

    assert isinstance(policy_list, list) # list of rollout policies
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper) or isinstance(env, SubprocVectorEnv)

    batched = isinstance(env, SubprocVectorEnv)

    ob_dict = env.reset()
    
    # seperate env language to language commands
    if env_lang is None:
        env_lang = env._ep_lang_str # contrain multiple tasks, seperated by comma
    lang_list = env_lang.split(', ')
    
    # seperate policy list to policy0 and policy1, and set language commands
    policy0 = policy_list[0]
    policy1 = policy_list[1]
    
    agent0_lang_list = [lang for lang in lang_list if lang.startswith("agent0")]
    agent1_lang_list = [lang for lang in lang_list if lang.startswith("agent1")]
    
    agent0_first_lang = agent0_lang_list[0].removeprefix("agent0").strip()
    agent1_first_lang = agent1_lang_list[0].removeprefix("agent1").strip()
    
    policy0.start_episode(lang=agent0_first_lang)
    policy1.start_episode(lang=agent1_first_lang)

    # e.g. env.is_success() = {'task':False, 'task1':False, 'task2':False}
    assert len(lang_list) == len(env.is_success()) - 1
    task_num = len(lang_list)
 
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = { k: False for k in env.is_success() } # success metrics

    if batched:
        end_step = [None for _ in range(len(env))]
    else:
        end_step = None

    if batched:
        video_frames = [[] for _ in range(len(env))]
    else:
        video_frames = []
    
    horizon_reached = False
    
    for task_i in range(task_num):
        
        # set active policy
        if lang_list[task_i].startswith("agent0"):
            command_lang = lang_list[task_i].removeprefix("agent0").strip()
            policy0.set_language(lang=command_lang)  # change to policy.start_episode()?
            policy = policy0
        elif lang_list[task_i].startswith("agent1"):
            command_lang = lang_list[task_i].removeprefix("agent1").strip()
            policy1.set_language(lang=command_lang)  # change to policy.start_episode()?
            policy = policy1
        else:
            raise ValueError("Language command must start with 'agent0' or 'agent1'")
        
        if verbose:
            print('Begin task{}: {}'.format(task_i, lang_list[task_i]))
        for step_i in range(horizon): #LogUtils.tqdm(range(horizon)):
            # get action from policy
            if batched:
                policy_ob = batchify_obs(ob_dict)
                ac = policy(ob=policy_ob, goal=goal_dict, batched=True) #, return_ob=True)
            else:
                policy_ob = ob_dict
                ac = policy(ob=policy_ob, goal=goal_dict) #, return_ob=True)

            # fill necessary zero actions
            if batched:
                raise NotImplementedError
            else:
                if lang_list[task_i].startswith("agent0"):
                    zero_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
                    ac = np.concatenate((ac, zero_action), axis=0)
                elif lang_list[task_i].startswith("agent1"):
                    zero_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
                    ac = np.concatenate((zero_action, ac), axis=0)
                else:
                    raise ValueError("Language command must start with 'agent0' or 'agent1'")
            
            # play action
            ob_dict, r, done, info = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            rews.append(r)

            # cur_success_metrics = env.is_success()
            if batched:
                cur_success_metrics = TensorUtils.list_of_flat_dict_to_dict_of_list([info[i]["is_success"] for i in range(len(info))])
                cur_success_metrics = {k: np.array(v) for (k, v) in cur_success_metrics.items()}
            else:
                cur_success_metrics = info["is_success"]

            if success is None:
                success = deepcopy(cur_success_metrics)
            else:
                for k in success:
                    success[k] = success[k] | cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    if batched:
                        # frames = env.render(mode="rgb_array", height=video_height, width=video_width)
                        
                        frames = []
                        policy_ob = deepcopy(policy_ob)
                        for env_i in range(len(env)):
                            cam_imgs = []
                            for im_name in ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]:
                                im = TensorUtils.to_numpy(
                                    policy_ob[im_name][env_i, -1]
                                )
                                im = np.transpose(im, (1, 2, 0))
                                if policy_ob.get("ret", None) is not None:
                                    im_ret = TensorUtils.to_numpy(
                                        policy_ob["ret"]["obs"][im_name][env_i,:,-1]
                                    )
                                    im_ret = np.transpose(im_ret, (0, 2, 3, 1))
                                    im = np.concatenate((im, *im_ret), axis=0)
                                cam_imgs.append(im)
                            frame = np.concatenate(cam_imgs, axis=1)
                            frame = (frame * 255.0).astype(np.uint8)
                            frames.append(frame)
                        
                        for env_i in range(len(env)):
                            frame = frames[env_i]
                            video_frames[env_i].append(frame)
                    else:
                        frame = env.render(mode="rgb_array", height=512, width=512)
                        video_frames.append(frame)

                video_count += 1

            # break if done
            if batched:
                for env_i in range(len(env)):
                    if end_step[env_i] is not None:
                        continue
                    
                    if done[env_i] or (terminate_on_success and success["task{}".format(task_i)][env_i]):
                        if task_i == 0:
                            end_step[env_i] = step_i
                        else:
                            end_step[env_i] += step_i
            else:
                if done or (terminate_on_success and success["task{}".format(task_i)]):
                    if task_i == 0:
                        end_step = step_i
                    else:
                        end_step += step_i
                    break
            
            if step_i == horizon - 1:
                horizon_reached = True
                if verbose:
                    print('Horizon reached, task{} failed'.format(task_i))
                break
        if [success["task{}".format(id)] for id in range(task_num)].count(True) == task_num:
            if verbose: 
                print('All task success in a multitask rollout!')
            break
        if done:
            if verbose:
                print('Done by some reasons')
            break
        if horizon_reached:
            break

    # post process, write video, calculate returns, etc.
    if video_writer is not None:
        if batched:
            for env_i in range(len(video_frames)):
                for frame in video_frames[env_i]:
                    video_writer.append_data(frame)
        else:
            for frame in video_frames:
                video_writer.append_data(frame)

    if batched:
        total_reward = np.zeros(len(env))
        rews = np.array(rews)
        for env_i in range(len(env)):
            end_step_env_i = end_step[env_i] or step_i
            total_reward[env_i] = np.sum(rews[:end_step_env_i+1, env_i])
            end_step[env_i] = end_step_env_i
        
        results["Return"] = total_reward
        results["Horizon"] = np.array(end_step) + 1
        results["Success_Rate"] = success["task"].astype(float)
    else:
        end_step = end_step or step_i
        total_reward = np.sum(rews[:end_step + 1])
        
        results["Return"] = total_reward
        results["Horizon"] = end_step + 1
        results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            if batched:
                results["{}_Success_Rate".format(k)] = success[k].astype(float)
            else:
                results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def run_trained_multitask_twoagent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent # should be a list

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore rollout policy
    policy0, ckpt_dict0 = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path[0], device=device, verbose=True)
    policy1, ckpt_dict1 = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path[1], device=device, verbose=True)
    
    # if rollout_horizon is None:
        # read horizon from config
    config0, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict0)
    config1, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict1)

    # create environment from saved checkpoint
    # env, _ = FileUtils.env_from_checkpoint(
    #     ckpt_dict=ckpt_dict, 
    #     env_name=args.env, 
    #     render=args.render, 
    #     render_offscreen=(args.video_path is None), 
    #     verbose=True,
    # )
    
    assert args.env.startswith("TwoAgent") # only support two agent tasks
    
    args.renderer = "mjviewer" # "mujoco" # or "mjviewer"
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
        "camera_names": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        "camera_heights": 128,
        "camera_widths": 128,
        "translucent_robot": False,
    }
    
    if args.env_lang is not None:
        env_lang = args.env_lang # "task0_lang, task1_lang, ..."
    else:
        env_lang = None
    
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
    env = EnvUtils.wrap_env_from_config(env, config=config0)  # apply environment warpper, if applicable, either config0 or config1 is ok

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # can choose to use logger, but need to change paths
    # data_logger = DataLogger(
    #     log_dir,
    #     config,
    #     log_tb=config.experiment.logging.log_tb,
    #     log_wandb=config.experiment.logging.log_wandb,
    # )
    # data_logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    data_logger = None

    # -------------------------------------------- copied from TrainUtils.rollout_with_stats() --------------------------------------------- #
    ### passing some parameters ###
    envs = [env]
    horizon = args.horizon
    use_goals = config0.use_goals or config1.use_goals
    num_episodes = args.n_rollouts
    render = args.render
    video_dir = args.video_path if write_video else None
    epoch = 3 # epoch number can be assigned manually
    video_skip = args.video_skip
    
    assert config0.experiment.rollout.terminate_on_success == config1.experiment.rollout.terminate_on_success
    terminate_on_success = config0.experiment.rollout.terminate_on_success or config1.experiment.rollout.terminate_on_success
    
    del_envs_after_rollouts = True
    verbose = args.verbose
    ### end of passing parameters ###
        
    assert isinstance(policy0, RolloutPolicy) and isinstance(policy1, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    if isinstance(horizon, list):
        horizon_list = horizon
    else:
        horizon_list = [horizon]

    for env, horizon in zip(envs, horizon_list):
        batched = isinstance(env, SubprocVectorEnv)

        if batched:
            env_name = env.get_env_attr(key="name", id=0)[0]
        else:
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
        if batched:
            iterator = range(0, num_episodes, len(env))
        else:
            iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            if verbose:
                print("\nStarting episode {}...".format(ep_i + 1))
            try:
                rollout_info = run_rollout_multitask_twoagent(
                    policy_list=[policy0, policy1],
                    env=env,
                    horizon=horizon,
                    render=render,
                    use_goals=use_goals,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                    verbose=verbose,
                    env_lang=env_lang # "agent0 task0_lang, agent 1 task1_lang, ..."
                )
            except Exception as e:
                print("Rollout exception at episode number {}!".format(ep_i))
                print(traceback.format_exc())
                break
            
            if batched:
                rollout_info["time"] = [(time.time() - rollout_timestamp) / len(env)] * len(env)

                for env_i in range(len(env)):
                    rollout_logs.append({k: rollout_info[k][env_i] for k in rollout_info})
                num_success += np.sum(rollout_info["Success_Rate"])
            else:
                rollout_info["time"] = time.time() - rollout_timestamp

                rollout_logs.append(rollout_info)
                num_success += rollout_info["Success_Rate"]
            
            if verbose:
                if batched:
                    raise NotImplementedError
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
        default=1000, # None by default
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
        default=False, # None by default
        help="(optional) debug for rollouts",
    )
    
    # manually assign environment language for 
    parser.add_argument(
        "--env_lang",
        type=str,
        default=None, # None by default, should be like "agent0 task0_lang, agent1 task1_lang, ..."
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    args.render == False
    run_trained_multitask_twoagent(args)
