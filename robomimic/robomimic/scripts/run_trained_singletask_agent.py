import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils # added
import robomimic.utils.env_utils as EnvUtils # added
import robomimic.utils.train_utils as TrainUtils # added
from robomimic.utils.log_utils import PrintLogger, DataLogger # added

from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def run_trained_singletask_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    
    # if rollout_horizon is None:
        # read horizon from config
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is None), 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # can choose to use logger
    # data_logger = DataLogger(
    #     log_dir,
    #     config,
    #     log_tb=config.experiment.logging.log_tb,
    #     log_wandb=config.experiment.logging.log_wandb,
    # )
    # data_logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    data_logger = None
    
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=policy,
        envs=[env],
        horizon=args.horizon,
        use_goals=config.use_goals,
        num_episodes=args.n_rollouts,
        render=args.render,
        video_dir=args.video_path if write_video else None,
        epoch=0, # epoch number can be assigned manually
        video_skip=args.video_skip,
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        del_envs_after_rollouts=True,
        data_logger=data_logger,
    )
    
    print('all rollout logs:')
    print(all_rollout_logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default='/home/ypz/project/model_opensingledoor_epoch_1000.pth',
        # required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=2,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=400, # None by default
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
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
        default="/home/ypz/msclab/robocasa_space/test",
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
        default=0, # None by default
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    args.render == False
    run_trained_singletask_agent(args)
