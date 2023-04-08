"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    eval_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --eval_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import importlib

import torch
from torch.utils.data import DataLoader
import tqdm

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

import MachinePolicy


def rollout(policy, env, horizon, render=False, video_writer=None, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert not (render and (video_writer is not None))

    obs = env.reset()
    policy.start_episode()

    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)
    video_count = 0  # video frame counter

    try:
        for step_i in range(horizon):
            print(step_i)
            # get action from policy
            # act = policy(ob=(obs, env.get_priv_info()))
            act = policy(ob=(obs))
            # play action

            next_obs, r, done, _ = env.step(act)

            # compute reward
            success = env.is_success()["task"]

            video_img = []
            if step_i % 1 == 0:
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                video_writer.append_data(video_img)
                video_count += 1

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt")
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
    return success



def run_trained_agent(args):
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # THIS IS FOR ROBOMIMIC
    # env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.data_path)

    ROBOVERSE_TYPE = 4

    # THIS IS FOR ROBOVERSE
    # env_meta = {
    #     "env_name": "Widow250OfficeRand-v0",
    #     "type": ROBOVERSE_TYPE,
    #     "env_kwargs": {
    #         "accept_trajectory_key": "table_clean", #needed for the wrapper
    #         "observation_img_dim": 84, #256, #
    #         'observation_mode': 'pixels_eye_hand',
    #         "control_mode" : "discrete_gripper",
    #         "num_objects" : 2,
    #         "object_names" : ['eraser', 'shed', 'pepsi_bottle', 'gatorade'],
    #         "random_shuffle_object" : True,
    #         "random_shuffle_target" : True,
    #         # "object_targets" : ["drawer_inside", 'tray'],
    #         "object_targets" : ["container", 'tray'],
    #         # "object_targets" : ['tray'],
    #         # "original_object_positions" : [
    #         # [0.43620103, 0.12358467, -0.35],
    #         # [0.55123888, -0.17699107, -0.35],
    #         # [0.42755662, -0.13711447, -0.35],
    #         # [0.39866522, 0.18929185, -0.35]],
    #         # "desired_config" : {"eraser" : "drawer_inside"}
    #     }
    # }

    env_meta = {
        "env_name": "Widow250OfficeRand-v0",
        "type": ROBOVERSE_TYPE,
        "env_kwargs": {
            "accept_trajectory_key": "table_clean",  # needed for the wrapper
            "observation_img_dim": 84,  # 256, #
            'observation_mode': 'pixels_eye_hand',
            "control_mode": "discrete_gripper",
            "num_objects": 1,
            "object_names": ['eraser', 'shed', 'pepsi_bottle', 'gatorade'],
            "random_shuffle_object": False,
            "random_shuffle_target": False,
            # "object_targets" : ["drawer_inside", 'tray'],
            "object_targets": ["tray", 'container'],
            # "desired_config" : {"eraser" : "drawer_inside"}
        }
    }

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=False, # only operate on lowdim
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    rollout_counter = 0

    # THIS IS FOR ROBOVERSE
    specs = { "obs": {
        "low_dim": ["robot"
                    ],
        "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
        "depth": [],
        "scan": []
    }}

    # THIS IS FOR ROBOMIMIC STUFF
    # specs = {"obs": {
    #             "low_dim": [
    #                 "robot0_eef_pos",
    #                 "robot0_eef_quat",
    #                 "robot0_gripper_qpos",
    #                 "object"
    #             ],
    #             "rgb": [],
    #             "depth": [],
    #             "scan": []
    #         }}

    ObsUtils.initialize_obs_utils_with_obs_specs(specs)
    success_counter = 0
    # for i in range(10):
    # while True:
        # try:
        #     input("press enter to begin next trial!")
        #     # importlib.reload(MachinePolicy) # allows you to run this live
            # policy = MachinePolicy.SquareAssemblyPolicy(noise = 0.2, paired = args.paired)
    # policy = MachinePolicy.OfficePolicy(env, policy_name = "tableclean")
    policy = MachinePolicy.OfficePolicy(env)
    # policy = MachinePolicy.ToolHangMachinePolicy(noise=0.5)
    video_writer = imageio.get_writer(args.eval_path + str(rollout_counter) + "eval.gif", fps=20)
    success = rollout(
        policy=policy,
        env=env,
        horizon=400,
        render=args.render,
        video_writer=video_writer,
        camera_names=args.camera_names,
    )
    success_counter += success
    video_writer.close()
            # rollout_counter += 1
        # except Exception as e:
        #     print("exception!" + str(e))
    print(success_counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="randomization seed",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--data_path",
        type=str,
        required = False,
        help="(optional) where the dataset is, for environment loading purposes. IF you don't specify this, you must give args manually",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--eval_path",
        type=str,
        required = True,
        help="where the dataset is, for environment loading purposes",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--paired",
        action='store_true',
        help="intentionally mess up 50% of the demos",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    args = parser.parse_args()
    run_trained_agent(args)


