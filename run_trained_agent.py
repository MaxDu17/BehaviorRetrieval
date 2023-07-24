# MODIFIED from the robomimic script
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
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
import time


import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.config import config_factory

import MachinePolicy

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None,
            machine_policy = False, real_robot = False, goal = None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)

    if machine_policy:
        assert isinstance(policy, MachinePolicy.MachinePolicy)
    else:
        assert isinstance(policy, RolloutPolicy)

    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = {}

    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            # print("\t", step_i)
            # get action from policy
            if machine_policy:
                clean_act, act = policy(ob = (obs, 0), noisy_mode = True) # the 0 used to be a different and defunct feature
            else:
                if goal is not None:
                    act = policy(ob = obs, goal = goal)
                else:
                    act = policy(ob=obs)

            # play action
            next_obs, r, done, info = env.step(act) #execute noisy actions for machine policy

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        if real_robot:
                            video_img.append(np.transpose(obs[cam_name], (1, 2, 0))) # no rendering in real robot
                        else:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            if machine_policy:
                traj["actions"].append(clean_act) #store clean actions
            else:
                traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            try:
                traj["states"].append(state_dict["states"])
            except:
                pass

            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(deepcopy(obs)))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(deepcopy(next_obs)))

            # break if done or if success
            if done or success:
                env.reset()
                break

            # update for next iter
            obs = deepcopy(next_obs)

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    # real robot eval framework.
    if real_robot:
        while True:
            answer = input("success? (y/n/x)")
            if answer == "y" or answer == "n" or answer == "x":
                break
        stats = dict(Return = 0, Horizon = (step_i + 1), Success_Rate = float(answer == "y"), Reject = float(answer == "x"))

    else:
        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.eval_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    if not args.machine_oracle:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        # create environment from saved checkpoint
        try: #hacky solution
            del ckpt_dict["env_metadata"]["env_kwargs"]["use_image_obs"]
        except:
            pass
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict,
            env_name=args.env,
            render=args.render,
            render_offscreen=(args.eval_path is not None),
            verbose=True,
        )
    else:
        # special setup for machine policies
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)

        # can also declare metadata manually. Here, we use a reference dataset.
        config.train.data = args.reference_data
        policy = MachinePolicy.MACHINE_DICT[ckpt_path](noise = 0.1, verbose = False) #use to be 0.2
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data )
        env_meta["env_kwargs"]["camera_widths"] = 84
        env_meta["env_kwargs"]["camera_heights"] = 84
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=True,
            use_image_obs=True, #default to image observations
        )
        ObsUtils.initialize_obs_utils_with_config(config)

    # loading goals
    if args.goal_data is not None:
        assert not args.machine_oracle #doesn't work with machine oracle
        ext_cfg = json.loads(ckpt_dict["config"])
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
        expert_data = TrainUtils.dataset_factory(config, config.all_obs_keys, dataset_path=args.goal_data,
                                                 num_samples=10, weighting=False)
    else:
        expert_data = None

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    i = 0
    rollout_stats = []

    start = time.time()
    while i < rollout_num_episodes:
        nonsense = i < rollout_num_episodes / 2 and args.paired and args.machine_oracle
        if args.machine_oracle:
            policy.set_nonsense(nonsense)
        video_writer = None
        if write_video and i % args.eval_skip == 0:
            video_writer = imageio.get_writer(args.eval_path + f"/eval{i}.gif", fps=20)
        print(f"Step {i} / {rollout_num_episodes}. Time taken: {int((time.time() - start) / 60)} mins")
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            machine_policy = args.machine_oracle,
            real_robot = args.real_robot,
            goal = expert_data.get_goal() if expert_data is not None else None #provide a goal if needed
        )
        if write_video and i % args.eval_skip == 0:
            video_writer.close()
        rollout_stats.append(stats)
        if args.success_only and stats["Success_Rate"] < 1 and not nonsense:
            # don't save failures
            continue
        if "Reject" in stats and stats["Reject"]:
            print("rejected!")
            continue # for real robot

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            if args.paired:
                ep_data_grp.attrs["nonsense"] = nonsense
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
        i += 1

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    try:
        # save to csv so we can extract numbers
        log_file = args.agent[ : args.agent.find("model_epoch") - 1]
        with open(f"{log_file}/success_rate_{args.checkpoint}.txt", "w") as f:
            f.write(str(avg_rollout_stats["Success_Rate"]))
    except:
        print(str(avg_rollout_stats["Success_Rate"])) # just in case, we always get the number output 

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #use this to override the parameters on a trained model. Use with caution
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
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
        "--eval_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this eval path",
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
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    #checkpoint label (for logging purposes ONLY; does not influence which checkpoint we select)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="--",
        help="(optional) checkpoint label",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--reference_data",
        type=str,
        default=None,
        help="(required for machine policies): a dataset to extract metadata from",
    )

    # used for goal-conditioned models
    parser.add_argument(
        "--goal_data",
        type=str,
        default=None,
        help="specify expert data for goal conditioned models",
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
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--eval_skip",
        type=int,
        default=1,
        help="(optional) used to render evals only every N times",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--success_only",
        action='store_true',
        help="Keep successful runs only (useful for demo collection)",
    )

    parser.add_argument(
        "--machine_oracle",
        action='store_true',
        help="Use a scripted policy",
    )

    parser.add_argument(
        "--paired",
        action='store_true',
        help="half meaningful policy, half nonsense policy",
    )

    parser.add_argument(
        "--real_robot",
        action='store_true',
        help="when you're evaluating on a real robot",
    )

    args = parser.parse_args()

    assert ".pth" in args.agent or args.machine_oracle, "Either provide a valid checkpoint or use a machine policy"
    assert not args.machine_oracle or args.reference_data is not None, "Machine policies must be provided metadata from an exernal dataset"
    assert not args.machine_oracle or args.config is not None, "Machine policies need a config file!"
    run_trained_agent(args)

