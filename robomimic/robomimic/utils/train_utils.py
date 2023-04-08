"""
This file contains several utility functions used to define the main training loop. It
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.utils.classifier_dataset import ClassifierDataset, DistanceClassifierDataset, TemporalEmbeddingDataset
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy


def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = config.train.output_dir
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(f"{base_output_dir}/{time_str}"):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(f"{base_output_dir}/{time_str}"))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(f"{base_output_dir}/{time_str}")

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir


def load_data_for_training(config, obs_keys, modifications = None, weighting = False, num_training_samples = None):

    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)
        modifications (str): modifications to sample behavior. this is mostly for deciding between classifier and sequence classes
        weighting (bool): enable/disable a weighted dataset
        num_training_samples (int): maximum number of training samples to use. Leave as "none" if you want to do a standard train/valid split

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    filter_by_attribute = config.train.hdf5_filter_key
    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        train_filter_by_attribute = "train"
        valid_filter_by_attribute = "valid"
        if filter_by_attribute is not None:
            train_filter_by_attribute = "{}_{}".format(filter_by_attribute, train_filter_by_attribute)
            valid_filter_by_attribute = "{}_{}".format(filter_by_attribute, valid_filter_by_attribute)
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute, modifications = modifications, weighting = weighting, num_samples = num_training_samples)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute, modifications = modifications, weighting = weighting)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=filter_by_attribute, modifications = modifications, weighting = weighting, num_samples = num_training_samples)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None, priority = False, weighting = False, modifications = None, robot_only = False, num_samples = None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

        priority (bool): if provided, this allows for sampling only the parts of the dataset that belong to an intervention

        weighting (bool): enable/disable a weighted dataset

        num_samples (int): maximum number of samples to use. Leave as "none" to take all that is avaiable

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    if priority:
        # require that we read this when we update the dataset
        with config.values_unlocked():
            if "corrections" not in config.train.dataset_keys:
                config.train.dataset_keys.append("corrections")

    if robot_only:
        with config.values_unlocked():
            config.train.dataset_keys.remove("actions")
            config.train.dataset_keys.append("robot_actions")

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        load_next_obs=True, # make sure dataset returns s'
        frame_stack=1, # no frame stacking
        seq_length=config.train.seq_length,
        pad_frame_stack=True,
        pad_seq_length=True, # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
        priority = priority,
        weighting = weighting,
        num_samples = num_samples
    )
    if modifications == "pos_neg_pair":
        ds_kwargs["radius"] = config.train.radius
        ds_kwargs["use_actions"] = config.train.actions
        # ds_kwargs["same_traj"] = config.train.same_traj
        dataset = ClassifierDataset(**ds_kwargs)
    elif modifications == "distance":
        ds_kwargs["use_actions"] = config.train.actions
        dataset = DistanceClassifierDataset(**ds_kwargs)
    elif modifications == "temporal_embedding":
        ds_kwargs["geometric_p"] = config.train.p
        dataset = TemporalEmbeddingDataset(**ds_kwargs)
    else:
        dataset = SequenceDataset(**ds_kwargs)

    return dataset


def run_rollout(
        policy,
        env,
        horizon,
        use_goals=False,
        goal = None,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase)

    policy.start_episode()

    ob_dict = env.reset()

    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs_dict = env.reset_to(state_dict)

    goal_dict = None
    if use_goals and goal is None: #if we don't specify a goal, ask the environent
        # retrieve goal from the environment
        goal_dict = env.get_goal()
    elif use_goals and goal is not None:
        goal_dict = goal


        # may consider instaed providing an image directly, or an observation directly

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = { k: False for k in env.is_success() } # success metrics
    import time
    try:
        for step_i in range(horizon):

            # get action from policy

            beg = time.time()
            ac = policy(ob=ob_dict, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            #FOR DEBUGGING
            # from matplotlib import pyplot as plt
            # anchor = ob_dict["robot0_eye_in_hand_image"]
            # # anchor = ob_dict["agentview_image"]
            # fig, ax = plt.subplots()
            # ax.imshow(np.transpose(anchor, (1, 2, 0)))
            # plt.savefig("debugging/test_frame.png")
            # print(np.mean(anchor)) #to see if things are normalized 0-1 or to 0-255
            # import pdb
            # pdb.set_trace()

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    agentview = env.render(mode="rgb_array", height=256, width=256)
                    # try adding an eye in hand image to help with diagnostics
                    try:
                        eye_in_hand = env.render(mode="rgb_array", height=256, width=256, camera_name="robot0_eye_in_hand_image")
                        video_img = np.concatenate((agentview, eye_in_hand), axis=0)
                    except:
                        video_img = agentview
                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        goal = None,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        record_first = False,
        terminate_on_success=False,
        verbose=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                goal = goal,
                video_writer=env_video_writer if not record_first or ep_i == 0 else None,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        optimizer = model.optim_serialize(), #save optimizer parameters
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))

def weld_batches(first_batch, second_batch):
    # assert first_batch.keys() == second_batch.keys(), "the batches have different keys!"
    combined = {}
    flag = False
    # a little hacky, but it's for a specific baseline only
    if "robot_actions" in second_batch:
        flag = True
    for key in first_batch.keys():
        # the problem is a mismatched key, which we solve by just rewiring them. It's not elegant but it works
        if flag:
            key2 = key if key != "actions" else "robot_actions"
        else:
            key2 = key

        val1 = first_batch[key]
        val2 = second_batch[key2]
        if type(val1) is torch.Tensor:
            combined[key] = torch.cat((val1, val2), dim = 0)
        else:# for obs
            combined[key] = {}
            for inner_key in val1.keys():
                inner_val1 = val1[inner_key]
                inner_val2 = val2[inner_key]
                combined[key][inner_key] = torch.cat((inner_val1, inner_val2), dim = 0)
    return combined

def run_epoch(model, data_loader,epoch, validate=False, num_steps=None, second_data_loader = None,
              stopping = "step",
              stopping_norm = 5000000,
              return_predictions = False,
              return_matrix = False,
              return_reconstruction = False):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        second_data_loader (DataLoader instance): second replay buffer to sample from (see data_loader)
        stopping (str): either "step", "grad", or "valid". Step means step limited epoch, "grad" means early termination
        with grad norms, and "valid" means early termination using a validation set (not implemented yet)

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        # print("wrong settings in train utils")
        # model.set_train()
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    if second_data_loader is not None:
        second_data_loader_iter = iter(second_data_loader)

    norm_list = list()
    for _ in LogUtils.custom_tqdm(range(num_steps)):
        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
            if second_data_loader is not None:
                second_batch = next(second_data_loader_iter)
                batch = weld_batches(batch, second_batch)

        except StopIteration: # pass through again if needed
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            if second_data_loader is not None:
                second_data_loader_iter = iter(second_data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # from matplotlib import pyplot as plt
        # anchor = batch["obs"]["agentview_image"][0, 3].cpu().detach().numpy()
        # anchor = batch["obs"]["robot0_eye_in_hand_image"][0, 3].cpu().detach().numpy()
        # fig, ax = plt.subplots()
        # ax.imshow(np.transpose(anchor, (1, 2, 0)))
        # plt.savefig("debugging/train.png")
        # print(np.mean(anchor))  # to see if things are normalized 0-1 or to 0-255
        # import pdb
        # pdb.set_trace()

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)

        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        if not validate:
            norm_list.append(info["policy_grad_norms"])
            if stopping == "norm" and np.mean(norm_list) < stopping_norm: # empirical value
                print("early termination (norm)")
                break
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])

    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    if return_predictions:
        # if this returns an error, then it might be because you're not using the vanilla classifier
        return step_log_all, info["predictions"]

    if return_matrix:
        return step_log_all, info["product_matrix"] #take the last part of the vlaidation

    if return_reconstruction:
        if "agentview_image" in batch["obs"]:
            # select a random image
            selected_image =  batch["obs"]["agentview_image"][10][0].detach().cpu().numpy()
            reconstructed_selected_image = info["reconstruction"]["agentview_image"][10].detach().cpu().numpy()
            concat_image = np.concatenate((np.transpose(selected_image, (1, 2, 0)),
                                               np.transpose(reconstructed_selected_image, (1, 2, 0))), axis = 0)
            if "robot0_eye_in_hand_image" in batch["obs"]:
                selected_image = batch["obs"]["robot0_eye_in_hand_image"][10][0].detach().cpu().numpy()
                reconstructed_selected_image = info["reconstruction"]["robot0_eye_in_hand_image"][10].detach().cpu().numpy()
                concat_eye_in_hand = np.concatenate(
                    (np.transpose(selected_image, (1, 2, 0)), np.transpose(reconstructed_selected_image, (1, 2, 0))), axis=0)
                concat_image = np.concatenate((concat_image, concat_eye_in_hand), axis = 1)

            return step_log_all, concat_image
        else:
            return step_log_all, np.zeros((20, 20)) #dummy return for lowdim

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval.
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)

    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
