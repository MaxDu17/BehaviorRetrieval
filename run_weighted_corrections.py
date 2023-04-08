"""
The main script for evaluating a policy in an environment.

Args:
    oracle_agent (str): path to saved checkpoint pth file of expert

    apprentice_agent (str): path to saved checkpoint pth file of lower-performing model

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
"""
import argparse
import json
import h5py
import psutil
import imageio
import numpy as np
import os
from copy import deepcopy
import csv
from collections import deque
import time
import random
import pickle
import pandas as pd

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.config import config_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger

import MachinePolicy

def switchboard(oracle, apprentice, intervention_threshold, takeover_threshold, intervention):
    """
    Helper function that decides when to intervene, using a simple upper bound and lower bound system
    :param oracle: oracle action vector
    :param apprentice: apprentice action vector
    :param intervention_threshold: trigger for intervention
    :param takeover_threshold: trigger for return of control
    :param intervention: current intervention state
    :return: new intervention state
    """
    # print(np.linalg.norm(oracle - apprentice))
    if np.linalg.norm(oracle - apprentice) > intervention_threshold:
        return True
    elif np.linalg.norm(oracle - apprentice) < takeover_threshold:
        return False
    else:
        return intervention #passthrough if the thresholds are not met


def rollout(oracle_policy, apprentice_policy, env, horizon, render=False, video_writer=None, video_skip=5,
            return_obs=False, camera_names=None, last_corrections = None, last_corrections_index = None,
            machine_script = False):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        oracle_policy (instance of RolloutPolicy): policy loaded from a checkpoint
        apprentice_policy (instance of RolloutPolicy): policy loaded from a checkpoint
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
        last_corrections (list): records last corrections to be used in reweighing process
        last_corrections_index (list): records the index of last corrections, for logging purposes
        machine_script (bool): if we are using a machine script for the oracle

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    apprentice_policy = RolloutPolicy(apprentice_policy)

    assert isinstance(env, EnvBase)
    assert isinstance(apprentice_policy, RolloutPolicy)
    if not machine_script:
        assert isinstance(oracle_policy, RolloutPolicy)
    else:
        assert isinstance(oracle_policy, MachinePolicy.MachinePolicy)
    assert not (render and (video_writer is not None))

    obs = env.reset() # SMALL BUT NECESSARY CHANGE FOR SCRIPTED POLICY

    apprentice_policy.start_episode()
    oracle_policy.start_episode()

    # obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    # obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], robot_actions=[], rewards=[], dones=[], states=[], corrections = [], initial_state_dict=state_dict)

    intervention_count = 0
    intervention = False
    if env._env_name == "PickPlaceCan":
        print("can!")
        INTERVENTION_THRESHOLD = 0.6
        TAKEOVER_THRESHOLD = 0.5

        ## ALTERNATIVE
        # INTERVENTION_THRESHOLD = 0.8
        # TAKEOVER_THRESHOLD = 0.6
    elif env._env_name == "NutAssemblySquare":
        # for square
        print("square!")
        # INTERVENTION_THRESHOLD = 0.03
        # TAKEOVER_THRESHOLD = 0.01

        #ALTERNATIVE
        INTERVENTION_THRESHOLD = 0.1
        TAKEOVER_THRESHOLD = 0.05
    elif env._env_name == "Widow250OfficeRand-v0":
        print("office!")
        INTERVENTION_THRESHOLD = 0.3
        TAKEOVER_THRESHOLD = 0.1
    else:
        raise Exception("invalid name")

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            apprentice_action = apprentice_policy(ob=deepcopy(obs))

            if not machine_script:
                oracle_action = oracle_policy(ob = deepcopy(obs))
                intervention = switchboard(oracle_action, apprentice_action, INTERVENTION_THRESHOLD, TAKEOVER_THRESHOLD,
                                           intervention)
            else:
                oracle_action = oracle_policy(ob = (obs, env.get_priv_info()))
                intervention = switchboard(oracle_action, apprentice_action, INTERVENTION_THRESHOLD, TAKEOVER_THRESHOLD,
                                           intervention) #let's see

            if args.baseline_type == "DAGGER":
                # always record the expert action, but play the apprentice action
                print("oracle", step_i)
                intervention = True
                act = oracle_action
                intervention_count += 1
                next_obs, r, done, _ = env.step(apprentice_action)
            else:
                if args.debug or intervention:
                    print("oracle", step_i)
                    intervention = True
                    act = oracle_action
                    intervention_count += 1
                else:
                    act = apprentice_action

                # play action
                next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        img = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                        if intervention:
                            img[0: 32, 0: 32, :] = 0
                            img[0 : 32, 0 : 32, 0] = 255
                        else:
                            img[0: 32, 0: 32, :] = 0
                            img[0: 32, 0: 32, 1] = 255
                        video_img.append(img)
                    video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            if intervention:
                traj["corrections"].append(int(step_i))
            traj["actions"].append(act)
            traj["robot_actions"].append(apprentice_action)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            # traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(deepcopy(obs)))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(deepcopy(next_obs))) # deepcopy to prevent corruption

                if intervention and last_corrections is not None and intervention_count % corrections_config["weighting"]["intervention_history_skip"] == 0:
                    # check that we are truncating correctly
                    processed_obs_dict = ObsUtils.unprocess_obs_dict(deepcopy(obs))
                    processed_obs_dict["actions"] = act
                    last_corrections.append(processed_obs_dict)
                    last_corrections_index.append(step_i)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success), Intervention_Prop = intervention_count / (step_i + 1), Intervention_Count = intervention_count)

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

def run_corrections(args, corrections_config, config):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"Are we using any baselines? {args.baseline_type}")
    print(corrections_config)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    #### INITIALIZING AGENTS ####
    PREEMPT_MODE = os.path.isdir(args.eval_path)
    apprentice_ckpt_path = args.apprentice_agent
    oracle_ckpt_path = args.oracle_agent
    embedder_ckpt_path = args.embedder

    if PREEMPT_MODE:
        raise Exception("not ready!")
        dirs = [k for k in os.listdir(args.eval_path) if ".pth" in k]
        if len(dirs) > 0:
            mtime_sorted = sorted(dirs, key=lambda t: os.stat(args.eval_path + t).st_mtime, reverse = True)
            apprentice_ckpt_path = args.eval_path + mtime_sorted[0]
            CHECKPOINT_STEP = int((mtime_sorted[0].split(".")[0]).split("_")[-1]) #expects XXXXXX_##.pth format to work
        else: #if there no checkpoints, we just start from the beginning
            PREEMPT_MODE = False

    apprentice_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=apprentice_ckpt_path, device=device, verbose=True,
                                                            trainable=True,
                                                            load_optimizer=corrections_config["load_optim_params"])


    embedder, _ = FileUtils.policy_from_checkpoint(ckpt_path=embedder_ckpt_path, device=device, verbose=True, trainable = False)
    embedder = embedder.policy #bypass the rollout class
    embedder.set_eval()

    # the config contains a lot of runtime parameters. We can load it, or we can get it from the original file
    if config is None:
        print("config loaded from checkpoint!")
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    with config.values_unlocked(): #modify some things for corrections
        config.train.hdf5_cache_mode = "low_dim"
        # config.train.num_data_workers = 0

    ObsUtils.initialize_obs_utils_with_config(config) #sets up things like normalizatoins, etc

    #### INITIALIZING ENVIRONMENT ####
    # read rollout settings
    rollout_horizon = corrections_config["horizon"]
    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon

    # getting metadata for later saving purposes
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    # create the rollout environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.eval_path is not None),
        use_image_obs=shape_meta["use_images"],
    )
    envs = {env.name: env}

     #used to be with the other policies
    if not args.machine_oracle:
        oracle_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=oracle_ckpt_path, device=device,
                                                                    verbose=True)
    else:
        oracle_policy = MachinePolicy.MACHINE_DICT[oracle_ckpt_path](noise=0, verbose=True, env=env)

    ### INITIALIZING DATASETS ##
    write_dataset = (args.intervention_save_path is not None)
    if write_dataset and not PREEMPT_MODE: #init from scratch if not preempting
        data_writer = h5py.File(args.intervention_save_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0
    elif write_dataset:
        data_writer = h5py.File(args.intervention_save_path, "a")
        data_grp = data_writer["data"]
        total_samples = data_grp.attrs["total"]

    correction_dataset = TrainUtils.dataset_factory(config, config.all_obs_keys, dataset_path=args.intervention_save_path, priority = True) #makes blank sampler, type SequenceDataset
    expert_dataset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"],
                                                          weighting = corrections_config["weighting"]["weighted"],
                                                          num_training_samples = args.train_size)

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        print("obs normalization!")
        obs_normalization_stats = expert_dataset.get_obs_normalization_stats()

    #BASELINE
    if args.baseline_type == "use_own_actions": #instead of the expert, use the robot actions
        robot_sampler = TrainUtils.dataset_factory(config, config.all_obs_keys, dataset_path=args.intervention_save_path, robot_only=True) #makes blank sampler, type SequenceDataset

    ### INITIALIZING INTERVENTION HISTORY ###
    last_corrections = deque([], maxlen = corrections_config["weighting"]["intervention_history"])
    last_corrections_index = deque([], maxlen = corrections_config["weighting"]["intervention_history"]) #for recording purposes
    if corrections_config["weighting"]["weighted"]:
        expert_dataset.compute_own_embeddings(embedder)
        if PREEMPT_MODE:
            try:
                with open(args.eval_path + "last_corrections.pkl") as f:
                    last_corrections = pickle.load(f)
                with open(args.eval_path + "last_corrections_index.pkl"):
                    last_corrections_index = pickle.load(f)
                expert_dataset.reweight_data(last_corrections, embedder, corrections_config["weighting"]["threshold"],
                                             corrections_config["weighting"]["epsilon"])
            except FileNotFoundError:
                print("can't load intervention history")
                PREEMPT_MODE = False

    #### LOGGING ####
    data_logger = DataLogger(
        log_dir = args.eval_path,
        log_tb=config.experiment.logging.log_tb,
    )

    rollout_stats = []
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    data_logger.record("Train/Weights", expert_dataset.get_weight_list(), 0, data_type = "distribution")
    data_logger.record("Train/Non_Zero_Weight_Prop", expert_dataset.get_active_weight_proportion(), 0, data_type = "scalar")

    #CSV process
    if PREEMPT_MODE:
        # essentially trim the logs down to size
        csv_file = pd.read_csv(args.eval_path + "corrections.csv", index_col = 1)
        csv_file = csv_file.iloc[0 : CHECKPOINT_STEP]
        csv_file.to_csv(args.eval_path + "corrections.csv")
        csv_file = open(args.eval_path + "corrections.csv", "a")
    elif not PREEMPT_MODE and os.path.exists(args.eval_path + "corrections.csv"):
        answer = input("do you want to overwrite the directory? (y/n)")
        if answer != "y":
            quit()
        csv_file = open(args.eval_path + "corrections.csv", "w")
    else:
        csv_file = open(args.eval_path + "corrections.csv", "w")

    csv_writer = csv.writer(csv_file)
    if not PREEMPT_MODE: csv_writer.writerow(["Env", "Correction", "Tries So Far", "Success Rate", "Intervention Prop"])
    total_tries = 0

    # maybe open hdf5 to write rollouts

    write_video = (args.eval_path is not None)
    assert not (args.render and write_video)  # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1


    # record the configurations of this run
    with open(os.path.join(args.eval_path, 'corrections_config.json'), 'w') as outfile:
        json.dump(corrections_config, outfile, indent=4)
    # save the config as a json file
    with open(os.path.join(args.eval_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)


    process = psutil.Process(os.getpid())
    mem_usage = int(process.memory_info().rss / 1000000000)
    print("\nInitial Memory Usage: {} GB\n".format(mem_usage))

    # some final initialization parameters
    num_eval_episodes = corrections_config["eval_episodes"] if not args.debug else 1
    start_step = CHECKPOINT_STEP + corrections_config["burn_in_episodes"] if PREEMPT_MODE else 0
    end_step =  corrections_config["rollout_episodes"] + corrections_config["burn_in_episodes"] if not args.debug else start_step + 2
    # currently, it's intervention -> eval -> train, which means that we should start out normally
    for i in range(start_step, end_step):
        epoch = i + 1 - corrections_config["burn_in_episodes"]  # one-indexed

        # save this history to a file for pre-empting purposes
        with open(args.eval_path + "last_corrections.pkl", "wb") as f:
            pickle.dump(last_corrections, f)
        with open(args.eval_path + "last_corrections_index.pkl", "wb") as f:
            pickle.dump(last_corrections_index, f)

        ###### INTERVENTION EPISODE #######
        success = False
        intervention_prop = 0
        last_corrections_copy = list()
        last_corrections_index_copy = list()
        while not success or intervention_prop < 0.01: #either we failed, or the model did too well. In either case, try again
            last_corrections_copy.clear()
            last_corrections_index_copy.clear()
            video_writer = None
            total_tries += 1
            if write_video and epoch % 1 == 0:
                video_writer = imageio.get_writer(args.eval_path + f"/corrections_{epoch}_{total_tries}.mp4", fps=20)
            stats, traj = rollout(
                oracle_policy=oracle_policy,
                apprentice_policy = apprentice_policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                last_corrections = last_corrections_copy,
                last_corrections_index = last_corrections_index_copy,
                machine_script = args.machine_oracle
            )
            success = stats["Success_Rate"]
            intervention_prop = stats['Intervention_Prop']
            print(f"EPISODE {epoch}, intervention prop: {stats['Intervention_Prop']}, success: {success}, tries: {total_tries}")
            if write_video and epoch % 5 == 0:
                video_writer.close()

            if args.baseline_type == "DAGGER":
                break # we don't need to care about failure when we are doing DAGGER

        rollout_stats.append(stats)
        # only add the interventions in a successful episode
        for correction, index in zip(last_corrections_copy, last_corrections_index_copy):
            last_corrections.append(correction)
            last_corrections_index.append(index)


        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("robot_actions", data=np.array(traj["robot_actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            ep_data_grp.create_dataset("corrections", data=np.array(traj["corrections"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]  # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]  # number of transitions in this episode
            ep_data_grp.attrs["num_interventions"] = stats["Intervention_Count"] # number of interventions in this episode
            total_samples += traj["actions"].shape[0]
            data_writer.flush()

            correction_dataset.update_dataset_in_memory(["demo_{}".format(i)], data_writer)
            if args.baseline_type == "use_own_actions":  # instead of the expert, use the robot actions
                robot_sampler.update_dataset_in_memory(["demo_{}".format(i)], data_writer)

        #### MODEL EVAL #####
        if args.debug or (epoch > 0 and epoch % corrections_config["eval_frequency"] == 0):
            rollout_model = RolloutPolicy(apprentice_policy, obs_normalization_stats=obs_normalization_stats)
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_eval_episodes,
                render=False,
                video_dir=args.eval_path if epoch % 5 == 0 else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                csv_writer.writerow(
                    [env_name, epoch, total_tries, rollout_logs["Success_Rate"], stats["Intervention_Prop"]])
                csv_file.flush()
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)
                data_logger.record("Rollout/Intervention_Prop", stats["Intervention_Prop"], epoch, log_stats=True)
                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            epoch_ckpt_name = "model_corrections_{}".format(epoch)
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            # Save model checkpoints based on conditions (success rate, validation loss, etc)
            if epoch % 10 == 0:
                TrainUtils.save_model(
                    model=apprentice_policy,
                    config=config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=os.path.join(args.eval_path, epoch_ckpt_name + ".pth")
                )

        if (epoch > 0 and epoch % corrections_config["corrections_per_train"] == 0)\
                or args.debug: # after burn-in
            print(epoch)
            train_num_steps = corrections_config["corrections"]["train_steps"] if not args.debug else 10
            ########## MODEL TRAIN #########
            apprentice_policy.set_train()  # just in case
            # make the loader for corrections
            if corrections_config["corrections"]["batch_type"] == "balanced":
                intervention_batch_size = config.train.batch_size
            elif corrections_config["corrections"]["batch_type"] == "ramped":
                # intervention_batch_size = int((4 * len(correction_dataset) / len(expert_dataset)) * config.train.batch_size)
                intervention_batch_size = int((args.int_scale * len(correction_dataset) / len(expert_dataset)) * config.train.batch_size)
                intervention_batch_size = max(intervention_batch_size, 1) # make sure we take at least one sample
                intervention_batch_size = min(intervention_batch_size, config.train.batch_size) #capping
            else:
                raise Exception("corrections/batch_type configuration not recognized!")

            train_loader = DataLoader(
                dataset=correction_dataset,
                batch_size=intervention_batch_size,
                shuffle=True,
                num_workers=config.train.num_data_workers,
                drop_last=True
            )

            if corrections_config["weighting"]["weighted"] and (args.debug or epoch % corrections_config["weighting"]["reweight_frq"] == 0):
                print(len(last_corrections))
                if args.threshold is not None:
                    threshold = args.threshold
                else:
                    threshold = corrections_config["weighting"]["threshold"]
                print(threshold)
                expert_dataset.reweight_data(last_corrections, embedder, threshold , corrections_config["weighting"]["epsilon"])

            weighting_sampler = expert_dataset.get_dataset_sampler() # if we are not weighting, then we should get a uniform distribution
            expert_loader = DataLoader(
                dataset=expert_dataset,
                sampler=weighting_sampler,
                batch_size=config.train.batch_size,
                shuffle = (weighting_sampler is None),
                num_workers=0, #config.train.num_data_workers,
                drop_last=True,
            )

            ### VISUALIZING NEW WEIGHTS ###
            if corrections_config["weighting"]["visualize_weights"]:
                weight_video_writer = imageio.get_writer(args.eval_path + f"/weights_{epoch}.mp4", fps=20)
                expert_dataset.visualize_demo(3, weight_video_writer)
                weight_video_writer.close()

            ##### SOME BASELINES ######
            if args.baseline_type == "use_own_actions":  # instead of the expert, use the robot actions
                robot_loader = DataLoader(
                    dataset=robot_sampler,
                    batch_size=config.train.batch_size,
                    shuffle=True,
                    num_workers=config.train.num_data_workers,
                    drop_last=True
                )
                step_log = TrainUtils.run_epoch(model=apprentice_policy, data_loader=train_loader, second_data_loader = robot_loader,
                                                epoch=epoch, num_steps=train_num_steps)
            elif args.baseline_type == "expert_only":
                step_log = TrainUtils.run_epoch(model=apprentice_policy, data_loader=expert_loader,
                                                epoch=epoch, num_steps=train_num_steps)
            elif args.baseline_type == "corrections_only":
                step_log = TrainUtils.run_epoch(model=apprentice_policy, data_loader=train_loader,
                                                epoch=epoch, num_steps=train_num_steps)
            #################
            else: # normal operations
                step_log = TrainUtils.run_epoch(model=apprentice_policy, data_loader=train_loader,
                                                second_data_loader=expert_loader,
                                                epoch=epoch, num_steps=train_num_steps, stopping = corrections_config["corrections"]["train_limit"])
            apprentice_policy.on_epoch_end(epoch)
            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            data_logger.record("Train/Set_Balance", intervention_batch_size / (intervention_batch_size + config.train.batch_size), epoch)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Train/{}".format(k), v, epoch)

            # if we are not weighting, then this should return near-uniform distributions
            weight_list = expert_dataset.get_weight_list()
            data_logger.record("Train/Weights", weight_list, epoch,
                               data_type="distribution")
            data_logger.record("Train/Non_Zero_Weight_Prop", expert_dataset.get_active_weight_proportion(), epoch,
                               data_type="scalar")

            ## TEMPORARY SCAFFOLD ##
            try:
                positive_negative_keys = expert_dataset.label_list
                data_logger.record("Train/Positive_Weights", weight_list[np.where(positive_negative_keys == 1)], epoch,
                                   data_type="distribution")
                data_logger.record("Train/Negative_Weights", weight_list[np.where(positive_negative_keys == 0)], epoch,
                                   data_type="distribution")

                data_logger.record("Train/Pos_Non_Zero_Weight_Prop", np.mean(weight_list[np.where(positive_negative_keys == 1)] > 0), epoch,
                                   data_type="scalar")
                data_logger.record("Train/Neg_Non_Zero_Weight_Prop", np.mean(weight_list[np.where(positive_negative_keys == 0)] > 0), epoch,
                                   data_type="scalar")
            except:
                pass
            ## TEMPORARY SCAFFOLD ##

            data_logger.record("Train/Intervention_Distribution", np.array(list(last_corrections_index)), epoch,
                               data_type="distribution")
            if args.baseline_type != "corrections_only" and args.baseline_type != "use_own_actions":
                try:
                    sample_distribution, sample_identity = expert_dataset.get_sample_distribution()
                    data_logger.record("Train/Sample_Distribution", sample_distribution, epoch,
                                       data_type="distribution")

                    ## TEMPORARY SCAFFOLD ##
                    try:
                        data_logger.record("Train/Positive_Sample_Distribution", sample_distribution[np.where(sample_identity == 1)], epoch,
                                           data_type="distribution")
                        data_logger.record("Train/Negative_Sample_Distribution", sample_distribution[np.where(sample_identity == 0)], epoch,
                                           data_type="distribution")
                    except:
                        pass
                    ## TEMPORARY SCAFFOLD ##

                except:
                    print("Threading error! There was a race condition and the ring buffer was not populated. A solution is to use zero workers")
            # Finally, log memory usage in GB
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000000)
            data_logger.record("System/RAM Usage (GB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} GB\n".format(epoch, mem_usage))
        else:
            print("BURN-IN PERIOD: SKIP TRAINING")


    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)  # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.intervention_save_path))

    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--oracle_agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # Path to trained model
    parser.add_argument(
        "--apprentice_agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # Path to embedder model
    parser.add_argument(
        "--embedder",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
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
        default=1,
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
        "--expert_dataset",
        type=str,
        default = None,
        help="The expert dataset for the model. Required if you are loading from fresh config",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--intervention_save_path",
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

    parser.add_argument(
        "--machine_oracle",
        action='store_true',
        help="Use a scripted policy",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--baseline_type",
        type=str,
        default=None,
        help="(optional) change behavior for certain baselines",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--correction_config",
        type=str,
        required = True,
        help=" correctoin config file",
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
             If omitted, default settings from checkpoing is used. This is the preferred way to run experiments.",
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="(optional) number of training episodes to use. Must be at most the maximum size alloted in the dataset\
            If omitted, we use the largest number of episodes.",
    )
    parser.add_argument(
        "--int_scale",
        type=int,
        default=4,
        help="(optional) number of effective intervention samples per sample. This is used to determin batch size.",
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="(optional) selection threshold. If given, we use this one. Otherwise, pull from json..",
    )

    args = parser.parse_args()
    corrections_config = json.load(open(args.correction_config, 'r'))

    # assert args.baseline_type != "expert_only" or not corrections_config["weighting"]["weighted"], "attempting expert only with a weighted buffer!"

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
        config.train.data = args.expert_dataset
    else:
        config = None

    run_corrections(args, corrections_config, config)
