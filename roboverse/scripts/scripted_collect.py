import numpy as np
import time
import os
import os.path as osp
import roboverse
from roboverse.policies import policies
import argparse
from tqdm import tqdm
import h5py
import imageio
import time
import json

from roboverse.utils import get_timestamp
EPSILON = 0.1

def add_transition(traj, observation, action, reward, info, agent_info, done,
                   next_observation, img_dim, image_rendered=True, video_writer = None):
    if image_rendered:
        observation["image"] = np.reshape(np.uint8(observation["image"]),
                                        (img_dim, img_dim, 3))
        next_observation["image"] = np.reshape(
            np.uint8(next_observation["image"]), (img_dim, img_dim, 3))

        observation["image_eye_in_hand"] = np.reshape(np.uint8(observation["image_eye_in_hand"]),
                                        (img_dim, img_dim, 3))
        next_observation["image_eye_in_hand"] = np.reshape(
            np.uint8(next_observation["image_eye_in_hand"]), (img_dim, img_dim, 3))

        if video_writer is not None:
            frame = np.concatenate((observation["image"], observation["image_eye_in_hand"]), axis=0)
            video_writer.append_data(frame)

    traj["observations"].append(observation)
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)


    return traj


def collect_one_traj(env, policy, num_timesteps, noise,
                     accept_trajectory_key, image_rendered, args, reshuffle, video_writer = None):
    num_steps = -1
    rewards = []
    success = False
    img_dim = env.observation_img_dim
    env.reset() # reshuffle argument determines if the environment will be reshuffled. Here, we always reshuffle
    # env.reset(reshuffle)

    # FOR RANDOM PICK PLACE
    current_object = env.task_object_names[0]
    current_target = env.object_targets[0]

    # policy.reset(object_target = current_target, object_name = current_object)

    # FOR ONLY TARGET TASK PICK AND PLACE
    # print("erasers only!")
    # policy.reset(object_target = env.target_object_target, object_name = env.target_object)

    # FOR VANILLA OFFICE CLEANING
    policy.reset()
    time.sleep(0.1)
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    is_opened = False
    is_closed = False
    total_reward = 0
    total_reward_thresh = sum([subtask.REWARD for subtask in env.subtasks])
    observation = env.get_observation()
    for j in range(num_timesteps):
        # print(j)
        # beg = time.time()
        action, agent_info, add_noise = policy.get_action()
        # In case we need to pad actions by 1 for easier realNVP modelling
        env_action_dim = env.action_space.shape[0]
        if env_action_dim - action.shape[0] == 1:
            action = np.append(action, 0)

        if add_noise:
            action += np.random.normal(scale=noise, size=(env_action_dim,))
        else:
            action += np.random.normal(scale=noise*0.3, size=(env_action_dim,))

        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)


        next_observation, reward, done, info = env.step(action)

        add_transition(traj, observation, action, reward, info, agent_info,
                       done, next_observation, img_dim, image_rendered, video_writer)
        total_reward += reward

        observation = next_observation.copy()
        # print(time.time() - beg)

        if accept_trajectory_key == 'table_clean':
            # print(total_reward)
            if total_reward == total_reward_thresh and num_steps < 0:
                num_steps = j
            if total_reward == total_reward_thresh :
                success = True
                # print(f"time {j}")
        else:
            if info[accept_trajectory_key] and num_steps < 0:
                num_steps = j
            if info[accept_trajectory_key]:
                success = True

        # print(f"step: {t4 - t3}, obs: {t3 - t2}, policy: {t2 - t1}")

        rewards.append(reward)
        if done or agent_info['done']:
            break

    return traj, success, num_steps


def dump2h5(traj, path, image_rendered, occurence):
    """Dumps a collected trajectory to HDF5 file."""
    # convert to numpy arrays

    states = np.array([o['state'] for o in traj['observations']])
    proprio = np.array([o['robot'] for o in traj['observations']])
    if image_rendered:
        images = np.array([o['image'] for o in traj['observations']])
        images_eye_in_hand = np.array([o['image_eye_in_hand'] for o in traj['observations']])
    actions = np.array(traj['actions'])
    rewards = np.array(traj['rewards'])
    terminals = np.array(traj['terminals'])

    # create HDF5 file
    f = h5py.File(path, "w")
    f.create_dataset("traj_per_file", data=1)

    # store trajectory info in traj0 group
    traj_data = f.create_group("traj0")

    # this is the oracle state!
    traj_data.create_dataset("relevant_demo", data=occurence, dtype = np.uint8)

    traj_data.create_dataset("states", data=states)
    traj_data.create_dataset("proprio", data=proprio)
    if image_rendered:
        traj_data.create_dataset("image", data=images, dtype=np.uint8)
        traj_data.create_dataset("image_eye_in_hand", data=images_eye_in_hand, dtype=np.uint8)
    traj_data.create_dataset("actions", data=actions)
    traj_data.create_dataset("rewards", data=rewards)

    if np.sum(terminals) == 0:
        terminals[-1] = True

    # build pad-mask that indicates how long sequence is
    is_terminal_idxs = np.nonzero(terminals)[0]
    pad_mask = np.zeros((len(terminals),))
    pad_mask[:is_terminal_idxs[0]] = 1.
    traj_data.create_dataset("pad_mask", data=pad_mask)

    f.close()


def main(args):

    timestamp = get_timestamp()
    data_save_path = args.save_directory

    data_save_path = osp.abspath(data_save_path)
    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)

    with open(args.config) as json_file:
        kwargs = json.load(json_file)

    # kwargs = {
    #     "observation_img_dim": 84, #256, #
    #     'observation_mode': 'pixels_eye_hand',
    #     "control_mode" : "discrete_gripper",
    #     "num_objects" : 1,
    #     "object_names" : ['eraser', 'shed', 'pepsi_bottle', 'gatorade'],
    #     "random_shuffle_object" : True,
    #     "random_shuffle_target" : True,
    #     "object_targets" : ["drawer_inside", 'tray'],
    #     "original_object_positions" : [
    #     [0.43620103, 0.12358467, -0.35],
    #     [0.55123888, -0.17699107, -0.35],
    #     [0.42755662, -0.13711447, -0.35],
    #     [0.39866522, 0.18929185, -0.35]],
    #     "desired_config" : {"eraser" : "drawer_inside"}
    # }

    env = roboverse.make(args.env_name,
                         gui=args.gui,
                         transpose_image=False, **kwargs)

    data = []
    assert args.policy_name in policies.keys(), f"The policy name must be one of: {policies.keys()}"
    # assert args.accept_trajectory_key in env.get_info().keys(), \
    #     f"""The accept trajectory key must be one of: {env.get_info().keys()}"""
    policy_class = policies[args.policy_name]
    # policy = policy_class(env)
    policy = policy_class(env)

    num_success = 0
    num_saved = 0
    num_attempts = 0
    accept_trajectory_key = args.accept_trajectory_key

    progress_bar = tqdm(total=args.num_trajectories)

    total_area_occurance = [0, 0, 0]
    total_object_occurance = {}
    total_task_occurance = 0
    for object_name in env.object_names:
        total_object_occurance[object_name] = 0

    last_success = True
    while num_saved < args.num_trajectories:
        num_attempts += 1
        video_writer = None
        if num_saved % 20 == 0:
            video_writer = imageio.get_writer(args.save_directory + f"/demo{num_saved}_{num_attempts}.gif", fps=20)
        traj, success, num_steps = collect_one_traj(
            env, policy, args.num_timesteps, args.noise,
            accept_trajectory_key, args.image_rendered, args, last_success, video_writer)
        if video_writer is not None:
            video_writer.close()
        # print("num_timesteps: ", num_steps)
        if success:
            if args.gui:
                print("num_timesteps: ", num_steps)

            data.append(traj)
            area_occurance, object_occurance, task_occurance = env.get_occurance()
            dump2h5(traj, os.path.join(data_save_path, 'rollout_{}.h5'.format(num_saved)),
                        args.image_rendered, task_occurance)
            num_success += 1
            num_saved += 1

            for i in range(len(area_occurance)):
                total_area_occurance[i] += area_occurance[i]
            for object_name in env.object_names:
                total_object_occurance[object_name] += object_occurance[object_name]
            total_task_occurance += task_occurance
            # print(total_area_occurance, total_object_occurance)
            fo = open(os.path.join(data_save_path, 'occurance.txt'), "w")
            str_area = f"area_occurance: {total_area_occurance}\n"
            str_object = f"object_occurance: {total_object_occurance}\n"
            str_task = f"task_occurance: {total_task_occurance}\n"
            fo.write(str_area)
            fo.write(str_object)
            fo.write(str_task)
            fo.close()
            progress_bar.update(1)
        elif args.save_all:
            data.append(traj)
            num_saved += 1
            progress_bar.update(1)
        else:
            print("failed! Trying again.")
        last_success = success
        if args.gui:
            print("success rate: {}".format(num_success/(num_attempts)))

    progress_bar.close()
    print("success rate: {}".format(num_success / (num_attempts)))
    print(total_area_occurance, total_object_occurance, total_task_occurance)
    # path = osp.join(data_save_path, "scripted_{}_{}.npy".format(
    #     args.env_name, timestamp))
    # print(path)
    # np.save(path, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-nt", "--num-task", type=int, default=3)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("-o", "--target-object", type=str)
    parser.add_argument("-config", "--config", type=str)
    parser.add_argument("-d", "--save-directory", type=str, default=""),
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("-r", "--image-rendered", type=int, default=1)
    parser.add_argument("-f", "--full-reward", type=int, default=0)

    args = parser.parse_args()

    main(args)
