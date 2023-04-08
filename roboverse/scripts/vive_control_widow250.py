import numpy as np
import pybullet as p
import gym
import numpy as np
import roboverse.bullet as bullet
import os
from tqdm import tqdm
import argparse
import time
import roboverse
import datetime

# =========================================================
# Index corresponds to POSITION, ORIENTATION, BUTTTON etc
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6
ORIENTATION_ENABLED = True
EPSILON = 0.005
# =========================================================


def collect_one_trajectory(env, num_timesteps):

    prev_vr_theta = 0

    def get_gripper_input(e):
        # Detect change in button, and change trigger state
        if e[BUTTONS][33] & p.VR_BUTTON_IS_DOWN:
            trigger = -0.8
        elif e[BUTTONS][33] & p.VR_BUTTON_WAS_RELEASED:
            trigger = 0.8
        else:
            trigger = 0
        return trigger

    def accept_traj(info):
        return info["grasp_success"]  # TODO: just grasping for now; will add info["push_success"] etc

    # get VR controller output at one timestamp
    def get_vr_output():
        nonlocal prev_vr_theta
        ee_pos, ee_theta = bullet.get_link_state(
            env.robot_id, env.end_effector_index)
        events = p.getVREvents()

        # detect input from controllers
        assert events, "no input from controller!"
        e = events[0]

        # obtain gripper state from controller trigger
        trigger = get_gripper_input(e)

        # pass controller position and orientation into the environment
        cont_pos = e[POSITION]
        cont_orient = bullet.deg_to_quat([180, 0, 0])
        if ORIENTATION_ENABLED:
            cont_orient = e[ORIENTATION]
            cont_orient = bullet.quat_to_deg(list(cont_orient))

        action = [cont_pos[0] - ee_pos[0],
                  cont_pos[1] - ee_pos[1],
                  cont_pos[2] - ee_pos[2]]
        action = np.array(action) * 3.5  # to make grasp success < 20 timesteps

        grip = trigger
        for _ in range(2):
            action = np.append(action, 0)
        wrist_theta = cont_orient[2] - prev_vr_theta

        action = np.append(action, wrist_theta)
        action = np.append(action, grip)
        action = np.append(action, 0)
        # ===========================================================
        # Add noise during actual data collection
        noise = 0.1
        noise_scalings = [noise] * 3 + [0.1 * noise] * 3 + [noise] * 2
        action += np.random.normal(scale=noise_scalings)
        # ===========================================================

        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
        prev_vr_theta = cont_orient[2]
        return action

    o = env.reset()
    time.sleep(1.5)
    images = []
    accept = False
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
        original_object_positions=env.original_object_positions,
    )
    first_time = True
    # Collect a fixed length of trajectory
    for i in range(num_timesteps):
        action = get_vr_output()
        observation = env.get_observation()
        traj["observations"].append(observation)
        next_state, reward, done, info = env.step(action)
        traj["next_observations"].append(next_state)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)
        time.sleep(0.03)
        if accept_traj(info) and first_time:
            print("num_timesteps: ", i)
            first_time = False

    # ===========================================================
    if accept_traj(info):
        accept = "y"
    # ===========================================================

    return accept, images, traj


def timestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    args = parser.parse_args()

    timestamp = timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data', timestamp)
    data_save_path = os.path.abspath(data_save_path)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    data = []
    env = roboverse.make(args.env_name,
                         gui=True,
                         control_mode='discrete_gripper')
    env.reset()

    for j in tqdm(range(args.num_trajectories)):
        success, images, traj = collect_one_trajectory(env, args.num_timesteps)
        while success != 'y' and success != 'Y':
            print("failed for trajectory {}, collect again".format(j))
            success, images, traj = collect_one_trajectory(env, args.num_timesteps)
        data.append(traj)

        if j % 50 == 0:
            path = os.path.join(data_save_path, "{}_{}_{}_{}.npy".format(args.env_name, args.task_name, timestamp, j))
            np.save(path, data)

    path = os.path.join(data_save_path, "{}_{}_{}.npy".format(args.env_name, args.task_name, timestamp))
    np.save(path, data)
