import argparse
import time
import subprocess
import datetime
import os

from roboverse.utils import get_timestamp
import numpy as np
import os
import os.path as osp
import roboverse
from roboverse.policies import policies
from tqdm import tqdm
import h5py


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-d", "--data-save-directory", type=str, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("--target-object", type=str, default="shed")
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("-f", "--full-reward", type=int, default=1)
    parser.add_argument("-r", "--image-rendered", type=int, default=0)
    args = parser.parse_args()
    print(f"using {args.num_parallel_threads} threads")
    num_trajectories_per_thread = int(
        args.num_trajectories / args.num_parallel_threads)
    if args.num_trajectories % args.num_parallel_threads != 0:
        num_trajectories_per_thread += 1

    timestamp = get_timestamp()
    commands = []
    script_name = "scripted_collect.py"
    
    for i in range(args.num_parallel_threads):
        save_directory = os.path.join("/iris/u/maxjdu/Repos/", 'roboverse', args.data_save_directory, f"p{i}")
        print(f"saving to: {save_directory}")
        command = ['python',
                'scripts/{}'.format(script_name),
                '--policy-name={}'.format(args.policy_name),
                '-a{}'.format(args.accept_trajectory_key),
                '-e{}'.format(args.env),
                '-n {}'.format(num_trajectories_per_thread),
                '-t {}'.format(args.num_timesteps),
                '-o{}'.format(args.target_object),
                '-d{}'.format(save_directory),
                '-f{}'.format(args.full_reward),
                '--image-rendered={}'.format(args.image_rendered),
                ]
        commands.append(command)

    if args.save_all:
        command.append('--save-all')

    subprocesses = []
    for i in range(args.num_parallel_threads):
        subprocesses.append(subprocess.Popen(commands[i]))
        time.sleep(1)

    exit_codes = [p.wait() for p in subprocesses]

    merge_command = ['python',
                     'scripts/combine_trajectories.py',
                     '-d{}'.format(save_directory)]

    subprocess.call(merge_command)

