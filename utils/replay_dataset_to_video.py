import torch
import h5py
import tqdm
import json
import numpy as np
import imageio
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import os
import cv2
import random
import robomimic.envs.env_base as EB

# dataset = h5py.File(f"../datasets/square_machine_policy/square_400_paired.hdf5", 'r')
# dataset = h5py.File(f"../datasets/can/paired/image.hdf5", 'r')
# dataset = h5py.File(f"../datasets/office/office_demos_individual_1200/office_image_new.hdf5", 'r')
# dataset = h5py.File(f"../datasets/widowx_peg_paired/paired_160_IMAGE.hdf5", 'r')
# dataset = h5py.File(f"../datasets/bridge/bridge_subset.hdf5", 'r')
dataset = h5py.File(f"../datasets/bridge/bridge_sink.hdf5", 'r')
# dataset = h5py.File(f"../datasets/bridge_own/pickle_cup.hdf5", 'r')
# dataset = h5py.File(f"../datasets/bridge_sink/kitchen_new.hdf5", 'r')

data_grp = dataset["data"]
total_samples = data_grp.attrs["total"]
num_demos = len(data_grp.keys())
save_dir = "demos/bridge_sink"

skip = 1
for i in tqdm.tqdm(range(20)):
    ep_data = data_grp[f"demo_{random.randint(0, num_demos - 1)}"]
    obs = ep_data["obs"]
    additional = ""
    if ep_data.attrs["target"] == 1:
        additional = "target"
        print("target!")
    camera = obs["agentview_image"]
    for frame in range(camera.shape[0]):
        if frame % skip == 0:
            imageio.imsave(f"{save_dir}/{i}_{frame}_{additional}.png", camera[frame])

