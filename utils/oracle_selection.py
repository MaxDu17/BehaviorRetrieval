import os
import json
import h5py
import argparse
import imageio
import numpy as np
import tqdm
import robomimic.envs.env_base as EB


# f = h5py.File("datasets/square_machine_policy/square_50_good.hdf5", "r")
f = h5py.File("../datasets/widowx_peg_paired/paired_160_IMAGE.hdf5", "r")

demos = list(f["data"].keys()) #access metadata with .attrs
inds = np.argsort([int(elem[5:]) for elem in demos])

demos = [demos[i] for i in inds if f["data"][demos[i]].attrs["target"] == 1]
# demos = [demos[i] for i in inds if f["data"][demos[i]]["rewards"][-1] > 0.99] #these are the productive demos

OUT_FILE = "../datasets/widowx_peg_paired/paired_160_IMAGE_ORACLE.hdf5"
if os.path.exists(OUT_FILE):
    print("This file already exists!")
    quit()

data_writer = h5py.File(OUT_FILE, "w")
data_grp = data_writer.create_group("data")
total_samples = 0


data = f["data"]

env_args = json.dumps({
        "env_name" : "online",
        "type" : EB.EnvType.GYM_TYPE,
        "env_kwargs": {
            "robot_model" : 'wx200',
            "ip_address" : '127.0.0.1',
            "port" : 9136,
            "use_image_obs" : False,
            "control_hz" : 3,
            "use_local_cameras" : False,
            "use_robot_cameras" : True,
            "camera_types" : ["cv2"],
            "reset_pos" : None,
            "control_mode" : "POS",
            "xlims" : [0.12, 0.4],
            "ylims" : [-0.23, 0.23],
            "zlims" : [0, 0.3]
        }
    }
)

# data_grp.attrs["env_args"] = data.attrs["env_args"] # copy environment params
data_grp.attrs["env_args"] = env_args

for demo in tqdm.tqdm(demos):
    print(demo)
    ep_data_grp = data_grp.create_group(demo)
    ep_data_grp.create_dataset("actions", data=np.array(data[demo]["actions"]))
    # ep_data_grp.create_dataset("states", data=np.array(data[demo]["states"]))
    ep_data_grp.create_dataset("rewards", data=np.array(data[demo]["rewards"]))
    ep_data_grp.create_dataset("dones", data=np.array(data[demo]["dones"]))
    for k in data[demo]["obs"]:
        ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(data[demo]["obs"][k]))
        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(data[demo]["next_obs"][k]))

    # episode metadata
    # ep_data_grp.attrs["model_file"] = data[demo].attrs["model_file"]  # model xml for this episode
    ep_data_grp.attrs["num_samples"] = data[demo].attrs["num_samples"] # number of transitions in this episode
    total_samples += data[demo].attrs["num_samples"]
    ep_data_grp.attrs["target"] = 1

data_grp.attrs["total"] = total_samples
print(total_samples)
