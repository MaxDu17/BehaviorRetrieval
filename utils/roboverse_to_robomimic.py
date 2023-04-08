import torch
import h5py
import tqdm
import json
import numpy as np
import imageio
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from roboverse.assets.shapenet_object_lists \
    import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS, PICK_PLACE_TRAIN_OBJECTS, \
    PICK_PLACE_TEST_OBJECTS, TRAIN_CONTAINERS, TEST_CONTAINERS, PICK_PLACE_DEMO_CONTAINERS, PICK_PLACE_DEMO_OBJECTS


### FOR ROBOVERSE STYLE DEMOS ##
ROBOVERSE_TYPE = 4
IMG_DIM = 84 #used to be 84 #used to be 256

ROOT_DIR = "office_welding_100"
# ROOT_DIR = "office_demos_individual_1200"
ORIGINAL_DIR = "../../roboverse/scripts/office_welding_expert_100"
# ORIGINAL_DIR = "../../roboverse/scripts/office_TA_pp_with_prop"
num_demos = 100

# ROOT_DIR = "office_erasers_more"
# ORIGINAL_DIR = "../../roboverse/scripts/office_eraser_more"
# num_demos = 400

# env_args = json.dumps({
#         "env_name" : "Widow250OfficeRand-v0",
#         "type" : ROBOVERSE_TYPE,
#         "env_kwargs": {
#             "object_names" : ('eraser', 'shed', 'pepsi_bottle', 'gatorade'), #, 'eraser_2', 'shed_2', 'pepsi_bottle_2'),
#             "object_targets" : ('tray', 'container'), # 'drawer_inside'),
#             "target_object" : 'eraser',
#             'random_shuffle_object': True,
#             'random_shuffle_target': True,
#             'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
#             "reward_type" : "pick_place",
#             "observation_img_dim" : IMG_DIM,
#         }
#     }
# )

env_args = json.dumps({
    "env_name" : "Widow250OfficeRand-v0",
    "type" : ROBOVERSE_TYPE,
    "env_kwargs": {
        "accept_trajectory_key": "table_clean",  # needed for the wrapper
        "observation_img_dim": 84,
          "observation_mode": "pixels_eye_hand",
          "control_mode": "discrete_gripper",
          "num_objects": 2,
          "object_names": [
            "eraser",
            "shed",
            "pepsi_bottle",
            "gatorade"
          ],
          "random_shuffle_object": False,
          "random_shuffle_target": False,
          "object_targets": [
            "tray",
            "container"
          ],
          "desired_config": {
            "eraser": "tray",
            "shed": "container"
          }
    }
})


data_writer = h5py.File(f'../datasets/office/{ROOT_DIR}/office_image_WELD.hdf5', 'w')
data_grp = data_writer.create_group("data")
data_grp.attrs["env_args"] = env_args

ep_count = 0
total_samples = 0

# for j in range(1): #just testing on 170 demos
j = 1 #used to be 0

video_skip = 10 #used to be 10

for i in tqdm.tqdm(range(num_demos)):
    try:
        rollout = h5py.File(f"{ORIGINAL_DIR}/rollout_{i}.h5", 'r')
    except:
        continue
    traj = rollout["traj0"]
    ep_data_grp = data_grp.create_group("demo_{}".format(ep_count))
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
    # ep_data_grp.create_dataset("states", data=np.array(traj["states"])) #not this state!
    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
    ep_data_grp.create_dataset("dones", data=np.zeros_like(np.array(traj["rewards"])))

    if "relevant_demo" in traj: #marking oracle state
        ep_data_grp.attrs["target"] = traj["relevant_demo"][()] #janky but it works
    elif "relevant_demo" in rollout:
        ep_data_grp.attrs["target"] = rollout["relevant_demo"][()] #janky but it works
    else:
        raise Exception("no oracle state found!")

    state_arr = np.array(traj["states"])
    next_state_arr = np.roll(state_arr, -1, axis = 0)
    next_state_arr[-1] = 0

    robot_arr = np.array(traj["proprio"])
    next_robot_arr = np.roll(robot_arr, -1, axis=0)
    next_robot_arr[-1] = 0

    img_arr = np.array(traj["image"])
    next_img_arr = np.roll(img_arr, -1, axis = 0)
    next_img_arr[-1] = 0

    hand_img_arr = np.array(traj["image_eye_in_hand"])
    next_hand_img_arr = np.roll(hand_img_arr, -1, axis=0)
    next_hand_img_arr[-1] = 0

    ep_data_grp.create_dataset("obs/state", data=state_arr)
    ep_data_grp.create_dataset("next_obs/state", data=next_state_arr)
    ep_data_grp.create_dataset("obs/robot", data=robot_arr)
    ep_data_grp.create_dataset("next_obs/robot", data=next_robot_arr)

    if ep_count % video_skip == 0:
        video_writer = imageio.get_writer(f"../datasets/office/{ROOT_DIR}/videos/demo_{ep_count}.gif", fps=20)
        for k in range(img_arr.shape[0]):
            combined_frame = np.concatenate((img_arr[k], hand_img_arr[k]), axis = 0)
            video_writer.append_data(combined_frame)
        video_writer.close()

    ep_data_grp.create_dataset("obs/agentview_image", data = img_arr)
    ep_data_grp.create_dataset("next_obs/agentview_image", data = next_img_arr)

    ep_data_grp.create_dataset("obs/robot0_eye_in_hand_image", data = hand_img_arr)
    ep_data_grp.create_dataset("next_obs/robot0_eye_in_hand_image", data = next_hand_img_arr)

    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
    ep_count += 1
    total_samples += ep_data_grp.attrs["num_samples"]

data_grp.attrs["total"] = total_samples
print(total_samples)
print(ep_count)
data_writer.close()

# traj0: actions, pad_mask, rewards, states
    # states contain the simulation state. Extract to replay!
# traj_per_file [empty]



### FOR ROBOMIMIC STYLE DEMOS ##
# hf = h5py.File('../datasets/can/paired/low_dim.hdf5', 'r')
#
# env_meta = json.loads(hf["data"].attrs["env_args"])

# env_meta["env_kwargs"] are the environment kwargs that get passed directly to the constructor
# stored in env_args

# data, mask
# data:
    # attrs: env_args, total
    # demo_0, demo_1, ...
        # demo_x:
            # attrs: model_file, num_samples
            # 'actions', 'dones', 'next_obs', 'obs', 'rewards', 'states'
