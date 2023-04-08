"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

# import mujoco_py
# import robosuite
# from robosuite.utils.mjcf_utils import postprocess_model_xml

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

import roboverse
from roboverse.utils import get_timestamp
from roboverse.assets.shapenet_object_lists \
    import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS, PICK_PLACE_TRAIN_OBJECTS, \
    PICK_PLACE_TEST_OBJECTS, TRAIN_CONTAINERS, TEST_CONTAINERS, PICK_PLACE_DEMO_CONTAINERS, PICK_PLACE_DEMO_OBJECTS


class EnvRoboverse(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""

    def __init__(
            self,
            env_name,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
            postprocess_visual_obs=True,
            accept_trajectory_key = "target_place_success", # required default for backward compatibility
            **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            observation_mode="pixels_eye_hand" if use_image_obs else "noimage",
            control_mode = "discrete_gripper",
        )

        kwargs.update(update_kwargs)

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)

        self.env = roboverse.make(env_name,
                             gui=render,
                             transpose_image=False, **kwargs)
        self.accept_trajectory_key = accept_trajectory_key
        self.total_reward = 0
        self.total_reward_thresh = sum([subtask.REWARD for subtask in self.env.subtasks])

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        self.total_reward += r
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        self.total_reward = 0
        self.total_reward_thresh = sum([subtask.REWARD for subtask in self.env.subtasks])
        return self.get_observation(di)

    def reset_to(self, state):
        #not working!
        return


    def render(self, mode="rgb_array", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        assert mode != "human"
        #TODO: IMPLEMENT EYE IN HAND HERE
        if mode == "rgb_array":
            if camera_name == "robot0_eye_in_hand_image": #compatible with the robomimic utils
                return self.env.render_obs(res = height, eye_in_hand = True)[1] #the rendering engine call directly
            elif camera_name == "agentview":
                return self.env.render_obs(res=height, eye_in_hand = False)
            else:
                raise Exception("invalid camera!")
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    # use this to get any sort of information that you need to get a demostration working
    # def get_priv_info(self):
    #     return self.env.get_priv_info()

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.
        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide
                as a dictionary. If not provided, will be queried from robosuite.
        """
        # di = self.env.get_observation()
        if di is None:
            raise Exception("not implemented!")
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}


        for k in di:
            k_label = k
            if k == "image":
                k_label = "agentview_image"
            elif k == "image_eye_in_hand":
                k_label = "robot0_eye_in_hand_image"

            if (k_label in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k_label, obs_modality="rgb"):
                ret[k_label] = di[k][::-1] #extract with original keys
                if self.postprocess_visual_obs:
                    ret[k_label] = ObsUtils.process_obs(obs=ret[k_label], obs_key=k_label)

        # ret["proprio"] = np.array(di["robot_state"])
        # ret["eef_pos"] = np.array(di["eef_pos"])
        # ret["eef_quat"] = np.array(di["eef_quat"])
        # ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        # ret["object"] = np.array(di["object"])
        ret["state"] = np.array(di["state"])
        ret["robot"] = np.array(di["robot"])

        # quirk in the environment
        if "agentview_image" in ret:
            ret["agentview_image"] = np.flip(ret["agentview_image"], axis = 1).copy()

        if "robot0_eye_in_hand_image" in ret:
            ret["robot0_eye_in_hand_image"] = np.flip(ret["robot0_eye_in_hand_image"], axis = 1).copy()

        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        return {}
        # p1 = self.env.env._p
        # base_po = []  # position and orientation of base for each body
        # base_v = []  # velocity of base for each body
        # joint_states = []  # joint states for each body
        # for i in range(p1.getNumBodies()):
        #     base_po.append(p1.getBasePositionAndOrientation(i))
        #     base_v.append(p1.getBaseVelocity(i))
        #     joint_states.append([p1.getJointState(i, j) for j in range(p1.getNumJoints(i))])
        # return {"states" : (base_po, base_v, joint_states)}

    def set_whole_state(self, state):
        #TODO: parsing here allows for separate state inputs for robot, drawers, etc
        self.env.set_whole_state(state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def get_priv_info(self):
        return {}

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """

        info = self.env.get_info()
        success = False
        # print(self.total_reward, self.total_reward_thresh)
        if self.accept_trajectory_key == 'table_clean':
            print(self.total_reward)
            if self.total_reward == self.total_reward_thresh :
                success = True
                # print(f"time {j}")
        else:
            # print(info)
            if info[self.accept_trajectory_key]:
                success = True

        # status = {"task" : info["target_place_success"]}
        status = {"task" : success}
        # print(status)
        # DO NOT CHANGE UNTIL CURRENT OFFICE RUSN ARE DONE
        return status #{"task" : info["target_place_success"]} #gets the target task

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_space.shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOVERSE_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(
            cls,
            env_name,
            camera_names,
            camera_height,
            camera_width,
            reward_shaping,
            **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions.

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        raise Exception("not ready yet")
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [],  # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False,
            render_offscreen=has_camera,
            use_image_obs=has_camera,
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        raise Exception("not ready yet")
        return (mujoco_py.builder.MujocoException)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
