import gym
import numpy as np

from roboverse.bullet.serializable import Serializable
import roboverse.bullet as bullet
from roboverse.envs import objects
from roboverse.bullet import object_utils
from roboverse.envs.multi_object import MultiObjectEnv
from roboverse.utils import transform_utils as T
import math

END_EFFECTOR_INDEX = 8
# RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.036, -0.036]
RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.036, -0.036]
# RESET_JOINT_VALUES = [0, 0.0, 0.0, 0.0, -1.57, 0., 0., 0.036, -0.036]
# [z, ]
RESET_JOINT_VALUES_GRIPPER_CLOSED = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.015, -0.015]
RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
GUESS = 3.14  # TODO(avi) This is a guess, need to verify what joint this is
# JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
#                      -0.037]

# JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]

JOINT_LIMIT_LOWER = [-3.14, -1.6, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
                     -0.037]

JOINT_LIMIT_UPPER = [3.14, 1.6, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]



JOINT_RANGE = []
for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
    JOINT_RANGE.append(upper - lower)

GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]
GRIPPER_OPEN_STATE = [0.036, -0.036]
GRIPPER_CLOSED_STATE = [0.015, -0.015]

ACTION_DIM = 8


def quat_to_rpy(quat):
    return T.mat2euler(T.quat2mat(quat))

class Widow250Env(gym.Env, Serializable):

    def __init__(self,
                 control_mode='continuous',
                 observation_mode='pixels',
                 observation_img_dim=48,
                 transpose_image=True,

                #  object_names=('bowl_small', 'gatorade'),
                 object_names=('beer_bottle', 'gatorade'),

                 object_scales=(0.75, 0.75),
                 object_orientations=((0, 0, 1, 0), (0, 0, 1, 0)),
                 object_position_high=(.7, .27, -.35), # (.7, .27, -.35)
                 object_position_low=(.5, .18, -.35),
                 target_object='gatorade',
                 load_tray=True,
                 num_sim_steps=10,
                 num_sim_steps_reset=50,
                 num_sim_steps_discrete_action=75,

                 reward_type='grasping',
                 grasp_success_height_threshold=-0.3,
                 grasp_success_object_gripper_threshold=0.25,

                 use_neutral_action=False,
                 neutral_gripper_open=True,
                 num_objects = 3,
                 xyz_action_scale=0.2,
                 abc_action_scale=20.0,
                 gripper_action_scale=20.0,

                 ee_pos_high=(1.2, .5, -0.1),
                 ee_pos_low=(.2, -.3, -.34),

                # ee_pos_high=(1.2, .4, -0.1),
                #  ee_pos_low=(.2, -.2, -.34),
                 camera_target_pos=(0.6, 0.2, -0.28),
                 camera_distance=0.29,
                 camera_roll=0.0,
                 camera_pitch=-40,
                 camera_yaw=180,

                 gui=False,
                 in_vr_replay=False,
                 ):
        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.observation_img_dim = observation_img_dim
        self.transpose_image = transpose_image

        self.num_sim_steps = num_sim_steps
        self.num_sim_steps_reset = num_sim_steps_reset
        self.num_sim_steps_discrete_action = num_sim_steps_discrete_action

        self.reward_type = reward_type
        self.grasp_success_height_threshold = grasp_success_height_threshold
        self.grasp_success_object_gripper_threshold = \
            grasp_success_object_gripper_threshold

        self.use_neutral_action = use_neutral_action
        self.neutral_gripper_open = neutral_gripper_open

        self.gui = gui
        # TODO(avi): This hard-coding should be removed
        self.fc_input_key = 'state'
        self.cnn_input_key = 'image'
        self.terminates = False
        self.scripted_traj_len = 30

        # TODO(avi): Add limits to ee orientation as well
        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        bullet.connect_headless(self.gui)

        # object stuff
        assert target_object in object_names
        assert len(object_names) == len(object_scales)
        # self.num_objects = num_objects
        self.load_tray = load_tray
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self.object_names = object_names
        self.target_object = target_object
        self.object_scales = dict()
        self.object_orientations = dict()
        for orientation, object_scale, object_name in \
                zip(object_orientations, object_scales, self.object_names):
            self.object_orientations[object_name] = orientation
            self.object_scales[object_name] = object_scale

        self.in_vr_replay = in_vr_replay
        self._load_meshes()
        self.movable_joints = bullet.get_movable_joints(self.robot_id)
        self.end_effector_index = END_EFFECTOR_INDEX
        self.reset_joint_values = RESET_JOINT_VALUES
        self.reset_joint_indices = RESET_JOINT_INDICES

        self.xyz_action_scale = xyz_action_scale
        self.abc_action_scale = abc_action_scale
        self.gripper_action_scale = gripper_action_scale

        self.camera_target_pos = camera_target_pos
        self.camera_distance = camera_distance
        self.camera_roll = camera_roll
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance,
                                yaw=self.camera_yaw,
                                pitch=self.camera_pitch,
                                roll=self.camera_roll,
                                up_axis_index=2)
        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_dim, self.observation_img_dim)

        self._set_action_space()
        self._set_observation_space()

        self.is_gripper_open = True  # TODO(avi): Clean this up

        self.reset()
        self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        if self.in_vr_replay:
            object_positions = self.original_object_positions
        else:
            object_positions = object_utils.generate_object_positions(
                self.object_position_low, self.object_position_high,
                self.num_objects,
            )
            self.original_object_positions = object_positions
        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True  # TODO(avi): Clean this up

        return self.get_observation()

    def set_robot_state(self, target_ee_pos, target_ee_quat, target_gripper_state):
        bullet.apply_action_ik(
            target_ee_pos, target_ee_quat, target_gripper_state,
            self.robot_id,
            self.end_effector_index, self.movable_joints,
            lower_limit=JOINT_LIMIT_LOWER,
            upper_limit=JOINT_LIMIT_UPPER,
            rest_pose=RESET_JOINT_VALUES,
            joint_range=JOINT_RANGE,
            num_sim_steps=100) #self.num_sim_steps)

        gripper_state = self.get_gripper_state()
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        return ee_pos, ee_quat, gripper_state

    def step(self, action):

        # TODO Clean this up
        if np.isnan(np.sum(action)):
            print('action', action)
            raise RuntimeError('Action has NaN entries')

        action = np.clip(action, -1, +1)  # TODO Clean this up

        xyz_action = action[:3]  # ee position actions
        abc_action = action[3:6]  # ee orientation actions
        gripper_action = action[6]
        neutral_action = action[7]

        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])

        target_ee_pos = ee_pos + self.xyz_action_scale * xyz_action
        ee_deg = bullet.quat_to_deg(ee_quat)
        target_ee_deg = ee_deg + self.abc_action_scale * abc_action
        target_ee_quat = bullet.deg_to_quat(target_ee_deg)

        if self.control_mode == 'continuous':
            num_sim_steps = self.num_sim_steps
            target_gripper_state = gripper_state + \
                                   [-self.gripper_action_scale * gripper_action,
                                    self.gripper_action_scale * gripper_action]

        elif self.control_mode == 'discrete_gripper':
            if gripper_action > 0.5 and not self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_OPEN_STATE
                self.is_gripper_open = True  # TODO(avi): Clean this up

            elif gripper_action < -0.5 and self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_CLOSED_STATE
                self.is_gripper_open = False  # TODO(avi): Clean this up
            else:
                num_sim_steps = self.num_sim_steps
                if self.is_gripper_open:
                    target_gripper_state = GRIPPER_OPEN_STATE
                else:
                    target_gripper_state = GRIPPER_CLOSED_STATE
                # target_gripper_state = gripper_state
        else:
            raise NotImplementedError

        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_low,
                                self.ee_pos_high)
        target_gripper_state = np.clip(target_gripper_state, GRIPPER_LIMITS_LOW,
                                       GRIPPER_LIMITS_HIGH)
        # import pdb; pdb.set_trace()

        bullet.apply_action_ik(
            target_ee_pos, target_ee_quat, target_gripper_state,
            self.robot_id,
            self.end_effector_index, self.movable_joints,
            lower_limit=JOINT_LIMIT_LOWER,
            upper_limit=JOINT_LIMIT_UPPER,
            rest_pose=RESET_JOINT_VALUES,
            joint_range=JOINT_RANGE,
            num_sim_steps=num_sim_steps)

        if self.use_neutral_action and neutral_action > 0.5:
            if self.neutral_gripper_open:
                bullet.move_to_neutral(
                    self.robot_id,
                    self.reset_joint_indices,
                    RESET_JOINT_VALUES)
            else:
                bullet.move_to_neutral(
                    self.robot_id,
                    self.reset_joint_indices,
                    RESET_JOINT_VALUES_GRIPPER_CLOSED)

        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return self.get_observation(), reward, done, info

    def get_observation(self):
        print(f"observation mode: {self.observation_mode}")
        gripper_state = self.get_gripper_state()
        gripper_binary_state = [float(self.is_gripper_open)]
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        # import ipdb;ipdb.set_trace()
        object_position, object_orientation = bullet.get_object_position(
            self.objects[self.target_object])

        if self.observation_mode == 'pixels':
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten()) / 255.0
            observation = {
                'object_position': object_position,
                'object_orientation': object_orientation,
                'state': np.concatenate(
                    (ee_pos, ee_quat, gripper_state, gripper_binary_state)),
                'image': image_observation
            }
        elif self.observation_mode == 'pixels_eye_hand':
            image_observation, eye_in_hand = self.render_obs(eye_in_hand = True)
            image_observation = np.float32(image_observation.flatten()) / 255.0
            observation = {
                'object_position': object_position,
                'object_orientation': object_orientation,
                'state': np.concatenate(
                    (ee_pos, ee_quat, gripper_state, gripper_binary_state)),
                'image': image_observation,
                'image_eye_in_hand' : eye_in_hand
            }
        else:
            # raise NotImplementedError
            observation = {
                'object_position': object_position,
                'object_orientation': object_orientation,
                'state': np.concatenate(
                    (ee_pos, ee_quat, gripper_state, gripper_binary_state)),
            }

        return observation

    def get_reward(self, info):
        if self.reward_type == 'grasping':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = {'grasp_success': False}
        for object_name in self.object_names:
            grasp_success = object_utils.check_grasp(
                object_name, self.objects, self.robot_id,
                self.end_effector_index, self.grasp_success_height_threshold,
                self.grasp_success_object_gripper_threshold)
            if grasp_success:
                info['grasp_success'] = True

        info['grasp_success_target'] = object_utils.check_grasp(
            self.target_object, self.objects, self.robot_id,
            self.end_effector_index, self.grasp_success_height_threshold,
            self.grasp_success_object_gripper_threshold)
        return info

    def render_obs(self, res=None, eye_in_hand = False):
        res = self.observation_img_dim if res is None else res
        img, depth, segmentation = bullet.render(
            res, res,
            self._view_matrix_obs, self._projection_matrix_obs, shadow=0)

        if eye_in_hand:
            ee_pos, ee_quat = bullet.get_link_state(
                self.robot_id, self.end_effector_index)

            roll, pitch, yaw = quat_to_rpy(ee_quat)

            # have camera focus on table
            target_pos = ee_pos.copy()
            target_pos[2] = -0.4
            target_pos[1] += 0.07
            distance = 0.11 + ee_pos[2] - (-0.4)
            eye_hand_view_matrix_args = dict(target_pos=target_pos,
                                    distance=distance,
                                    yaw=math.degrees(yaw),
                                    pitch=math.degrees(pitch) - 90,
                                    roll=math.degrees(yaw),
                                    up_axis_index=2)

            eye_hand_view_matrix_obs = bullet.get_view_matrix(**eye_hand_view_matrix_args)
            eye_hand_projection_matrix_obs = bullet.get_projection_matrix(
                self.observation_img_dim, self.observation_img_dim)

            eye_hand_img, _, _ = bullet.render(
                res, res,
                eye_hand_view_matrix_obs, eye_hand_projection_matrix_obs, shadow=0)

        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))

        if eye_in_hand:
            return img, eye_hand_img

        return img

    def _set_action_space(self):
        self.action_dim = ACTION_DIM
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.observation_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            # self.image_length = (self.observation_img_dim ** 2) * 3
            # img_space = gym.spaces.Box(0, 1, (self.image_length,),
            #                            dtype=np.float32)
            robot_state_dim = 76  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
            # raise NotImplementedError

    def get_gripper_state(self):
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray(joint_states[-2:])
        return gripper_state

    def close(self):
        bullet.disconnect()





class Widow250MultiObjectEnv(MultiObjectEnv, Widow250Env):
    """Grasping Env but with a random object each time."""


if __name__ == "__main__":
    env = Widow250Env(gui=True)
    import time

    env.reset()
    # import IPython; IPython.embed()

    for i in range(200):
        ee_pos, ee_orientation = bullet.get_link_state(
            env.robot_id, env.end_effector_index)
        print(f"ee_pos: {ee_pos}, ee_deg: {bullet.quat_to_deg(ee_orientation)}")
        obs, rew, done, info = env.step(
            np.asarray([0.0, 0., 0.0, 0.0, 0.0, 0.0, 0., 0.]))
        time.sleep(0.1)

    env.reset()
    time.sleep(1)
    for _ in range(100):
        env.step(np.asarray([0., 0., 0., 0., 0., 0., 0.0, 0.]))
        time.sleep(0.1)

    env.reset()
