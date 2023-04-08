from roboverse.envs.widow250_pickplace import Widow250PickPlaceEnv
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
from roboverse.assets.shapenet_object_lists import TRAIN_OBJECTS
from roboverse.utils.general_utils import AttrDict
import numpy as np
import random
from roboverse.envs.tasks import DrawerOpenTask, DrawerClosedTask, PickTask, PlaceTask


class Widow250OfficeEnv(Widow250PickPlaceEnv):
    def __init__(self,
                container_name='open_box',
                reward_type='pick_place',

                num_objects=4,
                object_names=('eraser', 'shed', 'pepsi_bottle', 'gatorade'), #, 'eraser_2', 'shed_2', 'pepsi_bottle_2'),
                # object_targets=('tray', 'container', 'drawer_inside'),
                object_targets=('tray', 'container'), #
                desired_config = None, #{"eraser" : "tray"},
                target_object='eraser',
                target_target = "tray",
                object_scales=(0.8, 0.8, 0.8, 0.8), #, 0.8, 0.8, 0.8),
                object_orientations=((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                                     # (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),

                object_position_high=(0.75, .9, -.35),
                object_position_low=(.3, .1, -.35),
                original_object_positions = (
                        (0.33620103,  0.12358467, -0.35),
                        (0.55123888, -0.17699107, -0.35),
                        (0.84287004, -0.1479069 , -0.35),
                        (0.75200037, -0.14383595, -0.35),
                        # (0.42755662, -0.13711447, -0.35),
                        # (0.39866522,  0.18929185, -0.35),
                        # (0.46422192, -0.23138137, -0.35),
                ),
                area_upper_left_low=(0.75, -0.15, -0.35),
                area_upper_left_high=(0.85, -0.1, -0.35),
                area_upper_middle_low=(0.4, -0.27, -0.35),
                area_upper_middle_high=(0.56, -0.13, -0.35),
                area_lower_right_low=(0.32, 0.12, -0.35),
                area_lower_right_high=(0.4, 0.19, -0.35),

                possible_objects=TRAIN_OBJECTS[:10],
                drawer_pos=(0.1, 0.0, -.35),
                random_drawer = False,
                drawer_pos_low = (0.1, 0.0, -.35),
                drawer_pos_high = (0.2, 0.2, -.35),
                drawer_quat=(0, 0, 0.707107, 0.707107),
                left_opening=True,  # False is not supported
                start_opened=False,

                min_distance_from_object = 0.12,
                min_distance_drawer=0.2,
                min_distance_container=0.11,
                min_distance_obj=0.09,

                load_tray = True,
                random_tray=False,
                tray_position = (0.26,-0.2, -.39),
                tray_position_high=(-0.1,-0.5, -.39),
                tray_position_low=(-0.1,-0.5, -.39),

                base_position_high=(0.62, 0.02, -0.4),
                base_position_low=(0.58, -0.02, -0.4),
                base_position=(0.6, 0.0, -0.4),
                base_orientation = (0, 0, -180),
                base_orientation_high =(0, 0, -182),
                base_orientation_low = (0, 0, -188),
                random_base = False,
                random_base_orientation = False,
                random_joint_values = False,

                xyz_action_scale = 0.7,

                random_shuffle_object = False,
                random_shuffle_target = False,
                random_object_position = False,
                object_jitter=0.01,

                observation_img_dim=256,
                camera_distance=0.55,

                fixed_init_pos=None,
                drawer_number = 2,
                **kwargs):

        self.random_object_position = random_object_position
        self.area_upper_left_low = area_upper_left_low
        self.area_upper_left_high = area_upper_left_high
        self.area_upper_middle_low = area_upper_middle_low
        self.area_upper_middle_high = area_upper_middle_high
        self.area_lower_right_low = area_lower_right_low
        self.area_lower_right_high = area_lower_right_high

        self.load_tray = load_tray
        self.tray_position = tray_position
        self.random_tray = random_tray
        self.tray_position_high = tray_position_high
        self.tray_position_low = tray_position_low

        self.random_base = random_base
        self.base_position = base_position
        self.base_position_high = base_position_high
        self.base_position_low = base_position_low
        self.random_base_orientation = random_base_orientation
        self.base_orientation = base_orientation
        self.base_orientation_high = base_orientation_high
        self.base_orientation_low = base_orientation_low
        self.random_joint_values = random_joint_values
        self.drawer_number = drawer_number
        self.random_drawer = random_drawer
        self.drawer_pos_low = drawer_pos_low
        self.drawer_pos_high = drawer_pos_high

        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.left_opening = left_opening
        self.start_opened = start_opened

        self.target_object = target_object
        self.target_object_target = target_target
        self.desired_config = desired_config #for oracle assignment only

        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.left_opening = left_opening
        self.start_opened = start_opened

        self.drawer_opened_success_thresh = 0.75
        self.drawer_closed_success_thresh = 0.4
        self.possible_objects = np.asarray(possible_objects)
        self.num_objects = num_objects
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self.original_object_positions = list(original_object_positions)
        self.object_names = list(object_names)
        self.object_targets = list(object_targets)
        self.random_shuffle_object = random_shuffle_object
        if self.random_shuffle_object:
            self.task_object_names = random.sample(self.object_names, self.num_objects)
        else:
            self.task_object_names = self.object_names[:self.num_objects]

        self.object_jitter = object_jitter

        self.random_shuffle_target = random_shuffle_target
        if self.random_shuffle_target:
            self.object_targets = random.sample(self.object_targets, len(self.object_targets))
            #if self.num_objects == 2:
            #    self.object_targets = [self.object_targets[0], self.object_targets[-1]]

        self.xyz_action_scale = xyz_action_scale

        self.inside_drawer_position = np.array(self.drawer_pos[:2] + (-.2,)) + np.array((0.12, 0, 0))
        self.top_drawer_position = np.array(self.drawer_pos[:2] + (0.1,))

        self.min_distance_drawer = min_distance_drawer
        self.min_distance_obj = min_distance_obj
        self.min_distance_container = min_distance_container

        self.xyz_action_scale = xyz_action_scale
        self.fixed_init_pos = fixed_init_pos
        self.subtasks = None

        super(Widow250OfficeEnv, self).__init__(
            object_names=object_names,
            target_object=target_object,
            object_orientations=object_orientations,
            object_scales=object_scales,
            container_name=container_name,
            observation_img_dim=observation_img_dim,
            camera_distance=camera_distance,
            **kwargs,
        )

    def generate_tasks(self):
        """Generate subtask list."""
        subtasks = []
        #TODO: adapt this codebase to work with single and multi-object pick-places
        for object_name, object_target in \
                zip(self.task_object_names, self.object_targets):
            object_position = self.object_name_pos_map[object_name]
            target_position = self.get_target_position(object_target)
            if object_target == 'drawer_inside':
                subtasks += [DrawerOpenTask(),
                             PickTask(object_name, object_target, object_position, target_position),
                             PlaceTask(object_name, object_target, object_position, target_position),
                             DrawerClosedTask()]
            else:
                subtasks += [PickTask(object_name, object_target, object_position, target_position),
                             PlaceTask(object_name, object_target, object_position, target_position)]

        return subtasks

    def reset(self, reshuffle = True):
        if self.random_shuffle_object and reshuffle:
            #self.object_names = random.sample(self.object_names, len(self.object_names))
            self.task_object_names = random.sample(self.object_names, self.num_objects)
        if self.random_shuffle_target and reshuffle:
            self.object_targets = random.sample(self.object_targets, len(self.object_targets))
            #if self.num_objects == 2:
            #    self.object_targets = [self.object_targets[0], self.object_targets[-1]]

        if self.random_drawer:
            self.drawer_pos = np.random.uniform(
                low=self.drawer_pos_low, high=self.drawer_pos_high)
        if self.random_tray:
            self.tray_position = np.random.uniform(
                low=self.tray_position_low, high=self.tray_position_high)
        if self.random_base:
            self.base_position = np.random.uniform(
                low=self.base_position_low, high=self.base_position_low)
        if self.random_base_orientation:
            self.base_orientation = np.random.uniform(
                low=self.base_orientation_low, high=self.base_orientation_high
            )
        if self.random_joint_values:
            bias = np.random.uniform(-0.05, 0.05)
            self.reset_joint_values[1:5] = [-0.6-bias, -0.6-bias, 0, -1.57-bias]
        # print(self.task_object_names, self.object_targets)
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True  # TODO(avi): Clean this up

        self.subtasks = self.generate_tasks()
        # print(len(self.subtasks))
        return self.get_observation()

    def get_observation(self):
        gripper_state = self.get_gripper_state()
        gripper_binary_state = [float(self.is_gripper_open)]
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)

        object_positions = []
        object_orientations = []

        drawer_pos = self.get_drawer_pos()
        drawer_handle_pos = self.get_drawer_handle_pos()

        '''
        State: 
            object_pos/orientation: 7 * 7, 
            container_pos/orientation: 7 * 3, box/tray/drawer
            drawer_handle_pos: 3,
            arm/gripper_states: 10,
        Extra:
            desk_object_pos/orientation: 7 * 2, monitor/lamp
        '''
        state = np.zeros(7 * 9 + 3 + 10)
        object_states = self.get_states(self.objects)
        container_states = self.get_states(self.containers)
        drawer_handle_state = np.asarray(self.get_drawer_handle_pos())
        arm_state = np.concatenate((ee_pos, ee_quat, gripper_state, gripper_binary_state))
        desk_object_states = self.get_states(self.desk_objects)

        state = np.concatenate((
            object_states,
            container_states,
            drawer_handle_state,
            # arm_state, #used to be here, but separated it out
            desk_object_states
        ))
        if self.observation_mode == 'pixels':
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation)  # / 255.0
            observation = {
                "robot": arm_state,
                'state': state,
                'image': image_observation
            }
        elif self.observation_mode == 'pixels_eye_hand':
            third_person, eye_in_hand = self.render_obs(eye_in_hand = True)
            third_person = np.float32(third_person)  # / 255.0
            eye_in_hand = np.float32(eye_in_hand)  # / 255.0
            observation = {
                "robot": arm_state,
                'state': state,
                'image': third_person,
                "image_eye_in_hand": eye_in_hand
            }
        else:
            observation = {
                "robot": arm_state,
                'state': state
            }

        return observation

        # proprio_state = arm_state
        # object_state = np.concatenate((
        #     object_states,
        #     container_states,
        #     drawer_handle_state,
        #     desk_object_states
        # ))
        #
        # if self.observation_mode == 'pixels':
        #     image_observation = self.render_obs()
        #     # used to be .flatten()
        #     image_observation = np.float32(image_observation) # / 255.0
        #     observation = {
        #         "robot_state": proprio_state,
        #         'object': object_state,
        #         "eef_pos" : ee_pos,
        #         "eef_quat" : ee_quat,
        #         "gripper_qpos": gripper_state,
        #         "gripper_binary": gripper_binary_state,
        #         'image': image_observation
        #     }
        # else:
        #     observation = {
        #         "robot_state": proprio_state,
        #         'object': object_state,
        #         "eef_pos" : ee_pos,
        #         "eef_quat" : ee_quat,
        #         "gripper_qpos": gripper_state,
        #         "gripper_binary": gripper_binary_state,
        #     }
        #
        # return observation

    def get_occurance(self):
        area_occurance = [0, 0, 0]
        object_occurance = {}
        for object_name in self.object_names:
            object_occurance[object_name] = 0

        for task_object_name in self.task_object_names:
            object_occurance[task_object_name] += 1
            object_pos = [self.object_name_pos_map[task_object_name]]

            if self.object_in_area(object_pos, self.area_upper_left_low, self.area_upper_left_high):
                area_occurance[0] += 1
            elif self.object_in_area(object_pos, self.area_upper_middle_low, self.area_upper_middle_high):
                area_occurance[1] += 1
            elif self.object_in_area(object_pos, self.area_lower_right_low, self.area_lower_right_high):
                area_occurance[2] += 1



        # target_index = self.task_object_names.index(self.target_object)
        # target_task = int(target_index < len(self.object_targets) and self.object_targets[target_index] == self.target_object_target)

        target_task = 1
        # basiclaly just see if this current objective satisfies the downstream task
        for object, target in self.desired_config.items():
            if object not in self.task_object_names:
                target_task = 0
                break
            obj_index = self.task_object_names.index(object)
            if self.object_targets[obj_index] != target:
                target_task = 0
                break


        # target_task = object_utils.check_in_container(self.target_object,
        #                                                      self.objects, self.get_target_position(self.target_object_target),
        #                                                      self.place_success_height_threshold,
        #                                                      # 0.25)
        #                                                      self.place_success_radius_threshold)
        # print(self.target_object, self.target_object_target, target_task)

        return area_occurance, object_occurance, int(target_task)

    def object_in_area(self, object_pos, area_low, area_high):
        if np.all(np.greater_equal(object_pos, area_low)) and np.all(np.greater_equal(area_high, object_pos)):
            return True
        else:
            return False

    def get_states(self, objects):
        object_states = np.zeros(7 * len(objects))
        for i, object_id in enumerate(objects):
            object_position, object_orientation = bullet.get_object_position(objects[object_id])
            object_states[7*i:7*i+3] = object_position
            object_states[7*i+3:7*i+7] = object_orientation
        return object_states

    def set_whole_state(self, state):
        # State:
        # object_pos / orientation: 7 * 7,
        # container_pos / orientation: 7 * 3, box / tray / drawer
        # drawer_handle_pos: 3,
        # arm / gripper_states: 10,
        # desk_object_pos / orientation: 7 * 2, monitor / lamp
        #
        state_tuple = tuple(np.split(state, [49, 70, 73, 83]))
        object, container, drawer, arm, desk = state_tuple
        ee_pos = arm[0 : 3]
        ee_quat = arm[3 : 7]
        gripper_state = arm[7 : 9]
        drawer_post = self.set_drawer_handle_pos(drawer)
        pos_post, quat_post, gripper_post = self.set_robot_state(ee_pos, ee_quat, gripper_state)

        object_post = self.set_states(self.objects, object)
        container_post = self.set_states(self.containers, container)
        desk_post = self.set_states(self.desk_objects, desk)


        # container: {np.linalg.norm(container - container_post)}
        # desk: {np.linalg.norm(desk_post - desk)},
        print(f"object:{np.linalg.norm(object_post - object)},"
              f"\t drawer:{np.linalg.norm(drawer_post - drawer)},"
              f"pos:{np.linalg.norm(pos_post - ee_pos)}, \t quat:{np.linalg.norm(quat_post - ee_quat)},"
              f"gripper:{np.linalg.norm(gripper_post - gripper_state)}")

    # set the object state in accordance to the given
    def set_states(self, objects, object_states):
        verify_object_states = np.zeros(7 * len(objects))
        for i, object_id in enumerate(objects):
            bullet.reset_object(objects[object_id], object_states[7 * i:7 * i + 3], object_states[7 * i + 3:7 * i + 7])
            object_position, object_orientation = bullet.get_object_position(objects[object_id])
            verify_object_states[7 * i:7 * i + 3] = object_position
            verify_object_states[7 * i + 3:7 * i + 7] = object_orientation
        return verify_object_states

    def get_info(self):
        info = AttrDict()
        # check whether target object is grasped
        if len(self.subtasks) > 0 and hasattr(self.current_subtask, "object"):
            info.grasp_success = object_utils.check_grasp(self.current_subtask.object,
                                                          self.objects, self.robot_id,
                                                          self.end_effector_index, self.grasp_success_height_threshold,
                                                          self.grasp_success_object_gripper_threshold)

        # check whether target object is placed in target container
        if len(self.subtasks) > 0 and hasattr(self.current_subtask, "target_pos"):
            info.place_success = object_utils.check_in_container(self.current_subtask.object,
                                                                 self.objects, self.current_subtask.target_pos,
                                                                 self.place_success_height_threshold,
                                                                 self.place_success_radius_threshold)
        # check whether target object is placed in target container
        info.target_place_success = object_utils.check_in_container(self.target_object,
                                                             self.objects, self.get_target_position(self.target_object_target),
                                                             self.place_success_height_threshold,
                                                             self.place_success_radius_threshold)
        # check whether drawer is opened
        info.drawer_opened_percentage = self.get_drawer_opened_percentage()
        info.drawer_opened = info.drawer_opened_percentage > self.drawer_opened_success_thresh
        info.drawer_closed = info.drawer_opened_percentage < self.drawer_closed_success_thresh
        # print(info)
        return info

    def get_reward(self, info):
        reward = 0.
        if self.current_subtask.done(info):
            reward += self.current_subtask.REWARD
            self.subtasks.pop(0)        # switch to the next subtask
        return reward

    def step(self, action):
        obs, reward, _, info = super().step(action)
        done = not self.subtasks
        return self.get_observation(), reward, done, info

    def get_target_position(self, object_target):
        if object_target == 'container':
            return self.container_position
        elif object_target == 'tray':
            return self.tray_position
        elif object_target == 'drawer_top':
            return list(self.top_drawer_position)
        elif object_target == 'drawer_inside':
            return list(self.inside_drawer_position)
        elif object_target == 'trashcan':
            return list(self.trashcan_position)
        else:
            raise NotImplementedError

    def generate_objects_positions(self):
        if self.fixed_init_pos is not None:
            return self.fixed_init_pos

        if self.num_objects == 1:
            container_position, object_positions = \
                object_utils.generate_object_positions_single(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        else:
            container_position, object_positions = \
                object_utils.generate_multiple_object_positions(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    drawer_pos=self.drawer_pos,
                    num_objects=self.num_objects,
                    min_distance_drawer=self.min_distance_drawer,
                    min_distance_container=self.min_distance_container,
                    min_distance_obj=self.min_distance_obj
                )
        return container_position, object_positions

    def _load_meshes(self):
        self.robot_id = objects.widow250(self.base_position, self.base_orientation)
        self.room_id = objects.room()
        self.officedesk_id = objects.officedesk()
        self.officedesk2_id = objects.officedesk(basePosition=(0.95, 0.35, -0.46))
        self.monitor_id = objects.monitor()
        self.keyboard_id = objects.keyboard()
        # self.desktop_id = objects.desktop()
        self.lamp_id = objects.lamp()

        self.desk_objects = {}
        self.desk_objects['monitor'] = self.monitor_id
        self.desk_objects['lamp'] = self.lamp_id

        self.objects = {}
        if self.load_tray:
            self.tray_id = objects.tray(base_position=self.tray_position, scale=0.3)

        self.drawer_id = object_utils.load_object(
            "drawer", self.drawer_pos, self.drawer_quat, scale=0.1)

        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.drawer_id)[0]

        opened_drawer_x_pos = object_utils.close_drawer(
            self.drawer_id)[0]

        if self.left_opening:
            self.drawer_min_x_pos = closed_drawer_x_pos
            self.drawer_max_x_pos = opened_drawer_x_pos
        else:
            self.drawer_min_x_pos = opened_drawer_x_pos
            self.drawer_max_x_pos = closed_drawer_x_pos
        if self.start_opened:
            object_utils.open_drawer(self.objects['drawer'])

        self.container_position = (0.7, -0.3, -0.35)
        self.container_id = object_utils.load_object(self.container_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        self.containers = {}
        self.containers[self.container_name] = self.container_id
        self.containers['tray'] = self.tray_id
        self.containers['drawer'] = self.drawer_id

        bullet.step_simulation(self.num_sim_steps_reset)

        # TODO: wrap random position for container and objects
        area_upper_left = object_utils.generate_two_object_positions(
            self.area_upper_left_low, self.area_upper_left_high,
            min_distance_small_obj=self.min_distance_obj,)

        area_upper_middle = object_utils.generate_three_object_positions(
                    self.area_upper_middle_low, self.area_upper_middle_high,
                    min_distance_small_obj=self.min_distance_obj,
        )
        area_lower_right = object_utils.generate_two_object_positions(
                    self.area_lower_right_low, self.area_lower_right_high,
                    min_distance_small_obj=self.min_distance_obj)

        if self.random_object_position:
            self.original_object_positions = [
                area_upper_left[0],
                area_upper_left[1],
                area_upper_middle[0],
                area_upper_middle[1],
                area_upper_middle[2],
                area_lower_right[0],
                area_lower_right[1],
            ]
            self.original_object_positions = random.sample(self.original_object_positions,
                                                            len(self.original_object_positions))
        self.object_name_pos_map = {}
        for object_name, object_position in zip(self.object_names,
                                                self.original_object_positions):
            if self.object_jitter is not None:
                object_position = np.asarray(object_position)
                object_position[:2] += self.object_jitter * 2 * (np.random.rand(2) - 0.5)
                object_position = tuple(object_position)
            self.objects[object_name] = object_utils.load_object(
                object_name[:-2] if object_name[-2] == '_' else object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            self.object_name_pos_map[object_name] = object_position
            bullet.step_simulation(self.num_sim_steps_reset)

    def set_drawer_handle_pos(self, handle_pos):
        object_utils.set_drawer_handle_pos(self.drawer_id, handle_pos)
        return object_utils.get_drawer_handle_pos(self.drawer_id)

    def get_drawer_handle_pos(self):
        handle_pos = object_utils.get_drawer_handle_pos(
            self.drawer_id)
        return handle_pos

    def is_drawer_open(self):
        # refers to bottom drawer in the double drawer case
        open_percentage = self.get_drawer_opened_percentage()
        return open_percentage > self.drawer_opened_success_thresh

    def get_drawer_opened_percentage(self, drawer_key="drawer"):
        # compatible with either drawer or upper_drawer
        drawer_x_pos = self.get_drawer_pos(drawer_key)[0]
        return object_utils.get_drawer_opened_percentage(
            self.left_opening, self.drawer_min_x_pos,
            self.drawer_max_x_pos, drawer_x_pos)

    def get_drawer_pos(self, drawer_key="drawer"):
        drawer_pos = object_utils.get_drawer_pos(
            self.drawer_id)
        return drawer_pos

    def is_drawer_closed(self):
        info = self.get_info()
        return info['drawer_closed']

    @property
    def current_subtask(self):
        return self.subtasks[0]


if __name__ == "__main__":
    from roboverse.envs.registration import ENVIRONMENT_SPECS
    kwargs = ENVIRONMENT_SPECS[-1]['kwargs']
    env = Widow250OfficeEnv(gui=True, **kwargs)
    env.reset()
    done = False
    while not done:
        action_xyz = [0., 0., 0.]
        action_angles = [0.0, 0., 0.2]
        action_gripper = [0.]
        neutral_action = [0.]
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        # obs, reward, done, info = env.step(env.action_space.sample()*0.1)
        obs, reward, done, info = env.step(action)

        print(info)
