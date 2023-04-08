from roboverse.envs.widow250 import Widow250Env
import roboverse
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
import numpy as np
import itertools
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS


class Widow250DrawerEnv(Widow250Env):

    def __init__(self,
                 drawer_pos=(0.5, 0.2, -.35),
                 drawer_quat=(0, 0, 0.707107, 0.707107),
                 left_opening=True,  # False is not supported
                 start_opened=True,
                 blocking_object_in_tray=True,
                 **kwargs):
        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.left_opening = left_opening
        self.start_opened = start_opened
        self.blocking_object_in_tray = blocking_object_in_tray
        self.drawer_opened_success_thresh = 0.95
        self.drawer_closed_success_thresh = 0.05
        obj_pos_high, obj_pos_low = self.get_obj_pos_high_low()
        super(Widow250DrawerEnv, self).__init__(
            object_position_high=obj_pos_high,
            object_position_low=obj_pos_low,
            **kwargs
        )

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        object_positions = object_utils.generate_object_positions(
            self.object_position_low, self.object_position_high,
            self.num_objects,
        )

        self.original_object_positions = object_positions

        self.objects["drawer"] = object_utils.load_object(
            "drawer", self.drawer_pos, self.drawer_quat, scale=0.1)
        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.objects['drawer'])[0]

        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

        opened_drawer_x_pos = object_utils.close_drawer(
            self.objects['drawer'])[0]

        if self.left_opening:
            self.drawer_min_x_pos = closed_drawer_x_pos
            self.drawer_max_x_pos = opened_drawer_x_pos
        else:
            self.drawer_min_x_pos = opened_drawer_x_pos
            self.drawer_max_x_pos = closed_drawer_x_pos

        if self.start_opened:
            object_utils.open_drawer(self.objects['drawer'])

    def get_obj_pos_high_low(self):
        obj_pos_high = np.array(self.drawer_pos[:2] + (-.2,)) \
                       + (1 - 2 * (not self.left_opening)) * np.array((0.12, 0, 0))
        obj_pos_low = np.array(self.drawer_pos[:2] + (-.2,)) \
            - (1 - 2 * (not self.left_opening)) * np.array((-0.12, 0, 0))

        # randomization along x-axis
        obj_pos_high[0] = obj_pos_high[0] + 0.015
        obj_pos_low[0] = obj_pos_low[0] - 0.015

        # randomization along y-axis
        # obj_pos_high[1] = obj_pos_high[1] + 0.01
        # obj_pos_low[1] = obj_pos_low[1] - 0.01
        
        return obj_pos_high, obj_pos_low

    def get_info(self):
        info = super(Widow250DrawerEnv, self).get_info()
        info['drawer_x_pos'] = self.get_drawer_pos()[0]
        info['drawer_opened_percentage'] = \
            self.get_drawer_opened_percentage()
        info['drawer_closed_percentage'] = \
            1 - self.get_drawer_opened_percentage()

        info['drawer_opened_success'] = info["drawer_opened_percentage"] > \
            self.drawer_opened_success_thresh

        info['drawer_closed_success'] = info["drawer_opened_percentage"] < \
            self.drawer_closed_success_thresh
            
        return info

    def is_drawer_closed(self):
        info = self.get_info()
        return info['drawer_closed_success']

    def get_drawer_handle_pos(self):
        handle_pos = object_utils.get_drawer_handle_pos(
            self.objects["drawer"])
        return handle_pos

    def is_drawer_open(self):
        # refers to bottom drawer in the double drawer case
        info = self.get_info()
        return info['drawer_opened_success']

    def get_drawer_opened_percentage(self, drawer_key="drawer"):
        # compatible with either drawer or upper_drawer
        drawer_x_pos = self.get_drawer_pos(drawer_key)[0]
        return object_utils.get_drawer_opened_percentage(
            self.left_opening, self.drawer_min_x_pos,
            self.drawer_max_x_pos, drawer_x_pos)

    def get_drawer_pos(self, drawer_key="drawer"):
        drawer_pos = object_utils.get_drawer_pos(
            self.objects[drawer_key])
        return drawer_pos

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "opening":
            return float(self.is_drawer_open())
        else:
            return super(Widow250DrawerEnv, self).get_reward(info)


class Widow250DrawerRandomizedEnv(Widow250DrawerEnv):

    def __init__(self, **kwargs):
        drawer_pos, drawer_quat, left_opening = \
            self.set_drawer_pos_and_quat()
        super(Widow250DrawerRandomizedEnv, self).__init__(
            drawer_pos=drawer_pos,
            drawer_quat=drawer_quat,
            left_opening=left_opening,
            **kwargs
        )

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        object_positions = object_utils.generate_object_positions(
            self.object_position_low, self.object_position_high,
            self.num_objects,
        )
        self.original_object_positions = object_positions

        self.objects["drawer"] = object_utils.load_object(
            "drawer", self.drawer_pos, self.drawer_quat, scale=0.1)
        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.objects['drawer'])[0]

        opened_drawer_x_pos = object_utils.close_drawer(
            self.objects['drawer'])[0]

        if self.left_opening:
            self.drawer_min_x_pos = closed_drawer_x_pos
            self.drawer_max_x_pos = opened_drawer_x_pos
        else:
            self.drawer_min_x_pos = opened_drawer_x_pos
            self.drawer_max_x_pos = closed_drawer_x_pos

        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def get_obj_pos_high_low(self):
        obj_pos_high = tuple(
            np.array(self.drawer_pos) + np.array([0.05, 0.06, 0.1]))
        obj_pos_low = tuple(
            np.array(self.drawer_pos) + np.array([-0.05, -0.06, 0.1]))
        return obj_pos_high, obj_pos_low

    def set_drawer_pos_and_quat(self):
        num_possible_drawer_positions = 4
        drawer_positions = [
            (0.75 - 0.1 * i, 0.15, -.35)
            for i in range(num_possible_drawer_positions)
        ]
        pos_orientation_pairings = list(itertools.product(
            [0], drawer_positions[:2])) + \
            list(itertools.product([1], drawer_positions[2:]))

        chosen_idx = np.random.randint(len(pos_orientation_pairings))
        left_opening, drawer_position = \
            pos_orientation_pairings[chosen_idx]

        drawer_quat = [(0, 0, -0.707107, 0.707107),
                       (0, 0, 0.707107, 0.707107)][left_opening]
        return drawer_position, drawer_quat, left_opening

    def reset(self):
        self.drawer_pos, self.drawer_quat, self.left_opening = \
            self.set_drawer_pos_and_quat()
        self.object_position_low, self.object_position_high = \
            self.get_obj_pos_high_low()
        return super(Widow250DrawerRandomizedEnv, self).reset()


class Widow250DoubleDrawerEnv(Widow250DrawerEnv):
    def __init__(self,
                 drawer_pos=(0.5, 0.2, -.35),
                 drawer_quat=(0, 0, 0.707107, 0.707107),
                 left_opening=True,  # False is not supported
                 start_opened=False,
                 start_top_opened=False,
                 **kwargs):
        self.start_top_opened = start_top_opened
        self.tray_position = (.8, 0.0, -.37)

        super(Widow250DoubleDrawerEnv, self).__init__(
            drawer_pos=drawer_pos,
            drawer_quat=drawer_quat,
            left_opening=left_opening,  # False is not supported
            start_opened=start_opened,
            **kwargs,
        )

    def _load_meshes(self):
        super(Widow250DoubleDrawerEnv, self)._load_meshes()
        self.objects["drawer_top"] = object_utils.load_object(
            "drawer_no_handle", self.drawer_pos + np.array([0, 0, 0.07]),
            self.drawer_quat, scale=0.1)

        self.tray_id = objects.tray(base_position=self.tray_position, scale=0.3)

        self.blocking_object_name = 'gatorade'

        if self.blocking_object_in_tray:
            object_position = np.random.uniform(
                low=(.79, .0, -.34), high=(.81, .0, -.34))
            # TODO Maybe randomize the quat as well
            object_quat = (0, 0, 1, 0)
        else:
            object_position = np.random.uniform(
                low=(.63, .2, -.34), high=(.65, .2, -.34))
            object_quat = (0, 0, 1, 0)

        self.blocking_object = object_utils.load_object(
            self.blocking_object_name,
            object_position,
            object_quat=object_quat,
            scale=1.0)

        bullet.step_simulation(self.num_sim_steps_reset)

        if self.start_top_opened:
            object_utils.open_drawer(
                self.objects["drawer_top"], half_open=True)

    def get_info(self):
        info = super(Widow250DoubleDrawerEnv, self).get_info()
        info['drawer_top_x_pos'] = self.get_drawer_pos("drawer_top")[0]
        info['drawer_top_opened_percentage'] = \
            self.get_drawer_opened_percentage("drawer_top")
        info['drawer_top_closed_success'] = info["drawer_top_opened_percentage"] \
            < self.drawer_closed_success_thresh
        return info

    def is_top_drawer_closed(self):
        info = self.get_info()
        return info['drawer_top_closed_success']


if __name__ == "__main__":
    env = roboverse.make('Widow250DoubleDrawerCloseOpenNeutral-v0',
                         gui=True, transpose_image=False)

    # env = Widow250DoubleDrawerEnv(
    #     gui=True,
    #     drawer_pos=(0.47, 0.2, -.35)
    # )
    import time
    env.reset()
    # import IPython; IPython.embed()

    for j in range(5):
        for i in range(20):
            obs, rew, done, info = env.step(
                np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0., 0.]))
            print("reward", rew, "info", info)
            time.sleep(0.1)
        env.reset()
