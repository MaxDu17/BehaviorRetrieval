from roboverse.envs.widow250 import Widow250Env
import roboverse
import roboverse.bullet as bullet
from roboverse.bullet import object_utils
from roboverse.envs import objects
import numpy as np

class Widow250ButtonEnv(Widow250Env):

    def __init__(self,
                 button_pos=(0.5, 0.2, -.3),
                 button_pos_low=None,
                 button_pos_high=None,
                 button_quat=(0, 0, 0.707107, 0.707107),
                 reward_type="button_press",
                 **kwargs):
        self.button_pos = button_pos
        self.button_pos_low = button_pos_low
        self.button_pos_high = button_pos_high
        self.button_quat = button_quat
        self.button_pressed_success_thresh = 0.8
        super(Widow250ButtonEnv, self).__init__(
            reward_type=reward_type, **kwargs)

    def set_button_pos(self):
        if (self.button_pos_low is not None and
                self.button_pos_high is not None):
            rand_button_pos = object_utils.generate_object_positions(
                self.button_pos_low, self.button_pos_high, 1)[0]
            return rand_button_pos
        elif self.button_pos is not None:
            return self.button_pos

    def _load_meshes(self):
        super(Widow250ButtonEnv, self)._load_meshes()

        self.button_pos = self.set_button_pos()

        self.objects['button'] = object_utils.load_object(
            "button", self.button_pos, self.button_quat, scale=0.15)

        self.button_min_z_pos = object_utils.push_down_button(
            self.objects['button'])[2]
        self.button_max_z_pos = object_utils.pop_up_button(
            self.objects['button'])[2]

    def get_info(self):
        info = super(Widow250ButtonEnv, self).get_info()
        info['button_z_pos'] = self.get_button_pos()[2]
        info['button_pressed_percentage'] = (
            (self.button_max_z_pos - info['button_z_pos']) /
            (self.button_max_z_pos - self.button_min_z_pos))
        info['button_pressed_success'] = float(
            info['button_pressed_percentage'] >
            self.button_pressed_success_thresh)
        return info

    def get_button_pos(self):
        return object_utils.get_button_cylinder_pos(
            self.objects['button'])

    def is_button_pressed(self):
        info = self.get_info()
        return bool(info['button_pressed_percentage'] >
                    self.button_pressed_success_thresh)

    def get_reward(self, info):
        if self.reward_type == "button_press":
            return float(info['button_pressed_success'])
        elif self.reward_type == "grasp":
            return float(info['grasp_success_target'])
        else:
            raise NotImplementedError


if __name__ == "__main__":
    env = roboverse.make("Widow250RandPosButtonPressTwoObjGrasp-v0",
                         gui=True, transpose_image=False)
    import time
    env.reset()
    # import IPython; IPython.embed()

    for j in range(5):
        object_utils.pop_up_button(env.objects['button'])
        time.sleep(1)
        object_utils.push_down_button(env.objects['button'])
        time.sleep(1)
        object_utils.pop_up_button(env.objects['button'])
        for i in range(20):
            obs, rew, done, info = env.step(
                np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0.]))
            print("reward", rew, "info", info)
            time.sleep(0.1)
        env.reset()
