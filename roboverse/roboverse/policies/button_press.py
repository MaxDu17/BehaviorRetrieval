import numpy as np
import roboverse.bullet as bullet


class ButtonPress:

    def __init__(self, env):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.02
        self.ending_height_thresh = 0.2
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.button_offset = np.array([0.01, 0.0, -0.01])
        self.gripper_closed = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        button_pos = self.env.get_button_pos() + self.button_offset
        gripper_button_xy_dist = np.linalg.norm(button_pos[:2] - ee_pos[:2])
        done = False

        if (gripper_button_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_button_pressed()):
            # print('xy - approaching handle')
            action_xyz = (button_pos - ee_pos) * self.xyz_action_scale
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.gripper_closed:
            # print("close gripper")
            action_xyz = np.array([0, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]  # close gripper
            self.gripper_closed = True
        elif not self.env.is_button_pressed():
            # print("opening drawer")
            action_xyz = (button_pos - ee_pos) * self.xyz_action_scale
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]  # close gripper
        elif (np.abs(ee_pos[2] - self.ending_height_thresh) >
                self.gripper_dist_thresh):
            # print("Lift upward")
            action_xyz = np.array([0, 0, 0.7])  # force upward action
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        else:
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        agent_info = dict(done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info
