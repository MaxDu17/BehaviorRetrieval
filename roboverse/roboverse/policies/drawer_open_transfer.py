import numpy as np
import roboverse.bullet as bullet


class DrawerOpenTransfer:

    def __init__(self, env, close_drawer=False, suboptimal=False):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.04
        self.ending_z = -0.25
        self.close_drawer = close_drawer  # closes instead of opens
        self.suboptimal = suboptimal
        self.reset()

    def reset(self):
        self.drawer_never_opened = True
        offset_coeff = (-1) ** (self.env.left_opening - 1 + self.close_drawer)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        done = False
        neutral_action = [0.0]
        if (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_drawer_open()):
            # print('xy - approaching handle')
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        elif (gripper_handle_dist > self.gripper_dist_thresh
                and not self.env.is_drawer_open()):
            # moving down toward handle
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.env.is_drawer_open():
            x_command = (-1) ** (self.env.left_opening - 1 + self.close_drawer)
            if self.suboptimal:
                # randomly decide whether to open or close the drawer
                if np.random.uniform() > 0.5:
                    x_command *= -1.0
            action_xyz = np.array([x_command, 0, 0])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif ee_pos[2] < self.ending_z:
            action_xyz = [0., 0., 0.5]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        else:
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.7]
            done = True

        agent_info = dict(done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class DrawerOpenTransferSuboptimal(DrawerOpenTransfer):

    def __init__(self, env):
        super(DrawerOpenTransferSuboptimal, self).__init__(
            env, suboptimal=True)


class DrawerCloseTransfer(DrawerOpenTransfer):

    def __init__(self, env):
        super(DrawerCloseTransfer, self).__init__(
            env, close_drawer=True)
