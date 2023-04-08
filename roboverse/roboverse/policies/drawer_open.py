import numpy as np
import roboverse.bullet as bullet


class DrawerOpen:

    def __init__(self, env, xyz_action_scale=7.0, angle_action_scale = 0.1,
                    return_origin_thresh=0.1):
        self.env = env
        self.xyz_action_scale = xyz_action_scale
        self.angle_action_scale = angle_action_scale
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.03
        self.ending_height_thresh = 0.2
        self.return_base_thresh = 0.4
        self.open_angle = [90.0, 0.0, 0.0]
        self.done = False

        self.return_origin_thresh = return_origin_thresh
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.drawer_never_opened = True
        self.done = False
        offset_coeff = (-1) ** (1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.0]) #-0.01

    def get_action(self):
        ee_pos, ee_orientation = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        ee_deg = bullet.quat_to_deg(ee_orientation)

        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        done = False
        noise = True
        # print(f"gripper_handle_xy_dist: {gripper_handle_xy_dist}")
        if (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_drawer_open()):
            # print('xy - approaching handle')
            action_xyz = (handle_pos - ee_pos) * self.xyz_action_scale
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            # action_angles = [0., 0., 0.]
            action_angles = (self.open_angle - ee_deg) * self.angle_action_scale
            action_gripper = [0.0]
        elif (gripper_handle_dist > self.gripper_dist_thresh
                and not self.env.is_drawer_open()):
            # print("moving down toward handle")
            noise = False
            action_xyz = (handle_pos - ee_pos) * self.xyz_action_scale
            # action_angles = [0., 0., 0.]
            action_angles = (self.open_angle - ee_deg) * self.angle_action_scale
            action_gripper = [0.0]
        elif not self.env.is_drawer_open():
            # print("opening drawer")
            noise = False
            x_command = (-1) ** (1 - self.env.left_opening)
            action_xyz = np.array([x_command, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            # action_angles = (self.open_angle - ee_deg) * self.angle_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (np.abs(ee_pos[2] - self.ending_height_thresh) >
                self.gripper_dist_thresh):
            
            # print("return")
            if (np.abs(ee_pos[2] - self.ending_height_thresh) < self.return_base_thresh):
                action_xyz = [0., 0., 0.]
                action_angles = [0., 0., 0.]
                action_gripper = [0.]
                done = True
                self.done = True
            else:
                self.drawer_never_opened = False
                action_xyz = np.array([0, 0, 0.7])  # force upward action
                # action_angles = [0., 0., 0.]
                noise = False
                action_angles = (self.open_angle - ee_deg) * self.angle_action_scale
                action_gripper = [0.0]
        else:
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        
        # if done:
        #     if np.linalg.norm(ee_pos - self.env.ee_pos_init) < self.return_origin_thresh:
        #         self.done = done
        #     else:
        #         action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
        #         # print(ee_pos, self.env.ee_pos_init)
        #         # print(np.linalg.norm(ee_pos - self.env.ee_pos_init)) 
        
        agent_info = dict(done=self.done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info, noise

class DrawerClose:

    def __init__(self, env, xyz_action_scale=3.0, return_origin_thresh=0.1,angle_action_scale = 0.1):
        self.env = env
        self.xyz_action_scale = xyz_action_scale
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.03
        self.ending_z = -0.25
        self.top_drawer_offset = np.array([0, 0, 0.02])
        self.push_angle = [90.0, 5.0, 0.0]
        self.done = False
        self.begin_closing = False
        self.angle_action_scale = angle_action_scale
        self.return_origin_thresh = return_origin_thresh
        self.reset()

    def reset(self):
        self.drawer_never_opened = True
        offset_coeff = (-1) ** (1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])
        self.reached_pushing_region = False
        self.neutral_taken = False
        self.begin_closing = False
        self.done = False


    def get_action(self):
        ee_pos, ee_orientation = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        ee_deg = bullet.quat_to_deg(ee_orientation)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        drawer_pos = self.env.get_drawer_pos("drawer")
        # drawer_push_target_pos = (
        #     drawer_pos + np.array([0.15, 0., 0.05]))
        # print(f"handle_pos: {handle_pos}, drawer_pos: {drawer_pos} ")
        drawer_push_target_pos = (
            self.env.get_drawer_handle_pos() + np.array([0.1, 0.0, 0.12]))
        
        is_gripper_ready_to_push = (
            ee_pos[0] > drawer_push_target_pos[0] - 0.05 and
            np.linalg.norm(ee_pos[1] - drawer_push_target_pos[1]) < 0.1 and
            ee_pos[2] < drawer_push_target_pos[2] + 0.05
        )
        # import pdb;pdb.set_trace()
        done = False
        neutral_action = [0.0]
        noise = False
        # print(f"ee_pos{ee_pos}, drawer_push_target_pos: {drawer_push_target_pos}")
        if (not self.env.is_drawer_closed() and
                not self.reached_pushing_region and
                not is_gripper_ready_to_push):
            action_xyz = (drawer_push_target_pos  - ee_pos) * self.xyz_action_scale
            # print(f"move to pushing region, action: {action_xyz}")
            # import pdb; pdb.set_trace()
            # action_xyz = [0., -0.1, -0.15] #-0.15

            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not self.env.is_drawer_closed():
            # print("close top drawer")
            self.reached_pushing_region = True
            action_xyz = [0,0,0]
            action_xyz[0] = (drawer_pos  - ee_pos)[0] * 3
            action_xyz[2] = (drawer_pos  - ee_pos)[2] * 3
            # action_angles = (self.push_angle - ee_deg) * 0.5
            # print(f"ee_deg: {ee_deg}")
            # action_xyz[0] *= 3
            # action_xyz[0] *= 1
            # action_xyz[1] *= 0.6
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7] #0.
            self.begin_closing = True
            # import pdb; pdb.set_trace()
        if self.env.is_drawer_closed() and self.begin_closing:
            # print("closed")
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            done = True
        
        if done:
            if np.linalg.norm(ee_pos - self.env.ee_pos_init) < self.return_origin_thresh:
                self.done = done
            else:
                action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
                # print(ee_pos, self.env.ee_pos_init)
                # print(np.linalg.norm(ee_pos - self.env.ee_pos_init)) 
        # print(ee_pos, drawer_push_target_pos)
        # import pdb; pdb.set_trace()
        agent_info = dict(done=self.done)
        action = np.concatenate((action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info, noise

