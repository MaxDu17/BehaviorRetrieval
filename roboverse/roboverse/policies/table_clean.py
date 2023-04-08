import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS
from .drawer_open_transfer import DrawerOpenTransfer
from .pick_place import PickPlace, PickPlaceTarget
from .drawer_open import DrawerOpen, DrawerClose

class TableClean:
    def __init__(self, env, pick_height_thresh=-0.31, xyz_action_scale=5.0,
                 pick_point_noise=0.00, drop_point_noise=0.00, return_origin_thresh=0.2, return_origin_thresh_drawer=0.1):
        self.env = env
        self.done = False
        self.object_names= env.task_object_names
        self.object_names = env.object_names

        self.num_objects = env.num_objects
        self.xyz_action_scale = self.env.xyz_action_scale
        self.pick_policies = []
        self.pick_drawer_policies = []

        self.pick_height_thresh = pick_height_thresh
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.drop_point_noise = drop_point_noise
        self.return_origin_thresh_drawer = return_origin_thresh_drawer
        self.return_origin_thresh = return_origin_thresh

        # for object_name, object_target in zip(self.object_names, self.object_targets):
        #     print(object_name, object_target)
        #     if object_target == 'tray':
        #         if not self.env.load_tray:
        #             raise NotImplementedError
        #
        #     if object_target == 'drawer_inside':
        #         pick_drawer_policy = {}
        #         pick_drawer_policy['pick_policy'] = PickPlaceTarget(env,
        #                 pick_height_thresh=pick_height_thresh,
        #                 xyz_action_scale=xyz_action_scale,
        #                 pick_point_noise=pick_point_noise,
        #                 drop_point_noise=drop_point_noise,
        #                 object_name=object_name,
        #                 object_target=object_target,
        #                 return_origin_thresh=return_origin_thresh_drawer
        #         )
        #         pick_drawer_policy['drawer_open_policy'] = DrawerOpen(env,
        #                 xyz_action_scale=xyz_action_scale,
        #                 return_origin_thresh=return_origin_thresh)
        #         pick_drawer_policy['drawer_close_policy'] = DrawerClose(env,
        #                 xyz_action_scale=xyz_action_scale,
        #                 return_origin_thresh=return_origin_thresh)
        #         self.pick_drawer_policies.append(pick_drawer_policy)
        #         # pick_drawer_policy['drawer_close_policy']
        #     else:
        #         pick_policy = PickPlaceTarget(env,
        #                 pick_height_thresh=pick_height_thresh,
        #                 xyz_action_scale=xyz_action_scale,
        #                 pick_point_noise=pick_point_noise,
        #                 drop_point_noise=drop_point_noise,
        #                 object_name=object_name,
        #                 object_target=object_target,
        #                 return_origin_thresh=return_origin_thresh,
        #         )
        #         self.pick_policies.append(pick_policy)
        self.reset()

    def reset(self):
        self.object_names= self.env.task_object_names
        self.object_targets = self.env.object_targets
        self.pick_policies.clear()
        self.pick_drawer_policies.clear()
        print(self.object_names, self.object_targets)
        num_drawer_target = 0
        num_container_target = 0
        for object_name, object_target in zip(self.object_names, self.object_targets):
            print(object_name, object_target)
            if object_target == 'drawer_inside':
                pick_drawer_policy = {}
                pick_drawer_policy['pick_policy'] = PickPlaceTarget(self.env,
                                                                    pick_height_thresh=self.pick_height_thresh,
                                                                    xyz_action_scale=self.xyz_action_scale,
                                                                    pick_point_noise=self.pick_point_noise,
                                                                    drop_point_noise=self.drop_point_noise,
                                                                    object_name=object_name,
                                                                    object_target=object_target,
                                                                    return_origin_thresh=self.return_origin_thresh_drawer
                                                                    )
                pick_drawer_policy['drawer_open_policy'] = DrawerOpen(self.env,
                                                                      xyz_action_scale=self.xyz_action_scale,
                                                                      return_origin_thresh=self.return_origin_thresh)
                pick_drawer_policy['drawer_close_policy'] = DrawerClose(self.env,
                                                                        xyz_action_scale=self.xyz_action_scale,
                                                                        return_origin_thresh=self.return_origin_thresh)
                self.pick_drawer_policies.append(pick_drawer_policy)



                pick_drawer_policy = self.pick_drawer_policies[num_drawer_target]
                pick_drawer_policy['pick_policy'].reset(object_name=object_name, object_target=object_target)
                pick_drawer_policy['drawer_open_policy'].reset()
                pick_drawer_policy['drawer_close_policy'].reset()
                self.pick_drawer_policies.append(pick_drawer_policy)
                # pick_drawer_policy['drawer_close_policy']
                num_drawer_target += 1
            else:
                pick_policy = PickPlaceTarget(self.env,
                                              pick_height_thresh=self.pick_height_thresh,
                                              xyz_action_scale=self.xyz_action_scale,
                                              pick_point_noise=self.pick_point_noise,
                                              drop_point_noise=self.drop_point_noise,
                                              object_name=object_name,
                                              object_target=object_target,
                                              return_origin_thresh=self.return_origin_thresh,
                                              )
                self.pick_policies.append(pick_policy)
                self.pick_policies[num_container_target].reset(object_name=object_name, object_target=object_target)
                num_container_target += 1

        self.done = False

    # def get_action(self):
    #     all_pick_done = True
    #     for i in range(self.num_objects):
    #         if self.pick_policies[i].done:
    #             continue
    #         else:
    #             action, agent_info = self.pick_policies[i].get_action()
    #             agent_info['done'] = self.done
    #             all_pick_done = False
    #             print(f"current policy: {self.pick_policies[i].object_name}, is agent done? {agent_info['done']}")

    #             return action, agent_info

    #     if all_pick_done:
    #         return self.drawer_policy.get_action()

    def get_action(self):
        for pick_policy in self.pick_policies:
            if pick_policy.done:
                continue
            else:
                action, agent_info, noise = pick_policy.get_action()
                agent_info['done'] = self.done
                # print(f"current policy: {pick_policy.object_name}, is agent done? {agent_info['done']}")
                return action, agent_info, noise

        if len(self.pick_drawer_policies) == 0:
            action, agent_info, noise = self.pick_policies[-1].get_action()
            agent_info['done'] = True
            # print(f"current policy: {pick_policy.object_name}, is agent done? {agent_info['done']}")
            return action, agent_info, noise

        for pick_drawer_policy in self.pick_drawer_policies:
            pick_policy = pick_drawer_policy['pick_policy']
            drawer_open_policy = pick_drawer_policy['drawer_open_policy']
            drawer_close_policy = pick_drawer_policy['drawer_close_policy']
            if drawer_open_policy.done and pick_policy.done and drawer_close_policy.done:
                continue
            else:
                if not drawer_open_policy.done:
                    action, agent_info, noise = drawer_open_policy.get_action()
                    agent_info['done'] = self.done
                else:
                    if not pick_policy.done:
                        action, agent_info, noise = pick_policy.get_action()
                        agent_info['done'] = self.done
                    else:
                        action, agent_info, noise = drawer_close_policy.get_action()
                        agent_info['done'] = self.done

                        all_done = True
                        for pick_drawer_policy in self.pick_drawer_policies:
                            pick_policy = pick_drawer_policy['pick_policy']
                            drawer_open_policy = pick_drawer_policy['drawer_open_policy']
                            drawer_close_policy = pick_drawer_policy['drawer_close_policy']
                            if drawer_open_policy.done and pick_policy.done and drawer_close_policy.done:
                                continue
                            else:
                                all_done = False

                        if all_done:
                            self.done = True
                            agent_info['done'] = True

                return action, agent_info, noise






