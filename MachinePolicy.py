try:
    import robosuite
    import robosuite.utils.transform_utils as T
except:
    print("robosuite not imported. Let's hope that you're using the roboverse env")
    from roboverse.policies import policies

import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from copy import deepcopy

class MachinePolicy():
    @staticmethod
    def quat_to_rpy(quat):
        return T.mat2euler(T.quat2mat(quat))

    @staticmethod
    def unit_vector(vec):
        if np.linalg.norm(vec) < 1e-7: #if we put a zero vector, we should get a zero vector back out
            return np.zeros_like(vec)
        return vec / np.linalg.norm(vec)

    @staticmethod
    def rotate_vector(vector, rad_angle):
        assert vector.shape[0] == 2, "can't rotate 3d yet"
        rot_matrix = np.array([[np.cos(rad_angle), -np.sin(rad_angle)], [np.sin(rad_angle), np.cos(rad_angle)]])
        return np.dot(rot_matrix, vector)

    @staticmethod
    def add_radians(base, addition):
        if base + addition > math.pi :
            return (base + addition) - (2 * math.pi)
        else:
            return base + addition

    @staticmethod
    def sub_radians(base, subtraction):
        result = base - subtraction
        if result < -math.pi :
            return result + (2 * math.pi)
        return result

    @staticmethod
    def diff_radians_batch(target, current):
        delta = np.zeros_like(target)
        for i in range(target.shape[0]):
            delta[i] = MachinePolicy.diff_radians(target[i], current[i])
        return delta

    @staticmethod
    def diff_radians(target, current):
        # make radians positive for ease of computation
        target = target + (2 * math.pi)  if target < 0 else target
        current = current + (2 * math.pi)  if current < 0 else current

        diff = target - current
        # take the shortest path if it's more than 180 degrees
        if diff < -math.pi :
            return diff + (2 * math.pi)
        if diff > math.pi :
            return diff - (2 * math.pi)
        return diff

    @staticmethod
    def closest_axis(angle):
        if -(math.pi / 4) < angle < (math.pi / 4):
            return 0
        elif (math.pi / 4) < angle < ((3 / 4) * math.pi):
            return math.pi / 2
        elif ((3 / 4) * math.pi) < angle < math.pi:
            return math.pi
        elif -math.pi < angle < -((3 / 4) * math.pi):
            return -math.pi
        else:
            return -(math.pi / 2)

    def grasp_new(self, gripper_pos, gripper_vel, threshold = 0.02):
        # threshold just ensures that we don't false trigger before we even start
        accomplished = gripper_vel[1] < 0.001 and gripper_pos[0] < threshold
        return self.to_action_vec(gripper = "close"), accomplished

    def release_new(self, gripper_pos, gripper_vel, threshold = 0.02):
        # threshold just ensures that we don't false trigger before we even start
        accomplished = gripper_vel[1] < 0.001 and gripper_pos[0] > threshold
        return self.to_action_vec(gripper = "open"), accomplished

    def hold(self): #dummy action for holding still
        return self.to_action_vec(), False

    def to_action_vec(self, delta_pos = None, delta_axes = None, gripper = None):
        action = np.zeros((7,))
        if delta_pos is not None:
            action[0 : 3] = delta_pos
        if delta_axes is not None:
            action[3 : 6] = delta_axes
        if gripper is not None:
            action[6] = -1 if gripper == "open" else 1 if gripper == "close" else 0
            self.last_gripper = action[6]
        else:
            action[6] = self.last_gripper
        return action

    # takes in a raw object vector and segments it
    def object_to_states(self, ojt):
        state_list = np.split(ojt, self.sensor_indexes)
        sensor_dict = {key: value for key, value in zip(self.sensor_order, state_list)}
        return sensor_dict

    def start_episode(self):
        raise NotImplementedError

    def epsilon(self, delta):
        """
        This function outputs a scaling factor for the delta
        :param delta:
        :return: epsilon value
        """
        norm = np.linalg.norm(delta)
        epsilon = min(norm + self.epsilon_naught, 1)
        # print(norm)
        return epsilon

    def clipped_normal(self, lower, upper, shape = None):
        """
        returns a random number sampled rom a clipped normal distribution. 95% of returned numbers are not clipped
        """
        middle = (lower + upper) / 2
        scale = (upper - lower) / 4 #2 standard deviations from the bounds
        return np.clip(np.random.default_rng().normal(loc = middle, scale = scale, size = shape), lower, upper)

    def hit_target(self, eef_pos, eef_ypr, eef_vel, eef_ang, pos_margin=0.005, rot_margin=0.05, active_ang = [1, 1, 1], active_pos = [1, 1, 1]):
        delta_xyz = (self.target_xyz - eef_pos)
        delta_xyz -= 0.2 * eef_vel  # D component
        delta_xyz = delta_xyz * np.array(active_pos)
        ep_delta = self.epsilon(delta_xyz)
        delta_ypr = np.zeros_like(eef_ypr)

        ep_ypr = 1
        if self.target_ypr is not None:
            delta_ypr = self.diff_radians_batch(self.target_ypr, eef_ypr)
            delta_ypr -= 0.05 * eef_ang # D component
            delta_ypr = delta_ypr * np.array(active_ang)
            ep_ypr = self.epsilon(delta_ypr)
        accomplished = np.linalg.norm(delta_xyz) < pos_margin and np.linalg.norm(delta_ypr) < rot_margin
        return self.to_action_vec(ep_delta * self.unit_vector(delta_xyz), 0.25 * ep_ypr * self.unit_vector(delta_ypr)), accomplished

class OfficePolicy(MachinePolicy):
    def __init__(self, env, noise = 0, verbose = False, paired = False, policy_name = "pickplace_target", downstream_obj = "eraser",
                 downstream_target = "tray"): # default for backward compatibiltiy
        assert policy_name in policies.keys(), f"The policy name must be one of: {policies.keys()}"
        # noise, verbose, and paired are passthroughs that don't actually do anything.
        policy_class = policies[policy_name]
        print(policy_name)
        self.policy_name = policy_name
        self.env = env.env
        self.policy = policy_class(self.env) # backward compatabiel with older implementation.
        self.done = False
        if self.policy_name == "pickplace_target":
            self.object_target = downstream_target
            self.object_name = downstream_obj

    def start_episode(self):
        if self.policy_name == "pickplace_target":
            self.policy.reset(object_target=self.object_target, object_name=self.object_name)
        else:
            self.policy.reset()
        self.done = False

    def __call__(self, ob):
        # ob is a passthrough; it does nothing here
        if not self.done:
            action, agent_info, add_noise = self.policy.get_action()
            self.done =  agent_info['done']
            # print(self.done)
            env_action_dim = self.env.action_space.shape[0]
            if env_action_dim == action.shape[0] + 1:
                action = np.append(action, 0) #there's a dummy dimension
        else: #if the policy is done, just stay in place
            return np.zeros(shape = self.env.action_space.shape)

        return action


class SquareAssemblyPolicy(MachinePolicy):
    sensor_order = ['SquareNut_pos', 'SquareNut_quat', 'SquareNut_to_robot0_eef_pos',
                    'SquareNut_to_robot0_eef_quat']
    sensor_dims = [3, 4, 3, 4]
    STATUS_LIST = ["NUT_REACH", "NUT_PLUNGE", "NUT_GRAB", "NUT_LIFT", "NUT_DROP", "NUT_RELEASE", "HOLD"]

    def __init__(self, noise, verbose = False, paired = False, env = None): #env is passthrough
        self.status = "NUT_REACH"
        self.status_counter = 0
        self.noise = noise
        self.noise_low = 0.03
        self.sensor_indexes = np.cumsum(self.sensor_dims[:-1])
        self.last_gripper = -1

        self.epsilon_naught = 0.1 #the slowest that the robot can be

        self.target_xyz = None
        self.target_ypr = None
        self.verbose = verbose
        self.last_switch = 0
        self.paired = paired
        self.alt_behavior = False

    def start_episode(self):
        self.status = "NUT_REACH"
        self.status_counter = 0
        self.last_gripper = -1

    def set_nonsense(self, status):
        self.alt_behavior = status
        if self.alt_behavior: print("MESSING UP!")

    def switchboard(self, oracle, apprentice, intervention):
        raise Exception("Not used anymore!")
        oracle_pos_ypr_norm = np.linalg.norm(oracle[0 : 6]) #ranges from 0.35 to around 0.1
        ypr_diff_cos = np.dot(oracle, apprentice) / (np.linalg.norm(oracle) * np.linalg.norm(apprentice))

        bottleneck_region = oracle_pos_ypr_norm < 0.15 # when it slows down
        # print(oracle_pos_ypr_norm)
        # if the grippers disagree, immediately defer to the oracle
        if abs(oracle[6] - apprentice[6]) > 0.05:
            return True

        if not bottleneck_region:
            intervention_threshold = 0.82
            takeover_threshold = 0.88
        else:
            print("\tBOTTLENECK!")
            intervention_threshold = 0.97
            takeover_threshold = 0.99

        if ypr_diff_cos < intervention_threshold:
            if not intervention: print("\t\tINTERVENTION")
            self.last_switch = 0
            return True
        elif ypr_diff_cos > takeover_threshold and self.last_switch > 20:
            if intervention: print("\t\tRELINQUISH")
            return False
        else:
            return intervention  # passthrough if the thresholds are not met

    def compute_noise(self, delta):
        norm = np.linalg.norm(delta)
        noise =  0.7 * norm #empirical number
        noise = np.clip(noise, self.noise_low, self.noise)
        return noise

    def __call__(self, ob, noisy_mode = False):
        self.last_switch += 1 #basically a counter for a lockout

        ob, _ = ob #no privileged information
        object_states = self.object_to_states(ob["object"])
        eef_pos = ob["robot0_eef_pos"]
        eef_quat = ob["robot0_eef_quat"]
        eef_ypr = self.quat_to_rpy(eef_quat)
        gripper_pos = ob["robot0_gripper_qpos"]
        gripper_vel = ob["robot0_gripper_qvel"]
        lin_vel = ob["robot0_eef_vel_lin"]
        ang_vel = ob["robot0_eef_vel_ang"]

        nut_rotation = self.quat_to_rpy(object_states["SquareNut_quat"])

        if self.status == "NUT_REACH":
            self.target_xyz = object_states["SquareNut_pos"].copy()
            self.target_xyz[2] += 0.03

            # we seek the handle, but we need to compensate a rotated vector
            # target_mean = 0.06

            # target_offset = np.clip(np.random.default_rng().normal(loc = target_mean, scale = 0.002), 0.055, 0.065)

            rotated_offset = self.rotate_vector(np.array([self.clipped_normal(0.055, 0.065),
                                                          self.clipped_normal(-0.005, 0.005)]), nut_rotation[2])

            self.target_xyz[0:2] += rotated_offset #moving towards the handle

            self.target_ypr = eef_ypr.copy()
            # lazy rotation: you can match either 0 or 180 degrees
            if abs(self.diff_radians(nut_rotation[2], eef_ypr[2])) > 1.57:
                self.target_ypr[2] = self.add_radians(nut_rotation[2], math.pi )
            else:
                self.target_ypr[2] = nut_rotation[2]


            action, accomplished = self.hit_target(eef_pos, eef_ypr, lin_vel, ang_vel)
        if self.status == "NUT_PLUNGE":
            action, accomplished = self.hit_target(eef_pos, eef_ypr, lin_vel, ang_vel, active_ang = [0, 0, 1])
        if self.status == "NUT_GRAB":
            action, accomplished = self.grasp_new(gripper_pos, gripper_vel)
        if self.status == "NUT_LIFT":
            action, accomplished = self.hit_target(object_states["SquareNut_pos"], eef_ypr, lin_vel, ang_vel, active_ang = [0, 0, 1])
        if self.status == "NUT_DROP":
            action, accomplished = self.hit_target(eef_pos, eef_ypr, lin_vel, ang_vel)
        if self.status == "NUT_RELEASE":
            action, accomplished = self.release_new(gripper_pos, gripper_vel)
        if self.status == "HOLD":
            action, accomplished = self.hold()

        # noise_level = self.compute_noise(action[0 : 6])
        # action[0 : 6] += self.clipped_normal(-noise_level, noise_level, shape = (6,))

        if self.verbose:
            print(self.status)
            # print(noise_level)

        if accomplished:
            self.status_counter += 1
            self.status = self.STATUS_LIST[self.status_counter]

            if self.status == "NUT_PLUNGE":
                self.target_xyz = eef_pos.copy()
                self.target_xyz[2] = object_states["SquareNut_pos"][2] #+ 0.003
                self.target_ypr = None

            if self.status == "NUT_LIFT":
                self.target_ypr = eef_ypr.copy()

                if not self.alt_behavior:
                    mean_target_xyz = [0.23, 0.1, 1.05]
                    target_xyz = np.random.default_rng().normal(loc=mean_target_xyz, scale= [0.001, 0.001, 0.01], size=(3,))
                    self.target_xyz = np.clip(target_xyz, [0.225, 0.095, 1.03], [0.235, 0.105, 1.07])
                    self.target_ypr[2] = self.closest_axis(self.target_ypr[2])
                else:
                    mean_target_xyz = [0.23, -0.1, 1.05]
                    target_xyz = np.random.default_rng().normal(loc=mean_target_xyz, scale=[0.001, 0.001, 0.01],                                          size=(3,))
                    self.target_xyz = np.clip(target_xyz, [0.225, -0.095, 1.03], [0.235, -0.105, 1.07])




            if self.status == "NUT_DROP":
                self.target_xyz = eef_pos.copy()
                self.target_xyz[2] -= 0.1
                self.target_ypr = None

        if noisy_mode:
            noisy_action = deepcopy(action)
            noise_level = self.compute_noise(noisy_action[0 : 6])
            noisy_action[0 : 6] += self.clipped_normal(-noise_level, noise_level, shape = (6,))
            return action, noisy_action

        else:
            return action

MACHINE_DICT = {"SquarePeg": SquareAssemblyPolicy, "ToolHang": ToolHangMachinePolicy, "OfficePP" : OfficePolicy}
