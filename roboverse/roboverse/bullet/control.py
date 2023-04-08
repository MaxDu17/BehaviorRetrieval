import pybullet as p
import numpy as np


def get_joint_states(body_id, joint_indices):
    all_joint_states = p.getJointStates(body_id, joint_indices)
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities)


def get_movable_joints(body_id):
    num_joints = p.getNumJoints(body_id)
    movable_joints = []
    for i in range(num_joints):
        if p.getJointInfo(body_id, i)[2] != p.JOINT_FIXED:
            movable_joints.append(i)
    return movable_joints


def get_link_state(body_id, link_index):
    position, orientation, _, _, _, _ = p.getLinkState(body_id, link_index)
    return np.asarray(position), np.asarray(orientation)

def set_link_state(body_id, link_index, position):
    joint_pos, _, _, _ = p.getJointState(body_id, link_index)
    COM_pos, _ = get_link_state(body_id, link_index)

    abs_diff = np.linalg.norm(COM_pos - position)
    diff = position[0] - COM_pos[0]
    print(position, COM_pos, diff)
    # print(joint_pos, diff)

    p.resetJointState(body_id, link_index, joint_pos + diff)
    return get_link_state(body_id, link_index)[0:2] #to sanity check

def get_joint_info(body_id, joint_id, key):
    keys = ["jointIndex", "jointName", "jointType", "qIndex", "uIndex",
            "flags", "jointDamping", "jointFriction", "jointLowerLimit",
            "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName",
            "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"]
    value = p.getJointInfo(body_id, joint_id)[keys.index(key)]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return value


def apply_action_ik(target_ee_pos, target_ee_quat, target_gripper_state,
                    robot_id, end_effector_index, movable_joints,
                    lower_limit, upper_limit, rest_pose, joint_range,
                    num_sim_steps=5):
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               lowerLimits=lower_limit,
                                               upperLimits=upper_limit,
                                               jointRanges=joint_range,
                                               restPoses=rest_pose,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.01)

    p.setJointMotorControlArray(robot_id,
                                movable_joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                # targetVelocity=0,
                                forces=[5] * len(movable_joints),
                                positionGains=[0.03] * len(movable_joints),
                                # velocityGain=1
                                )
    # set gripper action
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.03)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.03)

    for _ in range(num_sim_steps):
        p.stepSimulation()


def reset_robot(robot_id, reset_joint_indices, reset_joint_values):
    assert len(reset_joint_indices) == len(reset_joint_values)
    for i, value in zip(reset_joint_indices, reset_joint_values):
        p.resetJointState(robot_id, i, value)


def move_to_neutral(robot_id, reset_joint_indices, reset_joint_values,
                    num_sim_steps=75):
    assert len(reset_joint_indices) == len(reset_joint_values)
    p.setJointMotorControlArray(robot_id,
                                reset_joint_indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=reset_joint_values,
                                forces=[100] * len(reset_joint_indices),
                                positionGains=[0.03] * len(reset_joint_indices),
                                )
    for _ in range(num_sim_steps):
        p.stepSimulation()


def reset_object(body_id, position, orientation):
    p.resetBasePositionAndOrientation(body_id,
                                      position,
                                      orientation)


def get_object_position(body_id):
    object_position, object_orientation = \
        p.getBasePositionAndOrientation(body_id)
    return np.asarray(object_position), np.asarray(object_orientation)


def step_simulation(num_sim_steps):
    for _ in range(num_sim_steps):
        p.stepSimulation()


def quat_to_deg(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg


def deg_to_quat(deg):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad)
    return quat


def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])
