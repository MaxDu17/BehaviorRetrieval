import pybullet as p
import roboverse.bullet as bullet
import roboverse.bullet.control as control
import numpy as np


def pop_up_button(button):
    return slide_button(button, 1)


def push_down_button(button):
    return slide_button(button, -1)


def get_button_cylinder_pos(button):
    button_cylinder_pos, _ = bullet.get_link_state(
        button, get_button_cylinder_joint(button))
    return np.array(button_cylinder_pos)


def get_button_cylinder_joint(button):
    joint_names = [control.get_joint_info(button, j, 'jointName')
                   for j in range(p.getNumJoints(button))]
    button_cylinder_joint_idx = joint_names.index('base_button_joint')
    return button_cylinder_joint_idx


def slide_button(button, direction):
    assert direction in [-1, 1]
    # -1 = push down; 1 = pop up
    button_cylinder_joint_idx = get_button_cylinder_joint(button)
    num_ts = 20
    command = 10 * direction

    p.setJointMotorControl2(
        button,
        button_cylinder_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=command,
        force=10
    )

    control.step_simulation(num_ts)

    button_pos = get_button_cylinder_pos(button)

    p.setJointMotorControl2(
        button,
        button_cylinder_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=10
    )

    return button_pos
