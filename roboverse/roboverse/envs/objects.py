import pybullet_data
import pybullet as p
import os
import roboverse.bullet as bullet
import numpy as np

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')
SHAPENET_SCALE = 0.5
"""
NOTE: Use this file only for core objects, add others to bullet/object_utils.py
This file will likely be deprecated in the future.
"""

def tray(base_position=(.60, 0.3, -.37), scale=0.5):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    tray_path = os.path.join(ASSET_PATH,
                            'bullet-objects/tray/tray.urdf')
    tray_id = p.loadURDF(
                         tray_path,
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale
                         )
    return tray_id


def widow250(basePosition=[0.6, -0.0, -0.4], baseOrientation=[0.0, 0.0, -180]):
    baseOrientation_quad = bullet.deg_to_quat(baseOrientation)
    widow250_path = os.path.join(ASSET_PATH,
                                 'interbotix_descriptions/urdf/wx250s.urdf')
    widow250_id = p.loadURDF(widow250_path,
                             basePosition=basePosition,
                             baseOrientation=baseOrientation_quad
                             )
    return widow250_id

def laptop(basePosition=(1.2, 0.2, -0.38), scale=1.0):
    laptop_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/laptop/laptop.urdf')
    laptop_id = p.loadURDF(laptop_path,
                             basePosition=basePosition,
                             baseOrientation=bullet.deg_to_quat([90.0, 0., 180]),
                             globalScaling=scale,
                             )
    return laptop_id


def lamp(basePosition=(0.07, -0.37, -0.4), scale=0.6):
    lamp_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/lamp/lamp.urdf')
    lamp_id = p.loadURDF(lamp_path,
                             basePosition=basePosition,
                             baseOrientation=bullet.deg_to_quat([0.0, 0., 150]),
                             globalScaling=scale,
                             )
    return lamp_id

def room(basePosition=(1.72, 1.15,-0.8), scale=0.6):
    room_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/urdf/room.urdf')
    room_id = p.loadURDF(room_path,
                        globalScaling=scale,
                         basePosition=basePosition,
                         baseOrientation=bullet.deg_to_quat([0, 0., -90]),
                            )
    return room_id


def keyboard(basePosition=(1.0, -0.55, -0.4), scale=1.2):
    keyboard_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/keyboard/keyboard.urdf')
    keyboard_id = p.loadURDF(keyboard_path,
                        globalScaling=scale,
                         basePosition=basePosition,
                         baseOrientation=bullet.deg_to_quat([0, 0., 0]),
                            )
    return keyboard_id


def desktop(basePosition=(0.65, -0.25, -1.09), scale=0.018):
    keyboard_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/desktop/desktop.urdf')
    keyboard_id = p.loadURDF(keyboard_path,
                        globalScaling=scale,
                         basePosition=basePosition,
                         baseOrientation=bullet.deg_to_quat([0, 0., 0]),
                            )
    return keyboard_id

# def trashcan(basePosition=(0.8, 0.4, -0.8), scale=0.6):
#     room_path = os.path.join(ASSET_PATH,
#                                  'bullet-objects/trashcan/trashcan.urdf')
#     room_id = p.loadURDF(room_path,
#                         globalScaling=scale,
#                          basePosition=basePosition,
#                          baseOrientation=bullet.deg_to_quat([0, 0., 0]),
#                             )
#     return room_id

def monitor(basePosition=(0.42, -0.4, -0.42), scale=25):
    monitor_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/monitor/monitor.urdf')
    monitor_id = p.loadURDF(monitor_path,
                             basePosition=basePosition,
                             baseOrientation=bullet.deg_to_quat([0, 0., 180]),
                             globalScaling=scale
                             )
    return monitor_id

def officedesk(basePosition=(0.65, 0.05, -0.46), scale=2.4):
    officedesk_path = os.path.join(ASSET_PATH,
                                 'room_descriptions/officedesk/officedesk.urdf')
    officedesk_id = p.loadURDF(officedesk_path,
                             basePosition=basePosition,
                             baseOrientation=bullet.deg_to_quat([90, 0., -90]),
                             globalScaling=scale,
                             )
    return officedesk_id

