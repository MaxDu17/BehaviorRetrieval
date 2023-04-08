import pybullet as p
import numpy as np

def render(height, width, view_matrix, projection_matrix,
           shadow=1, light_direction=[1, 1, 1],
           renderer=p.ER_BULLET_HARDWARE_OPENGL):
    #  ER_BULLET_HARDWARE_OPENGL
    img_tuple = p.getCameraImage(width,
                                 height,
                                 view_matrix,
                                 projection_matrix,
                                 shadow=shadow,
                                 lightDirection=light_direction,
                                 renderer=renderer)
    _, _, img, depth, segmentation = img_tuple
    # import ipdb; ipdb.set_trace()
    # Here, if I do len(img), I get 9216.
    # img = np.reshape(np.array(img), (48, 48, 4))
    img_res = int(np.sqrt(np.array(img).size / 4))
    img = np.reshape(np.array(img), (img_res, img_res, 4))
    
    img = img[:, :, :-1]
    return img, depth, segmentation


def get_view_matrix(target_pos=(.75, -.2, 0), distance=0.9,
                    yaw=180, pitch=-20, roll=0, up_axis_index=2):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        target_pos, distance, yaw, pitch, roll, up_axis_index)
    return view_matrix


def get_projection_matrix(height, width, fov=60, near_plane=0.1, far_plane=2):
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane,
                                                     far_plane)
    return projection_matrix