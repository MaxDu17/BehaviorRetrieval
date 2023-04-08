import roboverse
import roboverse.bullet as bullet
from roboverse.policies import policies
from PIL import Image

import skvideo.io
import cv2
import os
import argparse
import numpy as np
import torch

ROBOT_VIEW_HEIGHT = 100
ROBOT_VIEW_WIDTH = 100
ROBOT_VIEW_CROP_X = 0


def reshape_image_obs(image_obs):
    # TODO remove hard coding
    image_obs = np.reshape(image_obs, (3, 48, 48))
    image_obs = np.transpose(image_obs, (1, 2, 0))
    image_obs = np.uint8(image_obs*255.0)
    return image_obs


class BulletVideoLogger:
    def __init__(self, env_name, checkpoint_filename,
                 num_timesteps_per_traj, accept_trajectory_key,
                 video_save_dir, save_all, save_images,
                 add_robot_view, noise=0.1):
        self.env_name = env_name
        self.num_timesteps_per_traj = num_timesteps_per_traj
        self.accept_trajectory_key = accept_trajectory_key
        self.noise = noise
        chkpt_split = checkpoint_filename.split('/')
        logdir = chkpt_split[-2] + '-' + chkpt_split[-1].split('.')[0]
        self.video_save_dir = os.path.join(video_save_dir, logdir)
        self.save_all = save_all
        if save_images:
            self.save_function = self.save_images
        else:
            self.save_function = self.save_video

        self.image_size = 512
        self.add_robot_view = add_robot_view

        if not os.path.exists(self.video_save_dir):
            os.makedirs(self.video_save_dir)

        # cam for pick and place (front view)
        # self.camera_target_pos = [0.55, 0.1, -0.30]
        # self.camera_roll = 0.0
        # self.camera_pitch = -30.0
        # self.camera_yaw = 180.0
        # self.camera_distance = 0.50

        # drawer cam (front view)
        # self.camera_target_pos = [0.60, 0.05, -0.30]
        # self.camera_roll = 0.0
        # self.camera_pitch = -30.0
        # self.camera_yaw = 180.0
        # self.camera_distance = 0.50

        # drawer cam (canonical view)
        self.camera_target_pos = [0.55, 0., -0.30]
        self.camera_roll = 0.0
        self.camera_pitch = -30.0
        self.camera_yaw = 150.0
        self.camera_distance = 0.64

        self.view_matrix_args = dict(target_pos=self.camera_target_pos,
                                     distance=self.camera_distance,
                                     yaw=self.camera_yaw,
                                     pitch=self.camera_pitch,
                                     roll=self.camera_roll,
                                     up_axis_index=2)
        self.view_matrix = bullet.get_view_matrix(
            **self.view_matrix_args)
        self.projection_matrix = bullet.get_projection_matrix(
            self.image_size, self.image_size)
        # end camera settings
        self.env = roboverse.make(self.env_name, gui=False,
                                  transpose_image=True)

        checkpoint = torch.load(checkpoint_filename)
        self.policy = checkpoint['evaluation/policy'].cpu()
        self.checkpoint_name = 'test123'
        self.trajectories_collected = 0

    def add_robot_view_to_video(self, images, robot_images):
        image_x, image_y, image_c = images[0].shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(images)):
            robot_view_margin = 5
            robot_view = cv2.resize(robot_images[i],
                                    (ROBOT_VIEW_HEIGHT, ROBOT_VIEW_WIDTH))
            robot_view = robot_view[ROBOT_VIEW_CROP_X:, :, :]
            image_new = np.copy(images[i])
            x_offset = ROBOT_VIEW_HEIGHT - ROBOT_VIEW_CROP_X
            y_offset = image_y - ROBOT_VIEW_WIDTH

            # Draw a background black rectangle
            image_new = cv2.rectangle(image_new, (self.image_size, 0),
                                      (y_offset - 2 * robot_view_margin,
                                       x_offset + 25 + robot_view_margin),
                                      (0, 0, 0), -1)

            image_new[robot_view_margin:x_offset + robot_view_margin,
            y_offset - robot_view_margin:-robot_view_margin,
            :] = robot_view
            image_new = cv2.putText(image_new, 'Robot View',
                                    (y_offset - robot_view_margin,
                                     x_offset + 18 + robot_view_margin),
                                    font, 0.55, (255, 255, 255), 1,
                                    cv2.LINE_AA)
            images[i] = image_new

        return images

    def collect_traj_and_save_video(self, path_idx):
        images = []
        robot_images = []
        obs = self.env.reset()
        # self.scripted_policy.reset()
        for t in range(self.num_timesteps_per_traj):
            img, depth, segmentation = bullet.render(
                self.image_size, self.image_size,
                self.view_matrix, self.projection_matrix)
            images.append(img)
            robot_images.append(reshape_image_obs(obs['image']))
            # action, _ = self.scripted_policy.get_action()
            action, _ = self.policy.get_action(obs['image'])
            obs, rew, done, info = self.env.step(action)

        if self.save_all:
            self.save_function(images, robot_images, path_idx)
        elif info[self.accept_trajectory_key]:
            self.save_function(images, robot_images, path_idx)
        else:
            return False

        return True

    def save_video(self, images, robot_images, path_idx):
        # Save Video
        save_path = "{}/{}_{}.mp4".format(
            self.video_save_dir, self.env_name,
            path_idx)
        if self.add_robot_view:
            dot_idx = save_path.index(".")
            save_path = save_path[:dot_idx] + "_with_robot_view" + \
                        save_path[dot_idx:]
        inputdict = {'-r': str(12)}
        outputdict = {'-vcodec': 'libx264', '-pix_fmt': 'yuv420p'}
        writer = skvideo.io.FFmpegWriter(
            save_path, inputdict=inputdict, outputdict=outputdict)

        if self.add_robot_view:
            self.add_robot_view_to_video(images, robot_images)
        for i in range(len(images)):
            writer.writeFrame(images[i])
        writer.close()

    def save_images(self, images, robot_images, path_idx):
        # Save Video
        save_path = "{}/{}_{}".format(
            self.video_save_dir, self.env_name,
            path_idx)
        if self.add_robot_view:
            save_path += "_with_robot_view"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.add_robot_view:
            self.add_robot_view_to_video(images, robot_images)
        for i in range(len(images)):
            im = Image.fromarray(images[i])
            im.save(os.path.join(save_path, '{}.png'.format(i)))

    def run(self, num_videos):
        i = 0
        while i < num_videos:
            saved = self.collect_traj_and_save_video(i)
            if saved:
                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, required=True)
    parser.add_argument("-ch", "--checkpoint", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str,
                        required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-d", "--video-save-dir", type=str,
                        default="data/video_rollouts")
    parser.add_argument("-n", "--num-videos", type=int, default=3)
    parser.add_argument("--add-robot-view", action="store_true", default=False)
    parser.add_argument("--save-images", action="store_true", default=False)
    parser.add_argument("--save-all", action="store_true", default=False)

    args = parser.parse_args()

    vid_log = BulletVideoLogger(
        args.env, args.checkpoint, args.num_timesteps,
        args.accept_trajectory_key, args.video_save_dir, args.save_all,
        args.save_images, args.add_robot_view)

    vid_log.run(args.num_videos)
