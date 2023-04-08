import argparse
import json
import h5py
import psutil
import imageio
import numpy as np
import os
from copy import deepcopy
import csv
from collections import deque
import time
import random
import pickle
import pandas as pd

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.config import config_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

device = TorchUtils.get_torch_device(try_to_use_cuda=True)

import matplotlib.pyplot as plt


def generate_embeddings(args, config):
    classifier, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.classifier, device=device, verbose=True, trainable = False)
    classifier = classifier.policy  # bypass the rollout class
    classifier.set_eval()
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    expert_dataset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"],
                                                          num_training_samples=100,
                                                          weighting = True)

    expert_dataset.compute_own_embeddings(classifier)
    return expert_dataset.offline_embeddings, expert_dataset.label_list


def generate_TSNE_points(embeddings, dims):
    tsne = TSNE(dims, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)
    return tsne_proj

def generate_PCA_points(embeddings, dims):
    pca = PCA(n_components=dims)
    pca.fit(embeddings)
    print(pca.explained_variance_ratio_)
    pca_proj = pca.transform(embeddings)
    return pca_proj


def main():
    parser = argparse.ArgumentParser()

    # Path to classifier model
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )
    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--expert_dataset",
        type=str,
        default=None,
        help="The expert dataset for the model. Required if you are loading from fresh config",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="The configs",
    )

    parser.add_argument(
        "--precomputed_points",
        type=str,
        default=None,
        help="The configs",
    )


    args = parser.parse_args()

    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    config.train.data = args.expert_dataset

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    fig, ax = plt.subplots(figsize=(8,8))

    if args.precomputed_points is None:
        embeddings, labels = generate_embeddings(args, config)
        projection = generate_TSNE_points(embeddings, 2)
        with open("tsne.csv", "w", newline = "") as f:
            writer = csv.writer(f)
            for i in range(projection.shape[0]):
                writer.writerow(projection[i])
        with open("labels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(labels)):
                writer.writerow([labels[i]])

    else:
        df = pd.read_csv(args.precomputed_points + "tsne.csv")
        projection = df.to_numpy()
        df = pd.read_csv(args.precomputed_points + "labels.csv")
        labels = df.to_numpy()[:, 0]

    # ax.plot(projection[:, 0], projection[:, 1], color='green', marker='o', markersize = 3)

    indices_successful = labels == 1
    ax.scatter(projection[indices_successful, 0], projection[indices_successful, 1],color='green', s = 1)
    indices_unsuccessful = labels == 0
    ax.scatter(projection[indices_unsuccessful, 0], projection[indices_unsuccessful, 1], color='red', s = 1)
    plt.savefig("embeddings.png", dpi = 200)
    plt.savefig("embeddings.pdf")

main()
