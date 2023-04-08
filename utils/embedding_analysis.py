import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
import torch
import argparse

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


def main(args):
    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    config.train.data = args.paired_data #paired_data

    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=args.paired_data, #paired_data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    paired_dataset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"],
                                                          weighting = True) #, num_training_samples = 10)

    config.train.data = args.good_data
    good_dataset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"],
                                                          weighting = True) #, num_training_samples = 10)

    classifier, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.classifier_path, device=device, verbose=True,
                                                     trainable=False)
    classifier = classifier.policy
    classifier.set_eval()



    paired_dataset.compute_own_embeddings(classifier)
    good_dataset.compute_own_embeddings(classifier)

    # try different algorithms here
    paired_dataset.reweight_data_from_dataset(good_dataset, THRESHOLD = 0, soft = True) #, epsilon = 0.01)

    weight_list = paired_dataset.get_weight_list()
    top_k_index = np.argpartition(weight_list, -20)[-20:]
    bottom_k_index = np.argpartition(-weight_list, -20)[-20:]
    np.random.shuffle(top_k_index)
    np.random.shuffle(bottom_k_index)
    top_k_index = top_k_index[-3:]
    bottom_k_index = bottom_k_index[-3:]
    img_list = list()
    for i, index in enumerate(top_k_index):
        print(index, weight_list[index])
        data = paired_dataset.get_item(index)
        try:
            image = data["obs"]["agentview_image"][-1].transpose((1, 2, 0))
            img_list.append(image)
            # plt.imsave(f"top_k/{args.modifier}_{i}.png", image)
        except:
            pass #don't render for images

    print("---------")
    for i, index in enumerate(bottom_k_index):
        print(index, weight_list[index])
        data = paired_dataset.get_item(index)
        try:
            image = data["obs"]["agentview_image"][-1].transpose((1, 2, 0))
            img_list.append(image)
            # plt.imsave(f"top_k/{args.modifier}_{i}.png", image)
        except:
            pass #don't render for images

    if len(img_list) > 0:
        # final_image = np.zeros((img_list[0].shape[0] * 2,  img_list[0].shape[1] * 3, 3))
        row_list = list()
        for i in range(2):
            print(i * 3, (i + 1) * 3)
            row = np.concatenate(img_list[i * 3 : (i + 1) * 3], axis = 1)
            row_list.append(row)
        final_img = np.concatenate(row_list, axis = 0)
        plt.imsave(f"{args.modifier}_MOSIAC.png", final_img)

    positive_negative_keys = paired_dataset.label_list
    good_weights = weight_list[np.where(positive_negative_keys == 1)]
    bad_weights = weight_list[np.where(positive_negative_keys == 0)]

    traj_weight_list = paired_dataset.get_traj_weights()
    traj_keys = paired_dataset.traj_label_list
    pos_weight_list = [traj_weight_list[k] for k in range(len(traj_weight_list)) if traj_keys[k] == 1]
    neg_weight_list = [traj_weight_list[k] for k in range(len(traj_weight_list)) if traj_keys[k] == 0]
    #janky, but essentially takes the ragged average across the trajecotries
    pos_combined_longest = np.array(list(zip_longest(*pos_weight_list)), dtype = np.float32)
    pos_combined_longest[np.where(pos_combined_longest == None)] = np.nan
    pos_weight_mean = np.nanmean(pos_combined_longest, axis = 1)
    pos_weight_std = np.nanstd(pos_combined_longest, axis = 1)
    # pos_weight_max = np.nanmax(pos_combined_longest, axis = 1)

    neg_combined_longest = np.array(list(zip_longest(*neg_weight_list)), dtype = np.float32)
    neg_combined_longest[np.where(neg_combined_longest == None)] = np.nan
    neg_weight_mean = np.nanmean(neg_combined_longest, axis = 1)
    neg_weight_std = np.nanstd(neg_combined_longest, axis = 1)
    # neg_weight_max = np.nanmax(neg_combined_longest, axis = 1)


    #plot distribution, not mean. Plot distribution by using maximum and minimum

    fig2, ax2 = plt.subplots()
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    if args.limit is not None:
        ax2.set_xlim(8, args.limit)
    # ax2.set_ylim(0.3, 1)
    # ax2.set_title()
    ax2.plot(pos_weight_mean, color = "green")
    ax2.plot(neg_weight_mean, color = "red")

    plt.fill_between(np.arange(neg_weight_mean.shape[0]), neg_weight_mean - neg_weight_std, neg_weight_mean + neg_weight_std, alpha=0.2, edgecolor="red", facecolor="red",
        antialiased=True)

    plt.fill_between(np.arange(pos_weight_mean.shape[0]), pos_weight_mean - pos_weight_std, pos_weight_mean + pos_weight_std, alpha=0.2, edgecolor="green", facecolor="green",
        antialiased=True)


    fig2.savefig(f"temporal_weights_{args.modifier}.png")
    fig2.savefig(f"temporal_weights_{args.modifier}.pdf")


    #weights by step (slightly more tricky but doable)
    fig1, ax1 = plt.subplots()
    ax1.hist(bad_weights, 100, density=True, facecolor='r', alpha=0.2) # should show a bell curve
    ax1.hist(good_weights, 100, density=True, facecolor='g', alpha=0.2) # should show a bell curve
    fig1.savefig(f"weight_distribution_{args.modifier}.png")
    fig1.savefig(f"weight_distribution_{args.modifier}.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paired_data",
        type=str,
        required = True,
        default=None,
        help="(optional) render rollouts to this eval path",
    )

    parser.add_argument(
        "--good_data",
        type=str,
        required = True,
        default=None,
        help="(optional) render rollouts to this eval path",
    )

    parser.add_argument(
        "--classifier_path",
        type=str,
        required = True,
        default=None,
        help="(optional) render rollouts to this eval path",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required = False,
        default=None,
        help="(optional) render rollouts to this eval path",
    )


    parser.add_argument(
        "--modifier",
        type=str,
        required=True,
        default=None,
        help="(optional) render rollouts to this eval path",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default=None,
        help="(optional) render rollouts to this eval path",
    )


    args = parser.parse_args()
    main(args)

    # direct usage of r3m to embed
    # class ClassifierTest:
    #     def __init__(self):
    #         from r3m import load_r3m
    #         self.net = load_r3m("resnet18").module.convnet  # resnet18, resnet34
    #     def compute_embeddings(self, data):
    #         relevant = torch.tensor(data["agentview_image"], device = "cuda")
    #         self.net.eval()
    #         with torch.no_grad():
    #             embeddings = self.net(relevant * 255)
    #         return torch.cat((embeddings, torch.tensor(data["actions"], device = "cuda")), axis = 1).detach().cpu().numpy()
    #
    # test_net = ClassifierTest()
