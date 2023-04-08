import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_nets import MIMO_MLP
from robomimic.models.distributions import DiscreteValueDistribution


class WeighterNet(MIMO_MLP):
    """
    A basic value network that predicts values from observations.
    Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        mlp_layer_dims,
        encoder_kwargs=None,
        weight_bounds = None,
        head_activation = "sigmoid"
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            weight_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. The network will rescale outputs
                using a tanh layer to lie within these bounds. If None, no tanh re-scaling is done.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        self.head_activation = head_activation

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        output_shapes = self._get_output_shapes()
        super(WeighterNet, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict values, but may instead predict the parameters
        of a value distribution.
        """
        return OrderedDict(value=(1,))

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [1]

    def forward(self, obs_dict_1, obs_dict_2):
        """
        Forward through value network, and then optionally use tanh scaling.
        """

        obs_dict_combined = OrderedDict()
        batch_size = obs_dict_1[list(obs_dict_1.keys())[0]].shape[0] # a nasty hack, but it is guarenteeed to work
        random_mask = np.zeros(batch_size, dtype=bool)
        random_mask[0:batch_size // 2] = 1
        np.random.shuffle(random_mask)
        bin_1 = np.where(random_mask)
        bin_2 = np.where(~random_mask)

        for key_1, key_2 in zip(obs_dict_1.keys(), obs_dict_2.keys()):
            assert key_1 == key_2, "The keys you're trying to fuse are not the same!"
            # permuted_1 = torch.zeros_like(obs_dict_1[key_1])
            # permuted_2 = torch.zeros_like(obs_dict_2[key_2])
            #
            # permuted_1[bin_1] = obs_dict_1[key_1][bin_1]
            # permuted_2[bin_2] = obs_dict_1[key_1][bin_2]
            #
            # permuted_2[bin_1] = obs_dict_2[key_2][bin_1]
            # permuted_1[bin_2] = obs_dict_2[key_2][bin_2]


            obs_dict_combined[key_1] = torch.cat((obs_dict_1[key_1], obs_dict_2[key_2]), dim = 1)
            # obs_dict_combined[key_1] = torch.cat((permuted_1, permuted_2), dim = 1)

        weights = super(WeighterNet, self).forward(obs=obs_dict_combined)["value"]
        if self.head_activation == "sigmoid":
            weights = torch.sigmoid(weights)
        elif self.head_activation == "relu":
            weights = weights
            # weights = torch.relu(weights)
        else:
            raise Exception("invalid head activation type")
        return weights

    def _to_string(self):
        return ""
