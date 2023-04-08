from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP




@register_algo_factory_func("lmp")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    return LMP, {}


class LMP(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict() #hack to work
        self.nets["policy"] = nn.ModuleDict()
        # LOTS OF HARDCODING FOR INITIAL PROGRESS
        self.nets["policy"]["visual_encoder"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=128,
            mlp_layer_dims=self.algo_config.visual_encoder_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets["policy"]["plan_proposal"] = PolicyNets.GaussianActorNetwork(
            obs_shapes= OrderedDict([('state_encoding', [128])]),
            goal_shapes=OrderedDict([('state_encoding', [128])]),
            ac_dim=256,
            mlp_layer_dims=self.algo_config.plan_proposal_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # needs to be larger 2048 latent (4 layers)
        self.nets["policy"]["plan_recognition"] = PolicyNets.TrajEncoder_Network(
            obs_shapes=OrderedDict([('state_encoding', [128])]),
            ac_dim=256,
            mlp_layer_dims=self.algo_config.plan_recognition_dims,
            std_activation=self.algo_config.gaussian.std_activation,
            fixed_std = self.algo_config.gaussian.fixed_std,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        # this needs to be larger 2048, 2 layers
        self.nets["policy"]["policy_network"] = PolicyNets.GMMActorNetwork(
            obs_shapes= OrderedDict([('state_encoding', [128]), ('latent_plan', [256])]),
            goal_shapes=OrderedDict([('state_encoding', [128])]),
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.policy_network_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets["policy"] = self.nets["policy"].float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        # batch from sequence should be of l
        # torch.nn.utils.rnn.pack_padded_sequence

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["plan_seq"] = batch["plan_seq"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        # print("sanity check")
        # batch["goal_obs"] = {k :  torch.zeros_like(v) for k, v in batch["goal_obs"].items()}
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(LMP, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions["likelihood"]) #quick fix for now
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        visual_encodings = OrderedDict()
        goal_visual_encodings = OrderedDict()
        plan_visual_encodings = OrderedDict()
        policy_input = OrderedDict()

        visual_encodings["state_encoding"] = self.nets["policy"]["visual_encoder"](obs_dict = batch["obs"])
        goal_visual_encodings["state_encoding"] = self.nets["policy"]["visual_encoder"](obs_dict = batch["goal_obs"])





        batch_size, seq_len = batch["plan_seq"]["agentview_image"].shape[0], batch["plan_seq"]["agentview_image"].shape[1]
        flattened_plan_seq = {k : torch.flatten(batch["plan_seq"][k], 0, 1) for k in batch["plan_seq"]}
        flattened_plan_encoding = self.nets["policy"]["visual_encoder"](obs_dict = flattened_plan_seq)
        unflatten_func = torch.nn.Unflatten(0, (batch_size, seq_len))
        plan_visual_encodings["state_encoding"] = unflatten_func(flattened_plan_encoding)

        plan_recog_dists = self.nets["policy"]["plan_recognition"].forward_train(obs_dict = plan_visual_encodings)
        plan_proposal_dists = self.nets["policy"]["plan_proposal"].forward_train(obs_dict = visual_encodings, goal_dict = goal_visual_encodings)

        policy_input["state_encoding"] = visual_encodings["state_encoding"]
        policy_input["latent_plan"] = plan_recog_dists.rsample() #take the longer horizon embedding

        action_dist = self.nets["policy"]["policy_network"].forward_train(obs_dict = policy_input, goal_dict = goal_visual_encodings)
        predictions["likelihood"] = action_dist.log_prob(batch["actions"])
        predictions["recog_latent"] = plan_recog_dists
        predictions["proposal_latent"] = plan_proposal_dists
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        # a_target = batch["actions"]
        # actions = predictions["actions"]
        losses["likelihood"] = -predictions["likelihood"].mean()
        # losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        # losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        # losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        # action_losses = [
        #     self.algo_config.loss.l2_weight * losses["l2_loss"],
        #     self.algo_config.loss.l1_weight * losses["l1_loss"],
        #     self.algo_config.loss.cos_weight * losses["cos_loss"],
        # ]
        # action_loss = sum(action_losses)
        action_loss = losses["likelihood"]
        latent_loss = sum(torch.distributions.kl.kl_divergence(predictions["recog_latent"], predictions["proposal_latent"]))
        # in the 700's
        losses["action_loss"] = action_loss
        losses["latent_loss"] = latent_loss

        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"] + 0.01 * losses["latent_loss"], #used to be 0.0001
        )
        info[f"policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(LMP, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Latent Loss"] = info["losses"]["latent_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        visual_encodings = OrderedDict()
        goal_visual_encodings = OrderedDict()
        policy_input = OrderedDict()

        visual_encodings["state_encoding"] = self.nets["policy"]["visual_encoder"](obs_dict=obs_dict)
        goal_visual_encodings["state_encoding"] = self.nets["policy"]["visual_encoder"](obs_dict=goal_dict)
        plan_proposal_dists = self.nets["policy"]["plan_proposal"].forward_train(obs_dict=visual_encodings,
                                                                                 goal_dict=goal_visual_encodings)

        policy_input["state_encoding"] = visual_encodings["state_encoding"]
        policy_input["latent_plan"] = plan_proposal_dists.sample()
        actions = self.nets["policy"]["policy_network"](obs_dict=policy_input, goal_dict=goal_visual_encodings)

        return actions
