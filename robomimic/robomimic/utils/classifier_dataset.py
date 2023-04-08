"""
This file contains weighted Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files. It inherits directly form SequenceDataset, but it changes the sampling approach
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.utils.dataset import SequenceDataset

class ClassifierDataset(SequenceDataset):

    def __init__(self,
                 hdf5_path,
                 obs_keys,
                 dataset_keys,
                 frame_stack=1,
                 seq_length=1,
                 pad_frame_stack=True,
                 pad_seq_length=True,
                 get_pad_mask=False,
                 goal_mode=None,
                 hdf5_cache_mode=None,
                 hdf5_use_swmr=True,
                 hdf5_normalize_obs=False,
                 filter_by_attribute=None,
                 load_next_obs=True,
                 priority=False,
                 weighting=False,
                 num_samples=None,
                 radius = 15,
                 use_actions = False):
        super(ClassifierDataset, self).__init__(
                 hdf5_path,
                 obs_keys,
                 dataset_keys,
                 frame_stack,
                 seq_length,
                 pad_frame_stack,
                 pad_seq_length,
                 get_pad_mask,
                 goal_mode,
                 hdf5_cache_mode,
                 hdf5_use_swmr,
                 hdf5_normalize_obs,
                 filter_by_attribute,
                 load_next_obs,
                 priority,
                 weighting,
                 num_samples,
            )

        self.radius = radius
        self.use_actions = use_actions
        # self.same_traj = same_traj

    # just overriding the sampling of the dataset.py for positive-negative classifying 
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        assert self.hdf5_cache_mode == "low_dim", "highdim not yet implemented on the classifier dataset"
        #varying positions

        demo_id = self._index_to_demo_id[index]

        #picking between the demos
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        viable_sample_size = self._demo_id_to_demo_length[demo_id]
        demo_end_index = demo_start_index + viable_sample_size
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)  # only works for one frame stack
        index_in_demo = index - demo_start_index + demo_index_offset

        lower_bound = max(0, index_in_demo - self.radius)
        upper_bound = min(viable_sample_size - 1, index_in_demo + self.radius)

        positive_index_in_demo = np.random.randint(lower_bound, upper_bound) #upper_bound - lower_bound) + lower_bound
        perturb = np.random.randint(viable_sample_size - (upper_bound - lower_bound))
        negative_index_in_demo = (upper_bound + perturb) % viable_sample_size

        # print(index_in_demo, positive_index_in_demo, negative_index_in_demo)
        # print(abs(positive_index_in_demo - negative_index_in_demo))
        data = {}

        # profiling this
        data["anchor"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length = 1,
            prefix="obs"
        )
        data["positive"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=positive_index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        data["negative"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=negative_index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )

        if self.use_actions:
            actions, _ = self.get_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=["actions"],
                num_frames_to_stack=0,  # don't frame stack for meta keys
                seq_length = 1,
            )
            data["anchor"].update(actions)
            actions, _ = self.get_sequence_from_demo(
                demo_id,
                index_in_demo=positive_index_in_demo,
                keys=["actions"],
                num_frames_to_stack=0,  # don't frame stack for meta keys
                seq_length=1,
            )
            data["positive"].update(actions)
            actions, _ = self.get_sequence_from_demo(
                demo_id,
                index_in_demo=negative_index_in_demo,
                keys=["actions"],
                num_frames_to_stack=0,  # don't frame stack for meta keys
                seq_length=1,
            )
            data["negative"].update(actions)
        return data


    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = super().__repr__()
        return "CLASSIFIER DATASET \n" + msg


class TemporalEmbeddingDataset(SequenceDataset):

    def __init__(self,
                 hdf5_path,
                 obs_keys,
                 dataset_keys,
                 frame_stack=1,
                 seq_length=1,
                 pad_frame_stack=True,
                 pad_seq_length=True,
                 get_pad_mask=False,
                 goal_mode=None,
                 hdf5_cache_mode=None,
                 hdf5_use_swmr=True,
                 hdf5_normalize_obs=False,
                 filter_by_attribute=None,
                 load_next_obs=True,
                 priority=False,
                 weighting=False,
                 num_samples=None,
                 geometric_p = 0.1
                 ):
        super(TemporalEmbeddingDataset, self).__init__(
            hdf5_path,
            obs_keys,
            dataset_keys,
            frame_stack,
            seq_length,
            pad_frame_stack,
            pad_seq_length,
            get_pad_mask,
            goal_mode,
            hdf5_cache_mode,
            hdf5_use_swmr,
            hdf5_normalize_obs,
            filter_by_attribute,
            load_next_obs,
            priority,
            weighting,
            num_samples,
        )
        self.geometric_p = geometric_p

    # just overriding the sampling funcitonality
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        assert self.hdf5_cache_mode == "low_dim", "highdim not yet implemented on the classifier dataset"
        # varying positions

        demo_id = self._index_to_demo_id[index]

        # picking between the demos
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        viable_sample_size = self._demo_id_to_demo_length[demo_id]
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)  # only works for one frame stack
        index_in_demo = index - demo_start_index + demo_index_offset

        # second_index_in_demo = index_in_demo

        # second_index_in_demo = min(viable_sample_size - 1, index_in_demo + 1) #SNITY CHECK
        second_index_in_demo = min(viable_sample_size - 1, np.random.geometric(self.geometric_p) + index_in_demo) #the s^+ is some steps in the future

        data = {}

        data["anchor"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )

        actions, _ = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=["actions"],
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=1,
        )
        data["anchor"].update(actions)

        data["future"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=second_index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        return data

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = super().__repr__()
        return "CLASSIFIER DATASET \n" + msg


class DistanceClassifierDataset(SequenceDataset):

    def __init__(self,
                 hdf5_path,
                 obs_keys,
                 dataset_keys,
                 frame_stack=1,
                 seq_length=1,
                 pad_frame_stack=True,
                 pad_seq_length=True,
                 get_pad_mask=False,
                 goal_mode=None,
                 hdf5_cache_mode=None,
                 hdf5_use_swmr=True,
                 hdf5_normalize_obs=False,
                 filter_by_attribute=None,
                 load_next_obs=True,
                 priority=False,
                 weighting=False,
                 num_samples=None,
                 radius=15,
                 use_actions=False):
        super(DistanceClassifierDataset, self).__init__(
            hdf5_path,
            obs_keys,
            dataset_keys,
            frame_stack,
            seq_length,
            pad_frame_stack,
            pad_seq_length,
            get_pad_mask,
            goal_mode,
            hdf5_cache_mode,
            hdf5_use_swmr,
            hdf5_normalize_obs,
            filter_by_attribute,
            load_next_obs,
            priority,
            weighting,
            num_samples,
        )

        self.radius = radius
        self.use_actions = use_actions
        # self.same_traj = same_traj

    # just overriding the sampling funcitonality
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        assert self.hdf5_cache_mode == "low_dim", "highdim not yet implemented on the classifier dataset"
        # varying positions

        demo_id = self._index_to_demo_id[index]

        # picking between the demos
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        viable_sample_size = self._demo_id_to_demo_length[demo_id]
        demo_end_index = demo_start_index + viable_sample_size
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)  # only works for one frame stack
        index_in_demo = index - demo_start_index + demo_index_offset

        second_index_in_demo = np.random.randint(0, viable_sample_size)
        index_distances = abs(second_index_in_demo - index_in_demo)

        data = {}

        # profiling this
        data["anchor"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        data["second"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=second_index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        data["distance"] = index_distances

        if self.use_actions:
            actions, _ = self.get_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=["actions"],
                num_frames_to_stack=0,  # don't frame stack for meta keys
                seq_length=1,
            )
            data["anchor"].update(actions)
            actions, _ = self.get_sequence_from_demo(
                demo_id,
                index_in_demo=second_index_in_demo,
                keys=["actions"],
                num_frames_to_stack=0,  # don't frame stack for meta keys
                seq_length=1,
            )
            data["second"].update(actions)
        return data

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = super().__repr__()
        return "CLASSIFIER DATASET \n" + msg
