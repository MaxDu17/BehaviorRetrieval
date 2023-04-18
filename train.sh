#!/bin/bash
#### FOR PRETRAINING ####

# example of a training setup
#python train.py --config configs/bc_rnn.json --name can_paired_lowdim --dataset datasets/can/paired/low_dim.hdf5 --debug
#python train.py --config configs/bc_rnn_image.json --name can_paired_image --dataset datasets/can/paired/image.hdf5 --debug


# use --train_size to select the size of the dataset (if not provided, the full thing will be used)
# use --debug to do a dry run
