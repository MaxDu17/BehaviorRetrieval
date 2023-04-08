#!/bin/bash

# EXAMPLE
# limit is the number of steps to plot
# ASSUMES oracle labels (to plot the correct and incorrect trajectories)

#python embedding_analysis.py \
#  --paired_data ../datasets/square_machine_policy/square_400_paired.hdf5 \
#  --good_data ../datasets/square_machine_policy/square_10_good.hdf5 \
#  --classifier_path ../bc_trained_models/Embedder_VAE_square_image_k0001/20221116102331/models/model_epoch_1000.pth \
#  --modifier output_image_square \
#  --config ../bc_rnn_image.json --limit 325
