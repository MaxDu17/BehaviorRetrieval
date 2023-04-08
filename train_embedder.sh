#!/bin/bash

#VAE EMBEDDER TRAINING 

#python train_classifier.py \
#  --config configs/embedder_VAE_image.json \
#  --name Embedder_VAE_square_image\
#  --dataset datasets/square_machine_policy/square_400_paired.hdf5

#python train_classifier.py \
#  --config configs/embedder_VAE_image.json \
#  --name Embedder_VAE_can_image\
#  --dataset datasets/can/paired/image.hdf5

#python train_classifier.py \
#  --config configs/embedder_VAE_lowdim.json \
#  --name Embedder_VAE_square_lowdim\
#  --dataset datasets/square_machine_policy/square_400_paired.hdf5

#python train_classifier.py \
#  --config configs/embedder_VAE_lowdim.json \
#  --name Embedder_VAE_can_lowdim\
#  --dataset datasets/can/paired/image.hdf5

#python train_classifier.py \
#  --config configs/embedder_VAE_image_office.json \
#  --name Embedder_VAE_office_image\
#  --dataset datasets/office/office_demos_individual_1200/office_image.hdf5

#python train_classifier.py \
#  --config configs/embedder_VAE_lowdim_office.json \
#  --name Embedder_VAE_office_lowdim\
#  --dataset datasets/office/office_demos_individual_1200/office_image.hdf5


# CONTRASTIVE embedder training 
#python train_classifier.py \
#  --config configs/classifier_temporal_image_office.json \
#  --name Embedder_office_image_contrastive \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5

# python train_classifier.py \
#   --config configs/classifier_temporal_office.json \
#   --name Embedder_Office_contrastive \
#   --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5

#python train_classifier.py \
#  --config configs/classifier_temporal_image.json \
#  --name Embedder_can_paired_image_contrastive\
#  --dataset datasets/can/paired/image.hdf5

#python train_classifier.py \
#  --config configs/classifier_temporal.json \
#  --name Embedder_can_paired_lowdim_contrastive \
#  --dataset datasets/can/paired/low_dim.hdf5

#python train_classifier.py \
#  --config configs/classifier_temporal_image.json \
#  --name Embedder_square_paired_image_time_contrastive \
#  --dataset datasets/square_machine_policy/square_400_paired.hdf5

#python train_classifier.py \
#  --config configs/classifier_temporal.json \
#  --name Embedder_square_paired_lowdim_time_contrastive \
#  --dataset datasets/square_machine_policy/square_400_paired.hdf5