#!/bin/bash

#### HOW TO RUN NORMAL EVALS ###
#checkpoint=400
#python run_trained_agent.py \
#  --agent bc_trained_models/can_paired_image_selected_BC/expert_only_10_seed3/model_epoch_$checkpoint.pth \
#  --checkpoint $checkpoint --mode rollout  --eval_path bc_trained_models/can_paired_image_selected_BC/expert_only_10_seed3/\
#  --n_rollouts 100 --horizon 400 --seed 1 --video_skip 2 --eval_skip 10

#checkpoint=600 # for goal-conditioned
#python run_trained_agent.py \
#  --agent bc_trained_models/can_paired_image_selected_BC/fixed_goal_conditioned_baseline1/model_epoch_$checkpoint.pth \
#  --checkpoint $checkpoint --mode rollout  --eval_path bc_trained_models/can_paired_image_selected_BC/fixed_goal_conditioned_baseline1/\
#  --n_rollouts 100 --horizon 400 --seed 1 --video_skip 2 --eval_skip 10 --goal_data datasets/can/paired/image_10_good.hdf5


### COLLECTING MACHINE GENERATED DATA ####
# for GOOD demos
#python run_trained_agent.py \
#  --agent SquarePeg \
#  --mode rollout  --eval_path datasets/square_machine_policy/ \
#  --reference_data datasets/square/ph/image.hdf5 \
#  --config configs/image_collection.json \
#  --machine_oracle --success_only --dataset_obs \
#  --dataset_path datasets/square_machine_policy/square_10_good.hdf5 \
#  --n_rollouts 10 --horizon 400 --seed 11

# for MIXED demos
#python run_trained_agent.py \
#  --agent SquarePeg \
#  --mode rollout  --eval_path datasets/square_machine_policy/ \
#  --reference_data datasets/square/ph/image.hdf5 \
#  --config configs/image_collection.json \
#  --machine_oracle --success_only --dataset_obs \
#  --dataset_path datasets/square_machine_policy/square_400_paired.hdf5 \
#  --n_rollouts 400 --horizon 400 --seed 11 --paired


#### HOW TO COLLECT EXPERT DEMOS FROM A PRETRAINED AGENT ####
#checkpoint=440
#python run_trained_agent.py \
#  --agent bc_trained_models/can_image/20220707164110/models/model_epoch_440.pth \
#  --checkpoint reset_$checkpoint --mode rollout  --dataset_path datasets/can/paired/image_10_good.hdf5\
#  --eval_path datasets/can/paired --dataset_obs\
#  --n_rollouts 10 --horizon 400 --success_only --seed 11



