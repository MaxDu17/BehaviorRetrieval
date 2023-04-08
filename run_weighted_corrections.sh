#!/bin/bash

# here are EXAMPLES of how the code can be run...you will need to provide your own checkpoints

###### SQUARE IMAGE #####
# for balanced batches, set threshold = 0
#seed=3
#run_name=threshold_085_VAE_selection$seed
#base_dir=square_paired_machine_image
#
#python run_weighted_corrections.py \
# --machine_oracle \
# --oracle_agent SquarePeg \
# --apprentice_agent bc_trained_models/square_paired_machine_image/20220908111304/models/model_epoch_300.pth \
# --embedder bc_trained_models/Embedder_VAE_square_image_k0001/20221108230711/models/model_epoch_200.pth \
# --config configs/bc_rnn_image.json \
# --expert_dataset datasets/square_machine_policy/square_400_paired.hdf5 \
# --correction_config configs/correction_config.json \
# --intervention_save_path bc_trained_models/$base_dir/$run_name.hdf5 --dataset_obs \
# --eval_path bc_trained_models/$base_dir/$run_name/ --seed $seed

#seed=3
#run_name=baseline_correction_only$seed
#base_dir=square_paired_machine_image
#python run_weighted_corrections.py \
# --machine_oracle \
# --oracle_agent SquarePeg \
# --apprentice_agent bc_trained_models/square_paired_machine_image/20220908111304/models/model_epoch_400.pth \
# --embedder bc_trained_models/RadiusClassifier_square_paired_image_time_contrastive_l2/20221019101939/models/model_epoch_1000.pth \
# --config configs/bc_rnn_image.json \
# --expert_dataset datasets/square_machine_policy/square_400_paired.hdf5 \
# --correction_config configs/correction_config.json \
# --intervention_save_path bc_trained_models/$base_dir/$run_name.hdf5 --dataset_obs \
# --baseline_type corrections_only \
# --eval_path bc_trained_models/$base_dir/$run_name/ --seed $seed
