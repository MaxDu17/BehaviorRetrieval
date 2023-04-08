#!/bin/bash
### Running Baselines
# use threshold = 0 for non-weighted (equivalent to balanced batches)
# use oracle dataset + threshold = 0 for oracle
# use flag GOOD_ONLY for expert demos only
# use pretrained_agent to load a pretrained model

## Other flags
# use dataset to specify the large D_prior dataset
# use sample_data to specify the small expert D_t dataset
# use classifier to specify the embedder used for BehaviorRetrieval

#### EXAMPLES ####
### normal / BB ##
#seed=2
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=threshold_08_VAE$seed
#python run_weighted_BC.py --config bc_rnn_office_image.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0.8 --seed $seed \
#  --eval_path bc_trained_models/$base_dir/$run_name/

### oracle ##
#seed=2
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=oracle_selection_seed$seed
#python run_weighted_BC.py --config bc_rnn_office_image.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new_ORACLE.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0 --seed $seed \
#  --eval_path bc_trained_models/$base_dir/$run_name/

### expert only ##
#seed=2
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=expert_only_10_seed$seed
#python run_weighted_BC.py --config bc_rnn_office_image.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0 --seed $seed \
#  --good_only \
#  --eval_path bc_trained_models/$base_dir/$run_name/

### goal-conditioned ##
#seed=2
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=goal_conditioned_baseline$seed
#python run_weighted_BC.py --config bc_rnn_office_image_goal.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0 --seed $seed \
#  --eval_path bc_trained_models/$base_dir/$run_name/ --goal

### goal-conditioned with selection ##
#seed=2
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=threshold_08_goal_conditioned$seed
#python run_weighted_BC.py --config bc_rnn_office_image_goal.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0.8 --seed $seed \
#  --eval_path bc_trained_models/$base_dir/$run_name/

### goal-conditioned with finetuning ##
#seed=201
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=goal_conditioned_finetuned$seed
#python run_weighted_BC.py --config bc_rnn_office_image_goal.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0 --seed $seed \
#  --pretrained_agent bc_trained_models/Office_PickPlace_Image_selected_BC/fixed_goal_conditioned_baseline2/model_epoch_100.pth \
#  --eval_path bc_trained_models/$base_dir/$run_name/

### finetuning ##
#seed=2
#base_dir=Office_PickPlace_Image_selected_BC
#run_name=finetune$seed
#python run_weighted_BC.py --config bc_rnn_office_image.json --name $run_name \
#  --dataset datasets/office/office_demos_individual_1200/office_image_new.hdf5 \
#  --classifier bc_trained_models/Embedder_VAE_office_image_k00001_eye_in_hand/20221117145626/models/model_epoch_400.pth \
#  --sample_data datasets/office/office_erasers_only_50/office_image_new.hdf5 --train_size 10 --threshold 0 --seed $seed \
#  --pretrained_agent bc_trained_models/office_task_eraser_image/20221212112524/models/model_epoch_200.pth \
#  --eval_path bc_trained_models/$base_dir/$run_name/
