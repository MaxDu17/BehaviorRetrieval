# Behavior Retrieval
This repository contains the code for BehaviorRetrieval, a submission to RSS 2023. 
# Installing
1. Install torch + torchvision 
2. Install [robosuite](https://robosuite.ai/) 
3. Install dependencies in `requirements.txt` (covers robomimic and roboverse dependencies)
4. Install robomimic by using `pip install -e .` inside the `robomimic` folder
5. (If you want to run Office) Install roboverse by using `pip install -e.` inside the `roboverse` folder

# Running 
## Configurations
Find all the configurations for training in `configs/`. We follow the Robomimic convention of keeping hyperparameters in the `.json` files. We have special `office` configuraitons for the Office task due to differences of the Roboverse environment. 
## Collecting Data
* Can task: use the `paired` data provided by Robomimic: [download](http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/image.hdf5)
* Square task: use the MachinePolicy to collect demonstrations. Read the script in `run_trained_agent.sh` for more information
* Office task: use the `scripted_collect.sh` in the `roboverse/scripts` folder. Use `utils/roboverse_to_robomimic.py` to convert the demo format to the one used by our codebase

## Training The Embedder
Use `train_embedder.py`, which can handle contrastive and VAE embeders. For configs, see `train_embedder.sh` 

## Running BehaviorRetrieval
Use `run_weighted_BC.py`, which runs BehaviorRetieval and our baselines. For example configs, see `run_weighted_BC.sh`

To train with vanilla BC, use `train.py`. We use this to pretrain the model for the HG-DAGGER experiments. See example config in `train.sh`. 

To run HG-DAGGER with BehaviorRetrieval, use `run_weighted_corrections.py`. See `run_weighted_corrections.sh` for an example config. 

## Evaluating BehaviorRetrieval and Baselines
Use `run_trained_agent.py` to run evals. See `run_trained_agent.sh` for configs. 

## Real Robot Experiments
If the robot interfaces with the Gym environment, this codebase works with real robots out of the box.  

# Visualizing Results
Use `visualizing_embeddings.py` to compute a T-SNE or PCA visualization of the embeddings. Use `embedding_analysis.py` to compute a plot of how similarity changes through an episode (like Figure 8 in our paper)

