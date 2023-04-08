# Modified Roboverse for BehaviorRetrieval Office Environment
This is the roboverse environment [repository](https://github.com/VentusYue/roboverse) with small modifications for Behavior Retrieval.
* added eye-in-hand camera 
* modified scripted policy to collect single and double pick-and-place 

For more information, visit our forked [repository](https://github.com/MaxDu17/roboverse/)

# roboverse
A set of environments utilizing [pybullet](https://github.com/bulletphysics/bullet3) was simulation of robotic manipulation tasks. 

## Usage
Creating and using environments is simple:
```python
import roboverse
env = roboverse.make('Widow250DoubleDrawerOpenNeutral-v0', gui=True)
env.reset()
for _ in range(25):
    env.step(env.action_space.sample())
```
## Setup
I recommend using [conda](https://docs.anaconda.com/anaconda/install/) for setup:

```
conda create -n roboverse python=3.6
source activate roboverse
pip install -r requirements.txt
```
When using this repository with other projects, run `pip install -e .` in the root directory of this repo. 

To test if things are working by visualizing a scripted robot policy, run the following command:

`python scripts/scripted_collect.py -n 100 -t 30 -e Widow250DoubleDrawerOpenNeutral-v0 -pl drawer_open_transfer -a drawer_opened_success --noise=0.1 --gui`

## If you want to dig into the code, start here:
`python roboverse/envs/widow250.py`

## Credit
Primary developers: [Avi Singh](https://www.avisingh.org/), Albert Yu, Jonathan Yang, [Michael Janner](https://people.eecs.berkeley.edu/~janner/), Huihan Liu, Gaoyue Zhou

## Office clean Environment 
This Environment simulate an office cleaning task.

run the following command to collect data:

`python scripts/scripted_collect.py -n 1000 -t 280 -e Widow250OfficePickPlaceRandom-v0 -pl tableclean -a table_clean --noise=0.1 --gui`

To use multi-threading, run the following:

`python scripts/scripted_collect_parallel.py -p 12 -n 16000 -t 280 -e Widow250OfficePickPlaceRandom-v0 -pl tableclean -a table_clean --noise=0.1 -d office -f 1`


To modify the configs, check:

`roboverse/envs/regieration.py, Widow250OfficePickPlaceRandom, Widow250OfficePickPlaceFixed`
