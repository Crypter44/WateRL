# Decentralized Operation of Urban Water Distribution Networks Using Multi-Agent Reinforcement Learning

This repository contains the code for my bachelor's thesis. 

## Abstract
This thesis explores the use of Multi-Agent Reinforcement Learning (MARL) for the efficient operation of urban
Water Distribution Networks (WDNs). While previous work focused on industrial networks with hierarchical
topologies, we extend the application of MARL to urban WDNs with cyclic structures and more complex flow
dynamics. Following the Centralized Training for Decentralized Execution (CTDE) paradigm, we investigate
the performance of two deep MARL algorithms, IDDPG and FACMAC, across a range of control tasks.
We evaluate the impact of different reward structures on energy efficiency and demonstrate that a multifaceted
reward function, combining demand deviation, power penalties, and a target for maximum valve openings,
leads to the most robust and efficient agent behavior. Our results further highlight the importance of
observability: fully observable agents achieve the highest efficiency and generalize better to unseen scenarios,
while partially observable agents face a trade-off between efficiency and robustness depending on training
duration.
Finally, we assess the importance of centralized learning in cooperative tasks with temporal planning. FACMAC,
leveraging centralized training, enables agents to coordinate pump and tank operations to preserve water for
peak demand hours, outperforming IDDPG’s decentralized learning approach. These findings demonstrate the
potential of MARL to enable smart and sustainable control of urban water infrastructure.

## Installation
To run the code, you need to install the required packages. You can  install them from the `requirements.txt` file.
Some experiments use wandb for logging, for those an account is required. Alternatively, they can be run with the paramatrized
training function provided in `utils.py`, but this will require some modifications to the code.

## Usage
To run a training experiment, run the script `train_xxx.py` where `xxx` is the algorithm you want to use. 
These scripts exist for each experiment, located in `Mushroom/training/fluid`.

## Reproducing the experiments 
### Observability Experiment comparing IDDPG with full and partial observability
This experiment is performed in the circular network and can be found in `Mushroom/training/fluid/circular_network/`.
To reproduce the experiment, choose the desired output folder in the scripts and run the script `reproduce_fully_obs.py`,
`reproduce_partially_obs.py` and `reproduce_partially_obs_early_stopped.py` for the different observability settings.

The results can then be evaluated using the weights saved in the previously chosen folder and the evaluation tools provided in the evaluation folder.
### Cooperation Experiment comparing IDDPG and FACMAC
When cloning the repository, the default training parameters of FACMAC are set to reproduce the results shown in the thesis.
Simply run the script `train_facmac.py` in the `Mushroom/training/fluid/minimal_tank_network` to train the model.
