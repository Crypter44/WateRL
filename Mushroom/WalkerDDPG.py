import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.core import Core
from mushroom_rl.environments.dm_control_env import DMControl
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.dataset import compute_J, compute_metrics
from tqdm import tqdm

from utils import render_episode, set_seed, set_mujoco_custom_rendering, plot_J

set_seed(526)
# TODO This requires my own custom code for the core of mushroom_rl to work, SET TO FALSE IF NOT AVAILABLE
set_mujoco_custom_rendering(True)

# Parametrization
horizon = 500
gamma = 0.99
gamma_eval = 1.

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200
n_features = 80
tau = .001

n_epochs = 50
n_steps = 1000
n_steps_test = 2000

# Define the neural networks for the actor and the critic
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

# MDP
mdp = DMControl('walker', 'stand', horizon, gamma)

# Policy
policy_class = OrnsteinUhlenbeckPolicy
policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

# policy_class = GaussianPolicy
# policy_params = dict(sigma=np.ones(6) * .2)

# Approximator
actor_input_shape = mdp.info.observation_space.shape
actor_params = dict(network=ActorNetwork,
                    n_features=n_features,
                    input_shape=actor_input_shape,
                    output_shape=mdp.info.action_space.shape)

actor_optimizer = {'class': optim.Adam,
                   'params': {'lr': .0001}} # not so big of a difference to critic

critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
critic_params = dict(network=CriticNetwork,
                     optimizer={'class': optim.Adam,
                                'params': {'lr': .001}},
                     loss=F.mse_loss,
                     n_features=n_features,
                     input_shape=critic_input_shape,
                     output_shape=(1,))

# Agent
agent = DDPG(mdp.info, policy_class, policy_params,
             actor_params, actor_optimizer, critic_params,
             batch_size, initial_replay_size, max_replay_size,
             tau)

# Algorithm
core = Core(agent, mdp)

# Fill the replay memory with random samples
core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)

# ------------- RUN EXPERIMENT ------------- #
dataset = core.evaluate(n_steps=n_steps_test, render=False)
Js = []
metrics = []

J = compute_J(dataset, gamma_eval)
Js.append(J)

pbar = tqdm(range(n_epochs), desc='Running... ', unit='epoch')
for n in pbar:
    # learning step, agent learns from 1000 steps -> 2 episodes
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)
    # evaluation step, agent evaluates 2000 steps -> 4 episodes
    dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
    if (n+1) % 5 == 0:
        render_episode(core, f"../Videos/walker-stand-ep{n+1}.mp4")
    J = compute_J(dataset, gamma_eval)
    Js.append(J)
    metric = compute_metrics(dataset, gamma_eval)
    pbar.set_postfix(J=np.mean(J), Metrics=metric)

# Plot the results
plot_J(Js)