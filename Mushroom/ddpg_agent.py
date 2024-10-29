import numpy as np
import torch
from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from torch import nn, optim
import torch.nn.functional as F


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
                                gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = F.sigmoid(self._h3(features2))

        return a


def create_ddpg_agent(
        mdp,
        n_features_actor,
        lr_actor,
        n_features_critic,
        lr_critic,
        batch_size,
        initial_replay_size,
        max_replay_size,
        tau,
        sigma=0.2,
        theta=0.15,
        dt=1e-2
):
    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features_actor,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}  # not so big of a difference to critic

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    # policy_class = GaussianPolicy
    # policy_params = dict(sigma=np.ones(6) * .2)

    # Agent
    return DDPG(mdp.info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params,
                 batch_size, initial_replay_size, max_replay_size,
                 tau)