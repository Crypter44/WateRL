import inspect
import json
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.utils.dataset import compute_metrics
from torch import nn, optim
from tqdm import tqdm

from Mushroom.agents.sigma_decay_policies import UnivariateGaussianPolicy


# Define the neural networks for the actor and the critic
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, agent_idx=-1, ma_critic=False, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._agent_idx = agent_idx
        self._ma_critic = ma_critic
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
        if self._agent_idx != -1:
            if state.ndim == 3:
                state = state[:, self._agent_idx, :]
        if action.ndim == 3:
            if not self._ma_critic:
                action = action[:, self._agent_idx, :]
            else:
                action = torch.squeeze(action)
        state_action = torch.cat((state.float(), action.float()), dim=-1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, agent_idx=-1, **kwargs):
        super(ActorNetwork, self).__init__()

        self._n_input = input_shape[-1]
        self._n_output = output_shape[0]

        self._agent_idx = agent_idx
        self._h1 = nn.Linear(self._n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, self._n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, state):
        if self._agent_idx != -1:
            if state.ndim == 3:
                state = state[:, self._agent_idx, :]
        if self._n_input != 1:
            state = torch.squeeze(state, 1)
        features1 = F.relu(self._h1(state.float()))
        features2 = F.relu(self._h2(features1))
        a = F.sigmoid(self._h3(features2))
        # a = a * (self.mdp.info.action_space.high - self.mdp.info.action_space.low) + self.mdp.info.action_space.low

        return a


class MADDPG(DDPG):
    def __init__(
            self, mdp_info, agend_idx, policy_class, policy_params, actor_params, actor_optimizer, critic_params, batch_size,
            initial_replay_size, max_replay_size, tau, policy_delay=1, critic_fit_params=None,
            actor_predict_params=None, critic_predict_params=None
    ):
        self.agents: Optional[List[MADDPG]] = None
        self.agent_idx = agend_idx
        super().__init__(
            mdp_info, policy_class, policy_params, actor_params, actor_optimizer, critic_params, batch_size,
            initial_replay_size, max_replay_size, tau, policy_delay, critic_fit_params, actor_predict_params,
            critic_predict_params
        )

    def _next_q(self, next_state, absorbing):
        q = self._target_critic_approximator.predict(
            next_state,
            self._draw_actions(next_state),
            **self._critic_predict_params
        )
        q *= 1 - absorbing

        return q

    def _loss(self, state):
        q = self._critic_approximator(
            state,
            self._draw_actions(state),
            output_tensor=True,
            **self._critic_predict_params
        )

        return -q.mean()

    def _draw_actions(self, state):
        actions = []
        for agent in self.agents:
            a = agent._actor_approximator(state, output_tensor=True, **self._actor_predict_params)
            if agent.agent_idx == self.agent_idx:
                a.detach()
            actions.append(a)
        actions = torch.cat(actions, 1)
        return actions


def create_ddpg_agent(
        mdp,
        agent_idx=-1,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=1e-3,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma_checkpoints=None,
        decay_type='exponential',
        sigma=None,
        target_sigma=None,
        sigma_transition_length=None,
        ma_critic=False,
        save_path=None
):
    # Approximator
    actor_input_shape = mdp.local_observation_space(agent_idx).shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features_actor,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space_for_idx(agent_idx).shape,
                        agent_idx=agent_idx)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    if ma_critic:
        critic_input_shape = actor_input_shape[0]
        for i in range(mdp.info.n_agents):
            critic_input_shape += mdp.local_action_space(agent_idx).shape[0]
        critic_input_shape = (critic_input_shape,)
    else:
        critic_input_shape = (actor_input_shape[0] + mdp.local_action_space(agent_idx).shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         agent_idx=agent_idx,
                         ma_critic=ma_critic)

    policy_class = UnivariateGaussianPolicy
    policy_params = dict(initial_sigma=sigma, target_sigma=target_sigma, decay_type=decay_type,
                         updates_till_target_reached=sigma_transition_length, sigma_checkpoints=sigma_checkpoints)

    if save_path is not None:
        with open(save_path + f"Agent_{agent_idx}.json", "w") as f:
            frame = inspect.currentframe()
            args_as_dict = inspect.getargvalues(frame).locals
            del args_as_dict['frame']  # Avoid including the frame reference
            json.dump({key: str(value) for key, value in args_as_dict.items()}, f, indent=4, )

    if not ma_critic:
        agent = DDPG(mdp.info, policy_class, policy_params,
                     actor_params, actor_optimizer, critic_params,
                     batch_size, initial_replay_size, max_replay_size,
                     tau)
    else:
        agent = MADDPG(mdp.info, agent_idx, policy_class, policy_params,
                       actor_params, actor_optimizer, critic_params,
                       batch_size, initial_replay_size, max_replay_size,
                       tau)
    return agent


def create_maddpg_agents(
        n_agents,
        mdp,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=1e-3,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma_checkpoints=None,
        decay_type='exponential',
        sigma=None,
        target_sigma=None,
        sigma_transition_length=None,
        save_path=None
):
    agents = []
    for i in range(n_agents):
        agents.append(
            create_ddpg_agent(
                mdp, i, n_features_actor, lr_actor, n_features_critic, lr_critic, batch_size,
                initial_replay_size, max_replay_size, tau, sigma_checkpoints, decay_type, sigma, target_sigma,
                sigma_transition_length, True, save_path
            )
        )

    for a in agents:
        a.agents = agents

    return agents


def run_ddpg_training(
        core,
        n_epochs,
        n_steps_learn=600,
        n_steps_test=400,
        n_steps_per_fit=1,
        initial_replay_size=500,
        sigma=0.2,
        target_sigma=0.001,
        gamma_eval=1,
        disable_noise_for_evaluation=True,
        save=False,
        save_every=5,
        save_postfix=""
):
    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)

    # ------------- RUN EXPERIMENT ------------- #
    dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
    data = []

    metrics = compute_metrics(dataset, gamma_eval)
    data.append(metrics)

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(n_steps=n_steps_learn, n_steps_per_fit=n_steps_per_fit, quiet=True)

        if disable_noise_for_evaluation:
            dataset = evaluate_without_noise(core, n_steps_test)
        else:
            dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)

        metrics = compute_metrics(dataset, gamma_eval)
        data.append(metrics)
        pbar.set_postfix(
            MinMaxMean=np.round(metrics[0:3], 2),
            sigma=np.round(core.agent.policy._sigma, 2)
        )

        if (n + 1) % save_every == 0 and save:
            core.agent.save(f"weights/ddpg_agent_epo_{n + 1}_{save_postfix}")

        core.agent.policy._sigma *= (target_sigma / sigma) ** (1 / n_epochs)

    pbar.close()
    return np.array(data)


def evaluate_without_noise(core, n_steps_test):
    tmp = core.agent.policy._sigma
    core.agent.policy._sigma = 0.0
    dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
    core.agent.policy._sigma = tmp
    return dataset
