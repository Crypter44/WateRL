import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.utils.dataset import compute_metrics
from torch import nn, optim
from tqdm import tqdm

from Mushroom.agents.sigma_decay_policies import OUPolicyWithNoiseDecay, UnivariateGaussianPolicy


# Define the neural networks for the actor and the critic
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, agent_idx=-1, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._agent_idx = agent_idx
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
            state = state[self._agent_idx]
            action = action[self._agent_idx]
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, agent_idx=-1, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._agent_idx = agent_idx
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
        if self._agent_idx != -1:
            state = state[self._agent_idx]
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = F.sigmoid(self._h3(features2))
        a = a * (self.mdp.info.action_space.high - self.mdp.info.action_space.low) + self.mdp.info.action_space.low

        return a


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
        sigma=0.2,
        target_sigma=0.001,
        sigma_transition_length=1,
        theta=0.15,
        dt=1e-2
):
    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features_actor,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        agent_idx=agent_idx)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}  # not so big of a difference to critic

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         agent_idx=agent_idx)

    # Policy
    # policy_class = OUPolicyWithNoiseDecay
    # policy_params = dict(initial_sigma=sigma, target_sigma=target_sigma,
    #                      updates_till_target_reached=sigma_transition_length,
    #                      theta=theta, dt=dt)

    policy_class = UnivariateGaussianPolicy
    policy_params = dict(initial_sigma=sigma, target_sigma=target_sigma,
                         updates_till_target_reached=sigma_transition_length)

    # Agent
    return DDPG(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params,
                batch_size, initial_replay_size, max_replay_size,
                tau)


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

        core.agent.policy._sigma *= (target_sigma/sigma) ** (1/n_epochs)

    pbar.close()
    return np.array(data)


def evaluate_without_noise(core, n_steps_test):
    tmp = core.agent.policy._sigma
    core.agent.policy._sigma = 0.0
    dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
    core.agent.policy._sigma = tmp
    return dataset
