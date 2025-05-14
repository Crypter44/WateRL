from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from mushroom_rl.approximators.parametric import TorchApproximator
from torch import optim

from Mushroom.agents.networks import CriticNetwork, ActorNetwork
from Mushroom.agents.sigma_decay_policies import UnivariateGaussianPolicy
from Mushroom.utils.replay_memories import ReplayMemoryObs
from aryaman import Agent, GaussianPolicy


class MixerDDPG(Agent):
    """
    This MixerDDPG implementation is used for multi-agent environments with centralized training.

    This agent supports the use of a mixer for the critic network. In this case, the fit method
    should be called on the mixer agent. The MixerDDPG will only be used to update the target networks.

    Attributes:
        _batch_size (int): The size of the mini-batches sampled from
            the replay memory.
        _target_update_frequency (int): Frequency of target network updates.
        _tau (float): Soft update parameter to control the interpolation
            between target and online network weights during updates.
        _warmup_replay_size (int): Minimum size of replay memory required
            before learning starts.
        _use_mixer (bool): Defines whether a mixer is used.
        _use_cuda (bool): Indicates whether CUDA is used for training.
        _replay_memory: Memory to store experiences sampled for training.
        _n_updates (int): Counter for the number of updates performed.
        _primary_agent: Reference to the primary agent in multi-agent setups.
        actor_approximator: Neural network representing the actor.
        target_actor_approximator: Copy of the actor used for stable training.
        critic_approximator: Neural network representing the critic.
        target_critic_approximator: Copy of the critic used for stable training.
        _optimizer: Optimizer used for updating the actor network.
    """

    def __init__(
            self,
            mdp_info,
            idx_agent,
            policy,
            actor_params,
            critic_params,
            batch_size,
            target_update_frequency,
            tau,
            warmup_replay_size,
            replay_memory,
            use_cuda,
            primary_agent,
            use_mixer,
    ):
        """
        Constructor.

        """
        super().__init__(mdp_info, policy, idx_agent)

        self._batch_size = batch_size
        self._target_update_frequency = target_update_frequency
        self._tau = tau
        self._warmup_replay_size = warmup_replay_size
        self._use_mixer = use_mixer
        self._use_cuda = use_cuda

        self._replay_memory = replay_memory

        self._n_updates = 0

        self._primary_agent = primary_agent

        target_actor_params = deepcopy(actor_params)
        self.actor_approximator = TorchApproximator(**actor_params)
        self.target_actor_approximator = TorchApproximator(**target_actor_params)
        target_critic_params = deepcopy(critic_params)
        self.critic_approximator = TorchApproximator(**critic_params)
        self.target_critic_approximator = TorchApproximator(**target_critic_params)
        self._update_targets_hard()
        self.policy.set_approximator(self.actor_approximator)
        self._optimizer = self.actor_approximator._optimizer

        self._add_save_attr(
            _batch_size="primitive",
            _target_update_frequency="primitive",
            _tau="primitive",
            _warmup_replay_size="primitive",
            _replay_memory="mushroom!",
            _n_updates="primitive",
            actor_approximator="mushroom",
            target_actor_approximator="mushroom",
            critic_approximator="mushroom",
            target_critic_approximator="mushroom",
            _optimizer="torch",
            _use_mixer="primitive",
            _use_cuda="primitive",
        )

    def draw_action(self, state, action_mask=None):
        """
        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """

        return self.policy.draw_action(state)

    def fit(self, dataset, **info):
        if self._use_mixer:
            actor_loss, critic_loss = 0, 0  # storage and fitting handled by mixer
        else:
            own_dataset = self.split_dataset(dataset)
            self._replay_memory.add(own_dataset)
            actor_loss, critic_loss = self._fit()

        self._n_updates += 1
        if self._idx_agent == 0 or self._primary_agent is None:
            self._update_targets_soft()

        return actor_loss, critic_loss

    def split_dataset(self, dataset):
        own_dataset = list()
        for sample in dataset:
            own_sample = {
                "state": sample["state"],
                "obs": sample["obs"][self._idx_agent],
                "action": sample["actions"][self._idx_agent],
                "reward": sample["rewards"][self._idx_agent],
                "next_state": sample["next_state"],
                "next_obs": sample["next_obs"][self._idx_agent],
                "absorbing": sample["absorbing"],
                "last": sample["last"],
            }
            own_dataset.append(own_sample)
        return own_dataset

    def _fit(self):
        if self._replay_memory.size > self._warmup_replay_size:
            _, obs, actions, rewards, _, next_obs, absorbing, _ = (
                self._replay_memory.get(self._batch_size)
            )

            # Convert to torch tensors
            obs_t = torch.tensor(obs, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.float32)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
            absorbing_t = torch.tensor(absorbing, dtype=torch.bool)

            # move to cuda if needed
            if self._use_cuda:
                obs_t = obs_t.cuda()
                actions_t = actions_t.cuda()
                rewards_t = rewards_t.cuda()
                next_obs_t = next_obs_t.cuda()
                absorbing_t = absorbing_t.cuda()

            # Critic update
            q_hat = self.critic_approximator.predict(
                obs_t, actions_t, output_tensor=True
            )
            q_next = self._next_q(next_obs_t)
            q_target = (
                    rewards_t + self.mdp_info.gamma * q_next * ~absorbing_t
            ).detach()
            critic_loss = self.critic_approximator._loss(q_hat, q_target)
            self.critic_approximator._optimizer.zero_grad()
            critic_loss.backward()
            self.critic_approximator._optimizer.step()

            # Actor update
            loss = self._loss(obs_t)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            return loss.item(), critic_loss.item()
        else:
            return 0, 0

    def _loss(self, state):
        action = self.actor_approximator.predict(state, output_tensor=True)
        q = self.critic_approximator.predict(state, action, output_tensor=True)
        return -q.mean()

    def _draw_target_action(self, next_state_t):
        """
        Draw an action from the target actor without noise.

        Args:
            next_state_t (torch.Tensor): the state where the action is drawn.

        Returns:
            next_mu (torch.Tensor): the greedy action drawn from the target actor network.
        """

        mu_target = self.target_actor_approximator.predict(
            next_state_t, output_tensor=True
        )

        return mu_target

    def _next_q(self, next_state_t):
        """
        Args:
            next_state (torch.Tensor): the states where next action has to be
                evaluated;
            absorbing (torch.Tensor): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a_next_t = self._draw_target_action(next_state_t)
        q_next = self.target_critic_approximator.predict(
            next_state_t, a_next_t, output_tensor=True
        )

        return q_next

    def _update_targets_soft(self):
        """
        Update the target network.

        """
        self._update_target_soft(
            self.actor_approximator, self.target_actor_approximator
        )
        self._update_target_soft(
            self.critic_approximator, self.target_critic_approximator
        )

    def _update_targets_hard(self):
        """
        Update the target network.

        """
        self._update_target_hard(
            self.actor_approximator, self.target_actor_approximator
        )
        self._update_target_hard(
            self.critic_approximator, self.target_critic_approximator
        )

    def _update_target_soft(self, online, target):
        weights = self._tau * online.get_weights()
        weights += (1 - self._tau) * target.get_weights()
        target.set_weights(weights)

    def _update_target_hard(self, online, target):
        target.set_weights(online.get_weights())

    def get_debug_info(self, **kwargs):
        return {}
