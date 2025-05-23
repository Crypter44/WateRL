from copy import deepcopy

import numpy as np
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import to_parameter

from Mushroom.environments.mdp_info import MAMDPInfo
from Mushroom.utils.replay_memories import ReplayMemoryObsMultiAgent

class IDDPG(DeepAC):
    """
    Represents the IDDPG (Independent Deep Deterministic Policy Gradient) algorithm.

    This class implements the IDDPG algorithm. It inherits from the
    DeepAC base class from MushroomRl and simply extends the DDPG implementation of MushroomRl to support
    Multi-Agent Systems.

    Attributes:
        agent_idx (int): Index of the agent in the multi-agent system.
        _actor_params (dict): Parameters for the actor network.
        _actor_optimizer_params (dict): Parameters for the actor optimizer.
        _critic_params (dict): Parameters for the critic network.
        _critic_fit_params (dict): Parameters for fitting the critic.
        _actor_predict_params (dict): Parameters for actor predictions.
        _critic_predict_params (dict): Parameters for critic predictions.
        _batch_size (Parameter): Batch size of the training dataset.
        _tau (Parameter): Soft update coefficient for target networks.
        _critic_tau (Parameter or None): Custom soft update coefficient for critic (if any).
        _policy_delay (Parameter): Delay frequency for policy updates.
        _fit_count (int): Counter for the number of training iterations.
        _initial_replay_size (int): Minimum number of samples in the replay memory before training.
        _replay_memory (ReplayMemoryObsMultiAgent): Replay buffer for storing experiences.
        _critic_approximator (Regressor): Critic network regressor.
        _target_critic_approximator (Regressor): Target critic network regressor.
        _actor_approximator (Regressor): Actor network regressor.
        _target_actor_approximator (Regressor): Target actor network regressor.
        _debug_logging (bool): Flag for enabling debug information logging.
        _debug_info (dict): Debug metrics recorded during training.
    """
    def __init__(
            self,
            agent_idx: int,
            mdp_info: MAMDPInfo,
            policy_class, policy_params,
            actor_params, actor_optimizer,
            critic_params,
            batch_size,
            initial_replay_size, max_replay_size,
            tau,
            policy_delay=1,
            critic_fit_params=None,
            actor_predict_params=None,
            critic_predict_params=None,
            replay_memory=None
    ):
        self.reset_params = {
            "agent_idx": agent_idx,
            "mdp_info": mdp_info,
            "policy_class": policy_class,
            "policy_params": policy_params,
            "actor_params": actor_params,
            "actor_optimizer": actor_optimizer,
            "critic_params": critic_params,
            "batch_size": batch_size,
            "initial_replay_size": initial_replay_size,
            "max_replay_size": max_replay_size,
            "tau": tau,
            "policy_delay": policy_delay,
            "critic_fit_params": critic_fit_params,
            "actor_predict_params": actor_predict_params,
            "critic_predict_params": critic_predict_params
        }

        self.agent_idx = agent_idx
        self._actor_params = actor_params
        self._actor_optimizer_params = actor_optimizer
        self._critic_params = critic_params
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._critic_tau = None
        self._policy_delay = to_parameter(policy_delay)
        self._fit_count = 0

        self._initial_replay_size = initial_replay_size
        self._replay_memory = ReplayMemoryObsMultiAgent(
            max_replay_size,
            mdp_info.state_space.shape[0],
            [mdp_info.observation_space_for_idx(i).shape[0] for i in range(mdp_info.n_agents)],
            [mdp_info.action_space_for_idx(i).shape[0] for i in range(mdp_info.n_agents)],
            mdp_info.n_agents,
            False
        ) if replay_memory is None else replay_memory

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)
        self._init_target(self._actor_approximator,
                          self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._debug_logging = True
        self._debug_info = {
            "actor_loss": [],
            "critic_loss": [],
            "actor_grad_norm": [],
            "critic_grad_norm": [],
            "rewards": [],
            "q_next": [],
            "q_target": []
        }

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='pickle',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_actor_approximator='mushroom'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.size > self._initial_replay_size:
            states, obs, actions, rewards, next_states, next_obs, absorbing, last = \
                self._replay_memory.get(self._batch_size())

            q_next = self._next_q(next_states, next_obs, absorbing)
            q_target = rewards[:, self.agent_idx] + self.mdp_info.gamma * q_next

            critic_loss = self._fit_critic(states, obs, actions, q_target)
            critic_grad_norm = self.critic_grad_norm()

            if self._fit_count % self._policy_delay() == 0:
                loss = self._loss(states, obs)
                self._optimize_actor_parameters(loss)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator,
                                self._critic_tau if self._critic_tau is not None else self._tau())

            self._update_target(self._actor_approximator,
                                self._target_actor_approximator,
                                self._tau())

            if self._debug_logging:
                self._debug_info["q_next"].append(q_next.mean().item())
                self._debug_info["q_target"].append(q_target.mean().item())
                self._debug_info["rewards"].append(rewards[:, self.agent_idx].mean().item())
                self._debug_info["critic_grad_norm"].append(critic_grad_norm)
                self._debug_info["critic_loss"].append(critic_loss)
                if self._fit_count % self._policy_delay() == 0:
                    self._debug_info["actor_loss"].append(loss.item())
                    self._debug_info["actor_grad_norm"].append(self.actor_grad_norm())

            self._fit_count += 1

        return -1, -1

    def _loss(self, states, obs):
        obs = obs[self.agent_idx]
        action = self._actor_approximator(obs, output_tensor=True, **self._actor_predict_params)
        q = self._critic_approximator(obs, action, output_tensor=True, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, next_states, next_obs, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        next_obs = next_obs[self.agent_idx]
        a = self._target_actor_approximator.predict(next_obs, **self._actor_predict_params)

        q = self._target_critic_approximator.predict(next_obs, a, **self._critic_predict_params)
        q *= 1 - absorbing

        return q

    def _optimize_actor_parameters(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._clip_gradient()
        self._optimizer.step()

    def _fit_critic(self, states, obs, actions, q):
        self._critic_approximator.fit(obs[self.agent_idx], actions[self.agent_idx], q,
                                      **self._critic_fit_params)
        return self._critic_approximator.model._last_loss

    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())

    def actor_grad_norm(self):
        total_norm = 0.0
        for layer in self._actor_approximator.model.network.parameters():
            layer_norm = layer.grad.data.norm(2)
            total_norm += layer_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def critic_grad_norm(self):
        total_norm = 0.0
        for layer in self._critic_approximator.model.network.parameters():
            layer_norm = layer.grad.data.norm(2)
            total_norm += layer_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def set_debug_logging(self, enabled):
        self._debug_logging = enabled

    def get_debug_info(self, previous_info=None, entries_as_list=True):
        averaged_info = {
            key: [np.mean(value).item()] for key, value in self._debug_info.items()
        }
        if previous_info is not None:
            for key, _ in previous_info.items():
                previous_info[key] += averaged_info[key]
            return previous_info

        if not entries_as_list:
            return {key: value[0] for key, value in averaged_info.items()}
        return averaged_info

    def _update_target(self, online, target, tau):
        for i in range(len(target)):
            weights = tau * online[i].get_weights()
            weights += (1 - tau) * target[i].get_weights()
            target[i].set_weights(weights)

    def reset_optimizers(self):
        self._critic_approximator.model._optimizer = self._critic_params['optimizer']['class'](
            self._critic_approximator.model.network.parameters(), **self._critic_params['optimizer']['params']
        )
        self._optimizer = self._actor_optimizer_params['class'](self._actor_approximator.model.network.parameters(),
                                                                **self._actor_optimizer_params['params'])
