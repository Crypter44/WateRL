from copy import deepcopy

import torch
import torch.nn.functional as F
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Agent
from torch import optim

from Mushroom.agents.ddpg import CriticNetwork, ActorNetwork
from Mushroom.agents.sigma_decay_policies import UnivariateGaussianPolicy
from Mushroom.utils.replay_memories import ReplayMemoryObs


class DDPG(Agent):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

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
            use_mixer=True,
    ):
        """
        Constructor.

        """
        super().__init__(mdp_info, policy)

        self._idx_agent = idx_agent

        self._batch_size = batch_size
        self._target_update_frequency = target_update_frequency
        self._tau = tau
        self._warmup_replay_size = warmup_replay_size
        self._use_mixer = use_mixer
        self._use_cuda = use_cuda

        self._replay_memory = replay_memory

        self._n_updates = 0

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
            action_mask (np.ndarray, None): the mask for the actions.

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

            # Critic update
            q_hat = self.critic_approximator.predict(
                obs_t, actions_t, output_tensor=True
            )
            q_next = self._next_q(next_obs_t)
            q_target = (
                    rewards_t + self.mdp_info.gamma * q_next * ~absorbing_t
            ).detach()
            critic_loss = self.critic_approximator._loss(q_hat, q_target)
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
            next_state_t (torch.Tensor): the states where next action has to be
                evaluated;

        Returns:
            Action-values returned by the critic for ``next_state_t`` and the
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


def setup_iddpg_agents(
        mdp,
        n_agents,
        policy=None,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=2e-4,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma=0.2,
        target_sigma=0.001,
        sigma_transition_length=1,
):
    agents = []
    for i in range(n_agents):
        actor_input_shape = mdp.info.observation_space_for_idx(i).shape
        actor_params = dict(
            network=ActorNetwork,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr_actor}
            },
            n_features=n_features_actor,
            input_shape=actor_input_shape,
            output_shape=mdp.info.action_space_for_idx(i).shape,
            agent_idx=i,
            use_cuda=True,
        )

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space_for_idx(i).shape[0],)
        critic_params = dict(
            network=CriticNetwork,
            optimizer={'class': optim.Adam,
                       'params': {'lr': lr_critic}},
            loss=F.mse_loss,
            n_features=n_features_critic,
            input_shape=critic_input_shape,
            output_shape=(1,),
            agent_idx=i,
            use_cuda=True
        )

        if policy is None:
            policy = UnivariateGaussianPolicy(
                initial_sigma=sigma,
                target_sigma=target_sigma,
                updates_till_target_reached=sigma_transition_length
            )

        agents.append(
            DDPG(
                mdp.info,
                i,
                policy,
                actor_params,
                critic_params,
                batch_size,
                target_update_frequency=-1,
                tau=tau,
                warmup_replay_size=initial_replay_size,
                replay_memory=ReplayMemoryObs(
                    max_replay_size,
                    mdp.info.state_space.shape[0],
                    mdp.info.observation_space_for_idx(i).shape[0],
                    mdp.info.action_space_for_idx(i).shape[0],
                    False
                ),
                use_cuda=True,
                use_mixer=True,
            )
        )

    return agents
