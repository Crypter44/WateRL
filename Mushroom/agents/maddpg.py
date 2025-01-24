import numpy as np
import torch
from copy import deepcopy
from itertools import chain
import torch.nn.functional as F
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Agent
from torch import optim

from Mushroom.agents.ddpg import CriticNetwork, ActorNetwork
from Mushroom.agents.ddpg_with_mixer_support import DDPG
from Mushroom.agents.sigma_decay_policies import UnivariateGaussianPolicy
from Mushroom.utils.replay_memories import ReplayMemoryObsMultiAgent


class MADDPG(Agent):
    """
    MADDPG Discrete Agent with a centralised critic common to all agents.
    """

    def __init__(
        self,
        mdp_info,
        batch_size,
        replay_memory,
        target_update_frequency,
        tau,
        warmup_replay_size,
        target_update_mode,
        actor_optimizer_params,
        critic_params,
        scale_critic_loss,
        scale_actor_loss,
        obs_last_action,
        host_agents,
        use_cuda=False,
    ):
        super().__init__(mdp_info, policy=None)

        self._batch_size = batch_size
        self._replay_memory = replay_memory
        self._target_update_frequency = target_update_frequency
        self._tau = tau
        self._warmup_replay_size = warmup_replay_size
        self._target_update_mode = target_update_mode
        self._scale_critic_loss = scale_critic_loss
        self._scale_actor_loss = scale_actor_loss
        self._obs_last_action = obs_last_action
        self._host_agents = host_agents  # The agents using this mixing network
        self._use_cuda = use_cuda

        self._n_updates = 0


        target_critic_params = deepcopy(critic_params)
        self.critic_approximator = TorchApproximator(**critic_params)
        self.target_critic_approximator = TorchApproximator(**target_critic_params)

        self.actor_params = list(
            chain(
                *[
                    agent.actor_approximator.network.parameters()
                    for agent in host_agents
                ]
            )
        )

        self._actor_optimizer = actor_optimizer_params["class"](
            self.actor_params, **actor_optimizer_params["params"]
        )
        self._critic_optimizer = self.critic_approximator._optimizer

        self.update_target_critic_hard()

        self._add_save_attr(
            _batch_size="primitive",
            _target_update_frequency="primitive",
            _tau="primitive",
            _warmup_replay_size="primitive",
            _replay_memory="mushroom!",
            _n_updates="primitive",
            critic_approximator="mushroom",
            target_critic_approximator="mushroom",
            _target_critic="torch",
            _actor_optimizer="torch",
            _critic_optimizer="torch",
            _use_cuda="primitive",
        )

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.size > self._warmup_replay_size:
            states, obs, actions, rewards, next_states, next_obs, absorbing, _ = (
                self._replay_memory.get(self._batch_size)
            )

            # Convert to tensors
            # Use rewards of agent 0, assume all agents have the same reward
            states_t = torch.tensor(states, dtype=torch.float32)
            obs_t = [
                torch.tensor(obs[idx_agent], dtype=torch.float32)
                for idx_agent in range(len(obs))
            ]
            actions_t = [
                torch.tensor(actions[idx_agent], dtype=torch.float32)
                for idx_agent in range(len(actions))
            ]
            rewards_t = torch.tensor(rewards[:, 0], dtype=torch.float32)
            next_states_t = torch.tensor(next_states, dtype=torch.float32)
            next_obs_t = [
                torch.tensor(next_obs[idx_agent], dtype=torch.float32)
                for idx_agent in range(len(obs))
            ]
            absorbing_t = torch.tensor(absorbing, dtype=torch.bool)

            # Update critic
            actions_cat = torch.cat(actions_t, dim=-1)
            next_actions_t = [
                self._host_agents[idx_agent].target_actor_approximator.predict(
                    next_obs_t[idx_agent], output_tensor=True
                )
                for idx_agent in range(self.mdp_info.n_agents)
            ]
            next_actions_cat = torch.cat(next_actions_t, dim=-1)
            q_hat = self.critic_approximator.predict(
                states_t, actions_cat, output_tensor=True
            )
            q_next = self.target_critic_approximator.predict(
                next_states_t, next_actions_cat, output_tensor=True
            )
            q_target = (
                    rewards_t + self.mdp_info.gamma * q_next * ~absorbing_t
            ).detach()
            critic_loss = F.mse_loss(q_hat, q_target)
            if self._scale_critic_loss:
                critic_loss /= self.mdp_info.n_agents
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._critic_optimizer.step()

            # Update actors
            actions_backprop = [
                self._host_agents[idx_agent].actor_approximator.predict(
                    obs_t[idx_agent], output_tensor=True
                )
                for idx_agent in range(self.mdp_info.n_agents)
            ]
            actions_backprop_cat = torch.cat(actions_backprop, dim=-1)
            q = self.critic_approximator.predict(
                states_t, actions_backprop_cat, output_tensor=True
            )
            actor_loss = -q.mean()
            if self._scale_actor_loss:
                actor_loss /= self.mdp_info.n_agents
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            if self._grad_norm_clip is not None:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_params, self._grad_norm_clip
                )
            self._actor_optimizer.step()

            # Update target mixer
            self._n_updates += 1
            if self._target_update_mode == "soft":
                self.update_target_critic_soft()
            elif self._target_update_mode == "hard":
                if self._n_updates % self._target_update_frequency == 0:
                    self.update_target_critic_hard()

            return actor_loss.item(), critic_loss.item()
        else:
            return 0, 0

    def get_critic_episodes(self, episodes, max_seq_len):
        """
        Global batch information for the critic and agents.
        """
        (
            batch_states,
            batch_rewards,
            batch_next_states,
            batch_absorbings,
            pad_masks,
        ) = ([], [], [], [], [])
        for episode in episodes:
            seq_len = len(episode)

            state_seq = np.array([sample["state"] for sample in episode])
            rewards_seq = np.array(
                [sample["rewards"][0] for sample in episode]
            )  # all agents have same reward, so we just take agent 0's reward
            next_state_seq = np.array([sample["next_state"] for sample in episode])
            absorbing_seq = np.array([sample["absorbing"] for sample in episode])
            mask = np.concatenate([np.ones(seq_len), np.zeros(max_seq_len - seq_len)])

            # Pad to max_seq_len
            state_pad = np.pad(
                state_seq, ((0, max_seq_len - seq_len), (0, 0)), "constant"
            )
            rewards_pad = np.pad(rewards_seq, (0, max_seq_len - seq_len), "constant")
            next_state_pad = np.pad(
                next_state_seq, ((0, max_seq_len - seq_len), (0, 0)), "constant"
            )
            absorbing_pad = np.pad(
                absorbing_seq,
                (0, max_seq_len - seq_len),
                "constant",
                constant_values=1,
            )

            # Append to the batch
            batch_states.append(state_pad)
            batch_rewards.append(rewards_pad)
            batch_next_states.append(next_state_pad)
            batch_absorbings.append(absorbing_pad)
            pad_masks.append(mask)

        # Transpose to [seq_len, batch_size, ...] format
        batch_states = np.array(batch_states).transpose(
            1, 0, 2
        )  # Shape: [seq_len, batch_size, state_dim]
        batch_rewards = np.array(batch_rewards).transpose(
            (1, 0)
        )  # Shape: [seq_len, batch_size]
        batch_next_states = np.array(batch_next_states).transpose(
            1, 0, 2
        )  # Shape: [seq_len, batch_size, state_dim]
        batch_absorbings = np.array(batch_absorbings).transpose(
            (1, 0)
        )  # Shape: [seq_len, batch_size]
        pad_masks = np.array(pad_masks).transpose(
            (1, 0)
        )  # Shape: [seq_len, batch_size]

        # Convert to torch tensors
        batch_states_t = torch.tensor(batch_states, dtype=torch.float32)
        batch_rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_next_states_t = torch.tensor(batch_next_states, dtype=torch.float32)
        batch_absorbings_t = torch.tensor(batch_absorbings, dtype=torch.bool)
        pad_masks_t = torch.tensor(pad_masks, dtype=torch.bool)

        return (
            batch_states_t,
            batch_rewards_t,
            batch_next_states_t,
            batch_absorbings_t,
            pad_masks_t,
        )

    def get_agent_episodes(self, episodes, idx_agent, max_seq_len):
        """
        Agent-specific episodes for the agent with index idx_agent.
        """
        # prepare data arrays
        (
            batch_obs,
            batch_action_masks,
            batch_actions,
            batch_next_obs,
            batch_next_action_masks,
        ) = ([], [], [], [], [])

        for episode in episodes:
            seq_len = len(episode)

            # Prepare the data arrays
            obs_seq = np.array([sample["obs"][idx_agent] for sample in episode])
            action_mask_seq = np.array(
                [sample["action_masks"][idx_agent] for sample in episode]
            )
            actions_seq = np.array([sample["actions"][idx_agent] for sample in episode])
            actions_seq_one_hot = np.eye(self.mdp_info.action_space[self._idx_agent].n)[
                actions_seq
            ].squeeze(1)
            next_obs_seq = np.array(
                [sample["next_obs"][idx_agent] for sample in episode]
            )
            next_action_masks_seq = np.array(
                [sample["next_action_masks"][idx_agent] for sample in episode]
            )

            # Pad to max_seq_len
            obs_pad = np.pad(obs_seq, ((0, max_seq_len - seq_len), (0, 0)), "constant")
            action_masks_pad = np.pad(
                action_mask_seq,
                ((0, max_seq_len - seq_len), (0, 0)),
                "constant",
                constant_values=1,
            )
            actions_one_hot_pad = np.pad(
                actions_seq_one_hot,
                ((0, max_seq_len - seq_len), (0, 0)),
                "constant",
            )
            next_obs_pad = np.pad(
                next_obs_seq, ((0, max_seq_len - seq_len), (0, 0)), "constant"
            )
            next_action_masks_pad = np.pad(
                next_action_masks_seq,
                ((0, max_seq_len - seq_len), (0, 0)),
                "constant",
                constant_values=1,
            )

            # Append to the batch
            batch_obs.append(obs_pad)
            batch_action_masks.append(action_masks_pad)
            batch_actions.append(actions_one_hot_pad)
            batch_next_obs.append(next_obs_pad)
            batch_next_action_masks.append(next_action_masks_pad)

        # Converts lists to numpy arrays with shape [seq_len, batch_size, ...]
        batch_obs = np.array(batch_obs).transpose(
            (1, 0, 2)
        )  # Shape: [seq_len, batch_size, obs_dim]
        batch_action_masks = np.array(batch_action_masks).transpose(
            (1, 0, 2)
        )  # Shape: [seq_len, batch_size, action_dim]
        batch_actions = np.array(batch_actions).transpose(
            (1, 0, 2)
        )  # Shape: [seq_len, batch_size, action_dim]
        batch_next_obs = np.array(batch_next_obs).transpose(
            (1, 0, 2)
        )  # Shape: [seq_len, batch_size, obs_dim]
        batch_next_action_masks = np.array(batch_next_action_masks).transpose(
            (1, 0, 2)
        )  # Shape: [seq_len, batch_size, action_dim]

        # Convert to torch tensors
        batch_obs_t = torch.tensor(batch_obs, dtype=torch.float32)
        batch_action_masks_t = torch.tensor(batch_action_masks, dtype=torch.bool)
        batch_actions_t = torch.tensor(batch_actions, dtype=torch.long)
        batch_next_obs_t = torch.tensor(batch_next_obs, dtype=torch.float32)
        batch_next_action_masks_t = torch.tensor(
            batch_next_action_masks, dtype=torch.bool
        )

        return (
            batch_obs_t,
            batch_action_masks_t,
            batch_actions_t,
            batch_next_obs_t,
            batch_next_action_masks_t,
        )

    def update_target_critic_soft(self):
        weights = self._tau * self.critic_approximator.get_weights()
        weights += (1 - self._tau) * self.target_critic_approximator.get_weights()
        self.target_critic_approximator.set_weights(weights)

    def update_target_critic_hard(self):
        self.target_critic_approximator.set_weights(
            self.critic_approximator.get_weights()
        )


def setup_maddpg_agents(
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
                replay_memory=None,
                use_cuda=True,
                use_mixer=True,
            )
        )

    maddpg = MADDPG(
        mdp_info=mdp.info,
        batch_size=batch_size,
        replay_memory=ReplayMemoryObsMultiAgent(
            max_size=max_replay_size,
            state_dim=mdp.info.state_space.shape[0],
            obs_dim=[mdp.info.observation_space_for_idx(i).shape[0] for i in range(n_agents)],
            action_dim=[mdp.info.action_space_for_idx(i).shape[0] for i in range(n_agents)],
            n_agents=n_agents,
            discrete_actions=False,
        ),
        target_update_frequency=-1,
        tau=tau,
        warmup_replay_size=initial_replay_size,
        target_update_mode="soft",
        actor_optimizer_params={
            "class": optim.Adam,
            "params": {"lr": lr_actor},
        },
        critic_params={
            "network": CriticNetwork,
            "optimizer": {"class": optim.Adam, "params": {"lr": lr_critic}},
            "loss": F.mse_loss,
            "n_features": n_features_critic,
            "input_shape": (n_features_actor * n_agents + n_agents,),
            "output_shape": (1,),
            "use_cuda": True,
        },
        scale_critic_loss=False,
        scale_actor_loss=False,
        obs_last_action=False,
        host_agents=agents,
        use_cuda=True,
    )

    return agents, maddpg