from copy import deepcopy
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Agent


class UnifiedCriticMADDPG(Agent):
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
            grad_norm_clip,
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
        self._grad_norm_clip = grad_norm_clip
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
        self.critic_params = self.critic_approximator.network.parameters()

        self._actor_optimizer = actor_optimizer_params["class"](
            self.actor_params, **actor_optimizer_params["params"]
        )
        self._critic_optimizer = self.critic_approximator._optimizer

        self.update_target_critic_hard()

        self._debug_logging = True
        self._debug_info = {
            "actor_loss": [],
            "critic_loss": [],
            "actor_grad_norm": [],
            "actor_grad_norm_clipped": [],
            "critic_grad_norm": [],
            "critic_grad_norm_clipped": [],
            "q_hat": [],
            "q_target": [],
            "q_next": [],
        }

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

            # move to cuda if necessary
            if self._use_cuda:
                states_t = states_t.cuda()
                obs_t = [obs.cuda() for obs in obs_t]
                actions_t = [actions.cuda() for actions in actions_t]
                rewards_t = rewards_t.cuda()
                next_states_t = next_states_t.cuda()
                next_obs_t = [obs.cuda() for obs in next_obs_t]
                absorbing_t = absorbing_t.cuda()

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
            critic_grad_norm = self.critic_grad_norm()
            if self._grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.critic_params, self._grad_norm_clip
                )
            critic_grad_norm_clipped = self.critic_grad_norm()
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
            actor_grad_norm = self.actor_grad_norm()
            if self._grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.actor_params, self._grad_norm_clip
                )
            actor_grad_norm_clipped = self.actor_grad_norm()
            self._actor_optimizer.step()

            # Update target mixer
            self._n_updates += 1
            if self._target_update_mode == "soft":
                self.update_target_critic_soft()
            elif self._target_update_mode == "hard":
                if self._n_updates % self._target_update_frequency == 0:
                    self.update_target_critic_hard()

            if self._debug_logging:
                self._debug_info["actor_loss"].append(actor_loss.mean().item())
                self._debug_info["critic_loss"].append(critic_loss.mean().item())
                self._debug_info["actor_grad_norm"].append(actor_grad_norm)
                self._debug_info["actor_grad_norm_clipped"].append(actor_grad_norm_clipped)
                self._debug_info["critic_grad_norm"].append(critic_grad_norm)
                self._debug_info["critic_grad_norm_clipped"].append(critic_grad_norm_clipped)
                self._debug_info["q_hat"].append(q_hat.mean().item())
                self._debug_info["q_target"].append(q_target.mean().item())
                self._debug_info["q_next"].append(q_next.mean().item())

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

    def actor_grad_norm(self):
        total_norm = 0.0
        for layer in self.actor_params:
            layer_norm = layer.grad.data.norm(2)
            total_norm += layer_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def critic_grad_norm(self):
        total_norm = 0.0
        for layer in self.critic_params:
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
