from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.core import Agent
from mushroom_rl.utils.torch import get_weights, set_weights

from Mushroom.agents.qmix import QMixer


class FACMAC(Agent):
    """
    Instantiates a FACMAC mixing network and hypernetwork layers.
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
            mixing_embed_dim,
            actor_optimizer_params,
            critic_optimizer_params,
            scale_critic_loss,
            scale_actor_loss,
            grad_norm_clip,
            obs_last_action,
            host_agents,
            use_cuda=True,
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
        self._grad_norm_clip = grad_norm_clip
        self._obs_last_action = obs_last_action
        self._host_agents = host_agents  # The agents using this mixing network
        self._use_cuda = use_cuda

        self._n_updates = 0

        self._state_shape_int = int(np.prod(self.mdp_info.state_space.shape))

        self._mixer = QMixer(
            state_shape=mdp_info.state_space.shape,
            mixing_embed_dim=mixing_embed_dim,
            n_agents=mdp_info.n_agents,
            use_cuda=use_cuda,
        )
        self._target_mixer = QMixer(
            state_shape=mdp_info.state_space.shape,
            mixing_embed_dim=mixing_embed_dim,
            n_agents=mdp_info.n_agents,
            use_cuda=use_cuda,
        )

        self.actor_params = list(
            chain(
                *[
                    agent.actor_approximator.network.parameters()
                    for agent in host_agents
                ]
            )
        )
        self.critic_params = list(
            chain(
                *[
                    agent.critic_approximator.network.parameters()
                    for agent in host_agents
                ],
                self._mixer.parameters(),
            )
        )

        self._actor_optimizer_params = actor_optimizer_params
        self._critic_optimizer_params = critic_optimizer_params
        self.reset_optimizers()

        self.update_target_mixer()

        self._debug_logging = True
        self._debug_info = {
            "actor_loss": [],
            "critic_loss": [],
            "actor_grad_norm": [],
            "actor_grad_norm_clipped": [],
            "critic_grad_norm": [],
            "critic_grad_norm_clipped": [],
            "q_tot": [],
            "q_tot_target": [],
            "q_tot_next": [],
            "q_hats": [],
            "q_nexts": [],
        }

        self._add_save_attr(
            _batch_size="primitive",
            _target_update_frequency="primitive",
            _tau="primitive",
            _warmup_replay_size="primitive",
            _replay_memory="mushroom!",
            _n_updates="primitive",
            _mixer="torch",
            _target_mixer="torch",
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
            rewards_t = torch.tensor(rewards[:, 0], dtype=torch.float32).unsqueeze(-1)
            next_states_t = torch.tensor(next_states, dtype=torch.float32)
            next_obs_t = [
                torch.tensor(next_obs[idx_agent], dtype=torch.float32)
                for idx_agent in range(len(obs))
            ]
            absorbing_t = torch.tensor(absorbing, dtype=torch.bool).unsqueeze(-1)

            # move to cuda if needed
            if self._use_cuda:
                states_t = states_t.cuda()
                obs_t = [obs.cuda() for obs in obs_t]
                actions_t = [actions.cuda() for actions in actions_t]
                rewards_t = rewards_t.cuda()
                next_states_t = next_states_t.cuda()
                next_obs_t = [next_obs.cuda() for next_obs in next_obs_t]
                absorbing_t = absorbing_t.cuda()

            # Buffers for calculating q_hats, q_nexts, actor actions for updates
            q_hats = []
            q_nexts = []
            actor_update_actions = []
            for idx_agent, agent in enumerate(self._host_agents):
                q_hat = agent.critic_approximator.predict(
                    obs_t[idx_agent], actions_t[idx_agent], output_tensor=True
                )
                target_action = agent._draw_target_action(next_obs_t[idx_agent])
                q_next = agent.target_critic_approximator.predict(
                    next_obs_t[idx_agent], target_action, output_tensor=True
                )

                q_hats.append(q_hat)
                q_nexts.append(q_next)

                # Get the actions for actor backprop
                actor_update_actions.append(
                    agent.actor_approximator.predict(
                        obs_t[idx_agent], output_tensor=True
                    )
                )

            # Compute mixer predictions
            q_hat = torch.stack(q_hats, dim=-1).unsqueeze(-1)
            q_next = torch.stack(q_nexts, dim=-1).unsqueeze(-1)

            q_tot = self.mix(q_hat, states_t.reshape(-1, self._state_shape_int))
            q_tot_next = self.target_mix(
                q_next, next_states_t.reshape(-1, self._state_shape_int)
            )
            q_tot_target = (
                    rewards_t + self.mdp_info.gamma * q_tot_next * ~absorbing_t
            ).detach()

            if self._debug_logging:
                self._debug_info["q_tot"].append(q_tot.mean().item())
                self._debug_info["q_tot_target"].append(q_tot_target.mean().item())
                self._debug_info["q_tot_next"].append(q_tot_next.mean().item())
                self._debug_info["q_hats"].append(q_hat.mean().item())
                self._debug_info["q_nexts"].append(q_next.mean().item())

            # Compute critic loss and back-propagate
            critic_loss = F.mse_loss(q_tot, q_tot_target)
            if self._scale_critic_loss:
                critic_loss /= self.mdp_info.n_agents
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = self.critic_grad_norm()
            if self._grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.critic_params, self._grad_norm_clip
                ).item()
            critic_grad_norm_clipped = self.critic_grad_norm()
            self._critic_optimizer.step()

            # Compute actor loss
            q_actors = []
            for idx_agent, agent in enumerate(self._host_agents):
                q = agent.critic_approximator.predict(
                    obs_t[idx_agent],
                    actor_update_actions[idx_agent],
                    output_tensor=True,
                )
                q_actors.append(q)
            q_actor = torch.stack(q_actors, dim=-1).unsqueeze(-1)
            q_tot_actor = self.mix(q_actor, states_t.reshape(-1, self._state_shape_int))
            actor_loss = -q_tot_actor.mean()
            if self._scale_actor_loss:
                actor_loss /= self.mdp_info.n_agents
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self.critic_grad_norm()
            actor_grad_norm = self.actor_grad_norm()
            if self._grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.actor_params, self._grad_norm_clip
                ).item()
            self._actor_optimizer.step()

            if self._debug_logging:
                self._debug_info["actor_loss"].append(actor_loss.item())
                self._debug_info["critic_loss"].append(critic_loss.item())
                self._debug_info["actor_grad_norm"].append(actor_grad_norm)
                self._debug_info["actor_grad_norm_clipped"].append(
                    self.actor_grad_norm()
                )
                self._debug_info["critic_grad_norm"].append(critic_grad_norm)
                self._debug_info["critic_grad_norm_clipped"].append(
                    critic_grad_norm_clipped
                )

            # Update target mixer
            self._n_updates += 1
            if self._target_update_mode == "soft":
                self.update_target_mixer_soft()
            elif self._target_update_mode == "hard":
                if self._n_updates % self._target_update_frequency == 0:
                    self.update_target_mixer()

            return actor_loss.item(), critic_loss.item()
        else:
            return 0, 0

    def mix(self, chosen_action_value, state):
        return self._mixer(chosen_action_value, state)

    def target_mix(self, chosen_action_value, state):
        return self._target_mixer(chosen_action_value, state)

    def update_target_mixer(self):
        w = get_weights(self._mixer.parameters())
        set_weights(self._target_mixer.parameters(), w, use_cuda=self._use_cuda)

    def update_target_mixer_soft(self):
        weights = self._tau * self.get_mixer_weights()
        weights += (1 - self._tau) * get_weights(self._target_mixer.parameters())
        set_weights(self._target_mixer.parameters(), weights, use_cuda=self._use_cuda)

    def get_mixer_weights(self):
        return get_weights(self._mixer.parameters())

    def actor_param(self):
        params = torch.cat([param.flatten() for param in self.actor_params])
        return params

    def actor_gradient(self):
        gradient = torch.cat(
            [
                param.grad.view(-1)
                for param in self.actor_params
                if param.grad is not None
            ]
        )
        return gradient

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

    def reset_optimizers(self):
        self._actor_optimizer = self._actor_optimizer_params["class"](
            self.actor_params, **self._actor_optimizer_params["params"]
        )
        self._critic_optimizer = self._critic_optimizer_params["class"](
            self.critic_params, **self._critic_optimizer_params["params"]
        )
