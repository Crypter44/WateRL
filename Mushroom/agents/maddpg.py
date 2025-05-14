from typing import List, Optional

import numpy as np
import torch

from Mushroom.agents.iddpg import IDDPG
from Mushroom.environments.mdp_info import MAMDPInfo


class MADDPG(IDDPG):
    """
    A specialized agent class implementing MADDPG (Multi-Agent Deep Deterministic Policy Gradient).

    The MADDPG class extends the IDDPG class for multi-agent environments.
    It features a shared critic for each agent, which is trained on the global state and actions.

    Attributes:
    agent_idx (int): An index representing the current agent's position among all agents.
    agents (Optional[List[MADDPG]]): List of other agents in the environment, including itself.
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
            critic_predict_params=None
    ):
        self.agents: Optional[List[MADDPG]] = None
        self.agent_idx = agent_idx
        super().__init__(
            agent_idx, mdp_info, policy_class, policy_params, actor_params, actor_optimizer, critic_params, batch_size,
            initial_replay_size, max_replay_size, tau, policy_delay, critic_fit_params, actor_predict_params,
            critic_predict_params
        )

    def _next_q(self, next_states, next_obs, absorbing):
        q = self._target_critic_approximator.predict(
            next_states,
            self._draw_actions(next_states, next_obs),
            **self._critic_predict_params
        )
        q *= 1 - absorbing

        return q

    def _loss(self, states, obs):
        q = self._critic_approximator(
            states,
            self._draw_actions(states, obs),
            output_tensor=True,
            **self._critic_predict_params
        )

        return -q.mean()

    def _draw_actions(self, states, obs):
        actions = []
        for agent in self.agents:
            action = agent._actor_approximator(obs[agent.agent_idx], output_tensor=True, **self._actor_predict_params)
            if agent.agent_idx == self.agent_idx:
                action.detach()
            actions.append(action)
        actions = torch.cat(actions, 1)
        return actions

    def _fit_critic(self, states, obs, actions, q):
        actions = np.array(actions)
        self._critic_approximator.fit(
            states,
            actions.T,
            q,
            **self._critic_fit_params
        )
        return self._critic_approximator.model._last_loss
