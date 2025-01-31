from typing import List, Optional

import numpy as np
import torch
from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm


# Define the neural networks for the actor and the critic


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