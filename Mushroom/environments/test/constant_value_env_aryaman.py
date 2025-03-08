import numpy as np

from Mushroom.environments.fluid.abstract_environments import AbstractEnvironment
from Mushroom.environments.mdp_info import MAMDPInfo
from Mushroom.utils.utils import exponential_reward
from aryaman import Box


class Plug(AbstractEnvironment):
    def __init__(
            self,
            n_agents=1,
            horizon=100,
            gamma=0.99,
            reward_fn="exponential_common",
            smoothness=0.0001,
            bound=1,
            value_at_bound=0.001,
            target=0.4,
            min=0,
            max=1,
            absorbing=False,
    ):
        self.reward_fn = reward_fn
        self.smoothness = smoothness
        self.bound = bound
        self.value_at_bound = value_at_bound
        self.target = target
        self.min = min
        self.max = max
        self.absorbing = absorbing

        self.actions = []

        self._n_agents = n_agents

        self._step_count = 0

        self.action_space = [Box(-1, 1, shape=(1,)) for _ in range(self._n_agents)]
        state_space = Box(-np.inf, np.inf, shape=(1 * self._n_agents,))
        observation_space = [
            Box(-np.inf, np.inf, shape=(1,)) for _ in range(self._n_agents)
        ]

        # Set the MDP info
        mdp_info = MAMDPInfo(
            state_space=state_space,
            observation_spaces=observation_space,
            action_spaces=self.action_space,
            discrete_actions=False,
            gamma=gamma,
            horizon=horizon,
            has_obs=True,
            has_action_masks=False,
            n_agents=self._n_agents,
        )

        super().__init__(mdp_info)

    def reset(self):
        self._step_count = 0
        self._state = np.array([0])

        obs = [self._state] * self._n_agents

        step = {"state": self._state, "obs": obs, "info": {}}

        return step

    def step(self, actions):
        """
        Returns the next state, obs, reward, done, and info.

        Arguments:
            actions (np.ndarray): The actions to take in the environment.

        actions are 2D movements between -1 and 1.
        """
        actions = np.clip(actions, 0, 1)
        self._step_count += 1
        rewards = []
        obs = []
        for i in range(self._n_agents):
            if self.reward_fn == "exponential_common":
                action = np.max(np.abs(np.array(actions) - self.target))
                reward = exponential_reward(action, 0, self.smoothness, self.bound, self.value_at_bound)
                reward *= self.max - self.min
                reward += self.min
            elif self.reward_fn == "exponential":
                reward = exponential_reward(actions[i], self.target, self.smoothness, self.bound, self.value_at_bound)
                reward *= self.max - self.min
                reward += self.min
            elif self.reward_fn == "quadratic":
                reward = float(-np.sum(actions[i] - self.target) ** 2)
            elif self.reward_fn == "quadratic_common":
                reward = float(-np.sum(np.abs(np.array(actions) - self.target)) ** 2)
            else:
                raise ValueError("Invalid reward function")
            rewards.append(float(reward))
            obs.append(self._state)

        self.actions = actions
        absorbing = self.absorbing if self._step_count < self._mdp_info.horizon else True

        step = {
            "state": self._state,
            "obs": obs,
            "rewards": rewards,
            "absorbing": absorbing,
            "info": {},
        }

        return step

    def get_debug_info(self):
        return {
            f"agent_{i}/mean_action": np.mean(self.actions, axis=1)[i] for i in range(self._n_agents)
        }

    def reset_actions(self):
        self.actions = []
