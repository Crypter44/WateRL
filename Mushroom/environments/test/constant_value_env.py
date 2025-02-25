import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.abstract_environments import AbstractEnvironment
from Mushroom.environments.mdp_info import MAMDPInfo


class ConstantValueEnv(AbstractEnvironment):
    def __init__(
            self,
            value,
            start_state=None,
            steps_until_state_change=10,
            reset_to_start=False,
            state_length=4,
            state_min=-10,
            state_max=10,
            action_min=0,
            action_max=1,
            reward_fn=None,
            num_agents=1,
            labeled_state=False
    ):
        self._value = value
        self._start_state = start_state
        self._steps_until_state_change = steps_until_state_change
        self._reset_to_start = reset_to_start
        self._state_length = state_length
        self._state_min = state_min
        self._state_max = state_max
        self._action_min = action_min
        self._action_max = action_max
        self._num_agents = num_agents
        self._labeled_state = labeled_state

        if reward_fn is None:
            raise ValueError("reward_fn must be provided")
        self._reward_fn = reward_fn

        observation_space = spaces.Box(low=state_min, high=state_max, shape=(self._state_length,))
        action_space = spaces.Box(low=action_min, high=action_max, shape=(1,))
        mdp_info = MAMDPInfo(
            observation_space,
            [observation_space],
            [action_space],
            False,
            .99,
            10,
            n_agents=num_agents
        )
        super().__init__(mdp_info)

        if self._start_state is not None:
            self._state = self._start_state
        else:
            self._state = np.random.uniform(-10, 10, self._state_length)

        self._state_log = None
        self._action_log = None

    def reset(self, state=None):
        if self._reset_to_start:
            self._state = self._start_state
        else:
            self._state = np.random.uniform(self._state_min, self._state_max, self._state_length)
        self._state_log = [self._state]
        self._action_log = []

        sample = {
            'state': self._state,
            'obs': [self._state] * self._num_agents,
        }
        if self._labeled_state:
            return sample
        else:
            return sample['obs']

    def step(self, action):
        action = np.clip(action, self._action_min, self._action_max)
        if len(self._state_log) % self._steps_until_state_change == 0:
            self._state = np.random.uniform(self._state_min, self._state_max, self._state_length)
        self._state_log.append(self._state)
        self._action_log.append(action)
        reward = self._reward_fn(action)

        sample = {
            'state': self._state,
            'obs': [self._state] * self._num_agents,
            'rewards': [reward] * self._num_agents,
            'absorbing': False,
            'info': {}
        }

        if self._labeled_state:
            return sample
        else:
            return sample['obs'], sample['rewards'], sample['absorbing'], sample['info']

    def render(self, save_path=None):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        state_plot = ax[0]
        state_plot.set_title("State")
        state_plot.set_ylim((self._state_min - 0.25, self._state_max + 0.25))
        for i in range(self._state_length):
            state_plot.plot([s[i] for s in self._state_log])

        action_plot = ax[1]
        action_plot.set_title("Action")
        action_plot.set_ylim((self._action_min - 0.25, self._action_max + 0.25))
        for i in range(self._num_agents):
            action_plot.plot([a[i] for a in self._action_log], alpha=0.5)

            # Calculate min and max action values
            min_action = min(float(a[i]) for a in self._action_log)
            max_action = max(float(a[i]) for a in self._action_log)

            # Add labels for min and max
            action_plot.text(np.argmin(self._action_log), min_action - 0.05, f'{min_action:.3f}', color='green',
                             va='center', ha='right')
            action_plot.text(np.argmax(self._action_log), max_action + 0.05, f'{max_action:.3f}', color='red',
                             va='center', ha='right')

        reward_plot = ax[2]
        reward_plot.set_title("Reward")
        reward_plot.plot([self._reward_fn(a) for a in self._action_log], alpha=0.5, color='green')

        fig.savefig(save_path)
        plt.close(fig)

    def local_observation_space(self, agent_index: int):
        return spaces.Box(low=self._state_min, high=self._state_max, shape=(self._state_length,))

    def local_action_space(self, agent_index: int):
        return spaces.Box(low=self._action_min, high=self._action_max, shape=(1,))
