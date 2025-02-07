import torch
import numpy as np
from mushroom_rl.core import Serializable


class ReplayMemory(Serializable):
    def __init__(self, max_size, state_dim, action_dim, discrete_actions=False):
        """
        Constructor.

        Args:
            max_size (int): maximum number of elements that the replay memory
                can contain.
            state_dim (int): dimension of the state space.
            action_dim (int): dimension of the action space.
            discrete_actions (bool): whether the action space is discrete or not.
        """
        self._max_size = max_size
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._discrete_actions = discrete_actions

        self.reset()

        self._add_save_attr(
            _max_size="primitive",
            _state_dim="primitive",
            _action_dim="primitive",
            _discrete_actions="primitive",
            _idx="primitive",
            _full="primitive",
            _states="numpy",
            _actions="numpy",
            _rewards="numpy",
            _next_states="numpy",
            _absorbing="numpy",
            _last="numpy",
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;

        """
        for sample in dataset:
            self._states[self._idx] = sample["state"]
            self._actions[self._idx] = sample["action"]
            self._rewards[self._idx] = sample["reward"]
            self._next_states[self._idx] = sample["next_state"]
            self._absorbing[self._idx] = sample["absorbing"]
            self._last[self._idx] = sample["last"]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of samples from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        indices = np.random.randint(self.size, size=n_samples)

        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        absorbing = self._absorbing[indices]
        last = self._last[indices]

        return states, actions, rewards, next_states, absorbing, last

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        if self._discrete_actions:
            self._actions = np.empty((self._max_size, 1), dtype=np.int32)
        else:
            self._actions = np.empty(
                (self._max_size, self._action_dim), dtype=np.float32
            )
        self._rewards = np.empty(self._max_size, dtype=np.float32)
        self._next_states = np.empty(
            (self._max_size, self._state_dim), dtype=np.float32
        )
        self._absorbing = np.empty(self._max_size, dtype=bool)
        self._last = np.empty(self._max_size, dtype=bool)

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size


class ReplayMemoryObs(ReplayMemory):
    """
    Replay memory that stores observations instead of states.
    """

    def __init__(
        self, max_size, state_dim, obs_dim, action_dim, discrete_actions=False
    ):
        self._obs_dim = obs_dim
        super().__init__(max_size, state_dim, action_dim, discrete_actions)

        self._add_save_attr(
            _obs_dim="primitive",
            _obs="numpy",
            _next_obs="numpy",
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;

        """
        for sample in dataset:
            self._states[self._idx] = sample["state"]
            self._obs[self._idx] = sample["obs"]
            self._actions[self._idx] = sample["action"]
            self._rewards[self._idx] = sample["reward"]
            self._next_states[self._idx] = sample["next_state"]
            self._next_obs[self._idx] = sample["next_obs"]
            self._absorbing[self._idx] = sample["absorbing"]
            self._last[self._idx] = sample["last"]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of samples from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        indices = np.random.randint(self.size, size=n_samples)

        states = self._states[indices]
        obs = self._obs[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        next_obs = self._next_obs[indices]
        absorbing = self._absorbing[indices]
        last = self._last[indices]

        return states, obs, actions, rewards, next_states, next_obs, absorbing, last

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._obs = np.empty((self._max_size, self._obs_dim), dtype=np.float32)
        if self._discrete_actions:
            self._actions = np.empty((self._max_size, 1), dtype=np.int32)
        else:
            self._actions = np.empty(
                (self._max_size, self._action_dim), dtype=np.float32
            )
        self._rewards = np.empty(self._max_size, dtype=np.float32)
        self._next_states = np.empty(
            (self._max_size, self._state_dim), dtype=np.float32
        )
        self._next_obs = np.empty((self._max_size, self._obs_dim), dtype=np.float32)
        self._absorbing = np.empty(self._max_size, dtype=bool)
        self._last = np.empty(self._max_size, dtype=bool)


class ReplayMemoryObsTorch(ReplayMemoryObs):
    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = torch.empty(
            (self._max_size, self._state_dim), dtype=torch.float32
        )
        self._obs = torch.empty((self._max_size, self._obs_dim), dtype=torch.float32)
        if self._discrete_actions:
            self._actions = torch.empty((self._max_size, 1), dtype=torch.int32)
        else:
            self._actions = torch.empty(
                (self._max_size, self._action_dim), dtype=torch.float32
            )
        self._rewards = torch.empty(self._max_size, dtype=torch.float32)
        self._next_states = torch.empty(
            (self._max_size, self._state_dim), dtype=torch.float32
        )
        self._next_obs = torch.empty(
            (self._max_size, self._obs_dim), dtype=torch.float32
        )
        self._absorbing = torch.empty(self._max_size, dtype=torch.bool)
        self._last = torch.empty(self._max_size, dtype=torch.bool)


class ReplayMemoryObsMultiAgent(ReplayMemoryObs):
    """
    Replay memory that stores observations instead of states.
    """

    def __init__(
        self,
        max_size: int,
        state_dim: int,
        obs_dim: list,
        action_dim: list,
        n_agents: int,
        discrete_actions: bool = False,
    ):
        self._n_agents = n_agents
        super().__init__(max_size, state_dim, obs_dim, action_dim, discrete_actions)

        self._add_save_attr(
            _n_agents="primitive",
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;

        """
        for sample in dataset:
            self._states[self._idx] = sample["state"]
            for idx_agent in range(self._n_agents):
                self._obs[idx_agent][self._idx] = sample["obs"][idx_agent]
                self._next_obs[idx_agent][self._idx] = sample["next_obs"][idx_agent]
                self._actions[idx_agent][self._idx] = sample["actions"][idx_agent]
            self._rewards[self._idx] = sample["rewards"]
            self._next_states[self._idx] = sample["next_state"]
            self._absorbing[self._idx] = sample["absorbing"]
            self._last[self._idx] = sample["last"]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of samples from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        indices = np.random.randint(self.size, size=n_samples)

        states = self._states[indices]
        obs = [agent_obs[indices] for agent_obs in self._obs]
        actions = [agent_actions[indices] for agent_actions in self._actions]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        next_obs = [agent_next_obs[indices] for agent_next_obs in self._next_obs]
        absorbing = self._absorbing[indices]
        last = self._last[indices]

        return states, obs, actions, rewards, next_states, next_obs, absorbing, last

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._obs = [
            np.empty((self._max_size, self._obs_dim[idx_agent]), dtype=np.float32)
            for idx_agent in range(self._n_agents)
        ]
        if self._discrete_actions:
            self._actions = [
                np.empty((self._max_size, 1), dtype=np.int32)
                for _ in range(self._n_agents)
            ]
        else:
            self._actions = [
                np.empty(
                    (self._max_size, self._action_dim[idx_agent]), dtype=np.float32
                )
                for idx_agent in range(self._n_agents)
            ]
        self._rewards = np.empty((self._max_size, self._n_agents), dtype=np.float32)
        self._next_states = np.empty(
            (self._max_size, self._state_dim), dtype=np.float32
        )
        self._next_obs = [
            np.empty((self._max_size, self._obs_dim[idx_agent]), dtype=np.float32)
            for idx_agent in range(self._n_agents)
        ]
        self._absorbing = np.empty(self._max_size, dtype=bool)
        self._last = np.empty(self._max_size, dtype=bool)


class ReplayMemoryObsMasks(ReplayMemoryObs):
    def add(self, dataset):
        for sample in dataset:
            self._obs[self._idx] = sample["obs"]
            self._action_masks[self._idx] = np.array(sample["action_mask"], dtype=bool)
            self._actions[self._idx] = sample["action"]
            self._rewards[self._idx] = sample["reward"]
            self._next_obs[self._idx] = sample["next_obs"]
            self._next_action_masks[self._idx] = np.array(
                sample["next_action_mask"], dtype=bool
            )
            self._absorbing[self._idx] = sample["absorbing"]
            self._last[self._idx] = sample["last"]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of samples from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        indices = np.random.randint(self.size, size=n_samples)

        obs = self._obs[indices]
        actions = self._actions[indices]
        action_masks = self._action_masks[indices]
        rewards = self._rewards[indices]
        next_obs = self._next_obs[indices]
        next_action_masks = self._next_action_masks[indices]
        absorbing = self._absorbing[indices]
        last = self._last[indices]

        return (
            obs,
            actions,
            action_masks,
            rewards,
            next_obs,
            next_action_masks,
            absorbing,
            last,
        )

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._obs = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._action_masks = np.empty((self._max_size, self._action_dim), dtype=bool)
        if self._discrete_actions:
            self._actions = np.empty((self._max_size, 1), dtype=np.int32)
        else:
            self._actions = np.empty(
                (self._max_size, self._action_dim), dtype=np.float32
            )
        self._rewards = np.empty(self._max_size, dtype=np.float32)
        self._next_obs = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._next_action_masks = np.empty(
            (self._max_size, self._action_dim), dtype=bool
        )
        self._absorbing = np.empty(self._max_size, dtype=bool)
        self._last = np.empty(self._max_size, dtype=bool)


class EpisodicReplayMemory(ReplayMemory):
    def __init__(self, max_size):
        """
        Constructor.

        Args:
            max_size (int): maximum number of elements that the replay memory
                can contain.
        """
        self._max_size = max_size

        self.reset()

        self._add_save_attr(
            _max_size="primitive",
            _idx="primitive",
            _full="primitive",
            _episodes="pickle",
        )

    def add(self, dataset):
        """
        Add elements to the replay memory
        Only complete episodes are added to the replay memory.

        Args:
            dataset (list): list of elements to add to the latest episode in the replay memory;
        """
        episode = []
        for sample in dataset:
            episode.append(sample)
            if sample["last"]:
                self._episodes[self._idx] = episode
                self._idx += 1
                episode = []
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0

    def get(self, n_episodes):
        """
        Returns the provided number of episodes from the replay memory.
        Args:
            n_episodes (int): the number of episodes to return.
        Returns:
            The requested number of episodes.
        """
        episodes = list()
        for i in np.random.randint(self.size, size=n_episodes):
            episodes.append(self._episodes[i])

        return episodes

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._episodes = [None for _ in range(self._max_size)]
