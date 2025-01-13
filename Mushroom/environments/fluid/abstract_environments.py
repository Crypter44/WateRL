from abc import abstractmethod
from typing import List

import numpy as np

from Mushroom.environments.environment import MAEnvironment
from mushroom_rl.utils import spaces

from Mushroom.environments.mdp_info import MAMDPInfo
from Sofirpy.simulation import ManualStepSimulator


class AbstractEnvironment(MAEnvironment):
    def __init__(self, mdp_info: MAMDPInfo):
        super().__init__(mdp_info)

    def local_observation_space(self, agent_index: int):
        return self._mdp_info.observation_space_for_idx(agent_index)

    def local_action_space(self, agent_index: int):
        return self._mdp_info.action_space_for_idx(agent_index)


class AbstractFluidNetworkEnv(AbstractEnvironment):
    def __init__(
            self,
            state_space: spaces.Box,
            observation_spaces: List[spaces.Box],
            action_spaces: List[spaces.Box],
            fluid_network_simulator: ManualStepSimulator,
            gamma: float,
            horizon: int,
            n_agents: int = 1,
            labeled_step: bool = False
    ):
        self._current_state = None
        mdp_info = MAMDPInfo(
            state_space,
            observation_spaces,
            action_spaces,
            False,
            gamma,
            horizon,
            has_obs=True,
            n_agents=n_agents
        )
        self.labeled_step = labeled_step
        self._horizon = horizon
        self.sim = fluid_network_simulator
        super().__init__(mdp_info)

    def render(self):
        raise NotImplementedError("Rendering is not supported for this environment.")

    def reset(self, state=None) -> dict:
        if state is not None:
            raise NotImplementedError("Resetting to a specific state is not supported.")

        self.sim.reset_simulation(self._horizon, 1, 1)
        self._current_state, _ = self._get_current_state()

        sample = {
            "state": self._current_state,
            "obs": self._get_observations(),
        }
        return sample if self.labeled_step else self._get_observations()

    def _get_observations(self) -> List:
        return [self._current_state for _ in range(self._mdp_info.n_agents)]

    @abstractmethod
    def step(self, action):
        pass

    def seed(self, seed):
        super().seed(seed)

    @abstractmethod
    def _get_current_state(self) -> np.ndarray:
        pass
