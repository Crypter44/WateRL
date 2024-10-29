from abc import abstractmethod

import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces

from Sofirpy.step_by_step_simulation import ManualStepSimulator


class AbstractFluidNetworkEnv(Environment):
    def __init__(
            self,
            observation_space: spaces.Box,
            action_space: spaces.Box,
            fluid_network_simulator: ManualStepSimulator,
            gamma: float,
            horizon: int,
            fluid_network_simulator_args: dict = None,
    ):
        self._current_state = None
        self._fns_args = fluid_network_simulator_args
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        self._horizon = horizon
        self._sim = fluid_network_simulator
        super().__init__(mdp_info)

    def render(self):
        raise NotImplementedError("Rendering is not supported for this environment.")

    def reset(self, state=None):
        if state is not None:
            raise NotImplementedError("Resetting to a specific state is not supported.")

        if self._fns_args is None:
            self._sim.reset_simulation(100, 1, 1)
        else:
            self._sim.reset_simulation(self._horizon, **self._fns_args)

        self._current_state, _ = self._get_current_state()

        return self._current_state

    def step(self, action):
        self._sim.do_simulation_step(action)
        new_state, absorbing = self._get_current_state()

        reward = self._reward_fun(self._current_state, action, new_state)

        self._current_state = new_state
        return self._current_state, reward, absorbing, {}

    def seed(self, seed):
        super().seed(seed)

    @abstractmethod
    def _get_current_state(self):
        pass

    @abstractmethod
    def _reward_fun(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray):
        pass


