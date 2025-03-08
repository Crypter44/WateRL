from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.abstract_environments import AbstractFluidNetworkEnv
from Sofirpy.networks.simple_network.config import get_simple_network_valve_config
from Sofirpy.simulation import ManualStepSimulator

class SimpleNetworkValve(AbstractFluidNetworkEnv):
    def __init__(self, gamma: float, horizon: int, fluid_network_simulator_args: dict = None):
        super().__init__(
            observation_space=spaces.Box(low=-10, high=10, shape=(2,)),
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            fluid_network_simulator=ManualStepSimulator(
                stop_time=horizon,
                step_size=1,
                config=get_simple_network_valve_config(),
                logging_step_size=1,
                get_units=False,
                verbose=False,
            ),
            gamma=gamma,
            horizon=horizon,
            fluid_network_simulator_args=fluid_network_simulator_args,
        )

    def render(self, title: str = None):
        dataframe = self.sim.get_results()
        demand = self.sim.systems["control_api"].simulation_entity.requested_volume_flow,

        last_non_zero_index = dataframe["time"].to_numpy().nonzero()[0]

        if len(last_non_zero_index) == 0:
            print("No data to plot.")
            return
        else:
            last_non_zero_index = last_non_zero_index[-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax2 = ax.twinx()

        ax.plot(
            dataframe["time"][1:last_non_zero_index],
            [demand] * (last_non_zero_index-1),
            lw=1.5,
            label="DEMAND",
            linestyle=(0, (2, 1)),
            c=[253 / 255, 202 / 255, 0 / 255],
        )
        ax.plot(
            dataframe["time"][1:last_non_zero_index],
            dataframe["water_network.V_flow_2"][1:last_non_zero_index],
            lw=1.5,
            label="VOLUME FLOW",
            c=[253 / 255, 202 / 255, 0 / 255],
        )
        ax2.plot(
            dataframe["time"][1:last_non_zero_index],
            dataframe["control_api.w_v_2"][1:last_non_zero_index],
            lw=1.5,
            label="VALVE OPENING",
            markersize=8,
            c=[0 / 255, 78 / 255, 115 / 255],
        )
        ax.set_xlabel("TIME in s")
        ax.set_xlim(0, self._horizon)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("VOLUME FLOW AT VALVE in m$^3$/h")
        ax2.set_ylabel("VALVE OPENING in %", c=[0 / 255, 78 / 255, 115 / 255])
        ax2.set_ylim(0, 1)
        ax2.spines["right"].set_visible(True)
        fig.suptitle(title)
        fig.legend()
        fig.show()

    def step(self, action):
        # clip action to action space
        action = np.clip(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)
        self.sim.do_simulation_step(action)
        new_state, absorbing = self._get_simulation_state()

        reward = self._reward_fun(self._current_sim_state, action, new_state)

        self._current_state = new_state
        return self._current_state, reward, absorbing, {}

    def _get_simulation_state(self):
        global_state, done = self.sim.get_current_state()
        try:
            local_state = global_state["control_api"]
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")
        return local_state[0:2], done

    def _reward_fun(self, state, action, new_state):
        demand = state[0]
        supply = state[1]

        new_demand = new_state[0]
        new_supply = new_state[1]

        return -100 * (new_demand - new_supply) ** 2
