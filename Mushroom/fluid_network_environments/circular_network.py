import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.utils import spaces

from Mushroom.fluid_network_environments.fluid_network_environment import AbstractFluidNetworkEnv
from Sofirpy.networks.control_api import ControlApiCircular
from Sofirpy.simulation import ManualStepSimulator

working_dir = Path(__file__).parent.parent.parent

fmu_dir_path = working_dir / "Fluid_Model" / "circular_water_network"

fmu_path = fmu_dir_path / "mini_circular_water_network.fmu"

agent_config_path = fmu_dir_path / "mini_circular_water_network.json"

connections_config_path = fmu_dir_path / "mini_circular_water_network_connections_config.json"

logging_config_path = fmu_dir_path / "mini_circular_water_network_parameters_to_log.json"

# create interface of multi-agent system to FMU
model_classes = {"control_api": ControlApiCircular}
fmu_paths = {"water_network": str(fmu_path)}

with open(connections_config_path) as connections_config_json:
    connections_config = json.load(connections_config_json)

with open(logging_config_path) as logging_config_json:
    parameters_to_log = json.load(logging_config_json)


class CircularFluidNetwork(AbstractFluidNetworkEnv):
    def __init__(self, gamma: float):
        super().__init__(
            # 4 demands of the network, 4 resulting volume flows
            observation_space=spaces.Box(low=-10, high=10, shape=(8,)),
            # 2 rotational speeds of the pumps
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            fluid_network_simulator=ManualStepSimulator(
                stop_time=200,
                step_size=1,
                fmu_paths=fmu_paths,
                model_classes=model_classes,
                connections_config=connections_config,
                parameters_to_log=parameters_to_log,
                logging_step_size=1,
                get_units=False,
                verbose=False,
            ),
            gamma=gamma,
            horizon=200,
        )
        self._current_simulation_state = None

    def render(self, title=None, save_path=None):
        results = self.sim.get_results()

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        valves = [2, 3, 5, 6]
        valve_colors = ['#FFA500', '#FF8C00', '#FF7F50', '#FF6347']
        pumps = [1, 4]
        pump_colors = ['#1E90FF', '#00BFFF']

        for i in range(4):
            ax2 = axs.flatten()[i].twinx()
            ax2.plot(
                results["time"],
                results[f"water_network.u_v_{valves[i]}"],
                label=f"Opening at v_{valves[i]}",
                color='gray',
                alpha=0.5,
                linewidth=1,
            )
            ax2.set_ylabel("Opening [%]", color='gray')

            axs.flatten()[i].plot(
                results["time"],
                results[f"control_api.w_v_{valves[i]}"],
                label=f"Demand at v_{valves[i]}",
                color=valve_colors[i],
                linestyle='--',
                linewidth=2,
            )
            axs.flatten()[i].plot(
                results["time"],
                results[f"water_network.V_flow_{valves[i]}"],
                label=f"Volume flow at v_{valves[i]}",
                color=valve_colors[i],
            )
            axs.flatten()[i].set_xlabel("Time [s]")
            axs.flatten()[i].set_ylabel("Volume flow [m³/h]")

        axs.flatten()[4].set_visible(False)
        axs.flatten()[7].set_visible(False)

        for i in range(2):
            axs.flatten()[i + 5].plot(
                results["time"],
                results[f"control_api.w_p_{pumps[i]}"],
                label=f"Pump speed at p_{pumps[i]}",
                color=pump_colors[i],
            )
            axs.flatten()[i + 5].set_xlabel("Time [s]")
            axs.flatten()[i + 5].set_ylim((0, 1))
            axs.flatten()[i + 5].set_ylabel("Rotational speed")

        fig.subplots_adjust(
            left=0.05,
            bottom=0.12,
            right=0.95,
            top=0.9,
            hspace=0.4,
            wspace=0.4
        )
        fig.legend(loc="lower center", ncol=6)
        fig.suptitle(title)

        if save_path is None:
            fig.show()
        else:
            fig.savefig(save_path + ".png")
            plt.close(fig)

    def step(self, action):
        # clip action to action space
        action = np.clip(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)
        simulation_states = []
        for i in range(10):  # simulate 10 time steps, to the next control step
            self.sim.do_simulation_step(action)
            simulation_states.append(self._get_current_simulation_state())  # save each sim state to calculate reward

        # calculate reward based on how the network behaved during two control steps
        reward = self._reward_fun(self._current_state, action, simulation_states)

        self._current_simulation_state = simulation_states[-1]
        self._current_state, absorbing = self._get_current_state()
        return self._current_state, reward, absorbing, {}

    def _get_current_state(self):
        """
        Return the observable state of the environment.
        """
        state, absorbing = self._get_current_simulation_state()
        return state[:8], absorbing

    def _get_current_simulation_state(self):
        """
        Return the current state of the simulation, even if it is not observable.
        """
        global_state, done = self.sim.get_current_state()
        try:
            controller_state = global_state["control_api"]
            return np.array(controller_state), done
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")

    def _reward_fun(self, state: np.ndarray, action: np.ndarray, sim_states: list):
        reward = 0
        for s, _ in sim_states:
            for i in range(4):
                reward -= 10 * (s[i] - s[i + 4]) ** 2
            reward -= 0.1 * (s[8] + s[9])

        return reward
