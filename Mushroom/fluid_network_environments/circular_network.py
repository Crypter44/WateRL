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
    def __init__(
            self,
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
            horizon=200,
            gamma: float = 0.99,
            power_penalty: float = 0.01,
            penalize_negative_flow: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            fluid_network_simulator=fluid_network_simulator,
            horizon=horizon,
            gamma=gamma,
        )
        self._power_penalty = power_penalty
        self._penalize_negative_flow = penalize_negative_flow
        self._current_simulation_state = None
        self.actions = []

    def render(self, title=None, save_path=None):
        results = self.sim.get_results()
        valves = [2, 3, 5, 6]
        pumps = [1, 4]
        self.plot_valve_and_pump_data(
            time=results["time"],
            valves=valves,
            valve_openings=[results[f"water_network.u_v_{v}"] for v in valves],
            valve_demands=[results[f"control_api.demand_v_{v}"] for v in valves],
            valve_flows=[results[f"water_network.V_flow_{v}"] for v in valves],
            pumps=pumps,
            pump_speeds=[results[f"control_api.w_p_{p}"] for p in pumps],
            pump_powers=[results[f"water_network.P_pum_{p}"] for p in pumps],
            pump_flows=[results[f"water_network.V_flow_{p}"] for p in pumps],
            title=title,
            save_path=save_path
        )

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
        self.actions.append(action)
        return self._current_state, reward, absorbing, {}

    def reset(self, state=None):
        self.actions = []
        return super().reset(state)

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
        return self._bound_reward(state, action, sim_states)

    def _unbound_summed_squared_error(self, state: np.ndarray, action: np.ndarray, sim_states: list):
        reward = 0
        for s, _ in sim_states[:]:
            for i in range(4):  # loop over 4 valves
                demand = s[i]
                volume_flow = s[i + 4]
                reward -= 10 * (demand - volume_flow) ** 2
            reward -= self._power_penalty * (s[12] + s[13])

        return reward

    def _bound_reward(self, state: np.ndarray, action: np.ndarray, sim_states: list):
        def r_deviation(d):
            # return -d ** 2 / 1.3 ** 2 + 1
            # return 0.9 * np.exp(-40*d**2) + 0.1 * np.exp(-2*d**2)
            # return np.exp(-500 * (d - .1) ** 2)
            return np.exp(-4 * d ** 2)

        def r_power(p):
            return 1 / 187 * (193 - p)
            #
            # b = 0.005
            # a = 1 / (np.exp(-6 * b) - np.exp(-193 * b))
            # c = -a * np.exp(-193 * b)
            # return a * np.exp(-b * p) + c

        reward = 0
        for s, _ in sim_states[:]:
            tmp = 0
            for i in range(4):
                tmp += 0.25 * (1 - self._power_penalty) * r_deviation(s[i] - s[i + 4])

            tmp += 0.5 * self._power_penalty * (r_power(s[16]))  # TODO change to actual power draw, if sim is fixed
            tmp += 0.5 * self._power_penalty * (r_power(s[17]))  # TODO change to actual power draw, if sim is fixed

            # tmp += 0.5 * self._power_penalty * (1-s[12])
            # tmp += 0.5 * self._power_penalty * (1-s[13])

            if self._penalize_negative_flow and (s[14] < 0 or s[15] < 0):
                return -10

            reward += tmp

        return reward

    @staticmethod
    def plot_valve_and_pump_data(
            time,
            valves,
            valve_openings,
            valve_demands,
            valve_flows,
            pumps,
            pump_speeds,
            pump_powers,
            pump_flows,
            pump_actions=None,
            title=None,
            save_path=None
    ):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        valve_colors = ['#FFA500', '#FF8C00', '#FF7F50', '#FF6347']
        pump_colors = ['#1E90FF', '#00BFFF']

        for i in range(4):
            ax2 = axs.flatten()[i].twinx()
            ax2.plot(
                time,
                valve_openings[i],
                label=f"Opening at v_{valves[i]}",
                color='gray',
                alpha=.9,
                linewidth=1,
            )
            ax2.set_ylabel("Opening [%]", color='gray')

            axs.flatten()[i].plot(
                time,
                valve_demands[i],
                label=f"Demand at v_{valves[i]}",
                color=valve_colors[i],
                linestyle='--',
                linewidth=3,
            )
            axs.flatten()[i].plot(
                time,
                valve_flows[i],
                label=f"Volume flow at v_{valves[i]}",
                color=valve_colors[i],
                linewidth=2,
            )
            axs.flatten()[i].set_xlabel("Time [s]")
            axs.flatten()[i].set_ylabel("Volume flow [m³/h]")

        for i in range(2):
            axs.flatten()[i + 5].plot(
                time,
                pump_speeds[i],
                label=f"Pump speed at p_{pumps[i]}",
                color=pump_colors[i],
                linewidth=2,
                zorder=0
            )

            if pump_actions is not None:
                axs.flatten()[i + 5].scatter(
                    range(0, len(pump_actions) * 10, 10),
                    np.array(pump_actions)[:, i],
                    label=f"Action p_{pumps[i]}",
                    color=pump_colors[i],
                    edgecolors='black',
                    s=7,
                    linewidths=0.5,
                    zorder=2
                )
            ax2 = axs.flatten()[i + 5].twinx()
            ax2.plot(
                time,
                pump_powers[i],
                label=f"Power consumption of p_{pumps[i]}",
                color='gray',
                alpha=.9,
                linewidth=1,
                zorder=1
            )
            ax2.set_ylabel("Power consumption [W]", color='gray')
            ax2.set_ylim((0, 200))
            axs.flatten()[i + 5].set_xlabel("Time [s]")
            axs.flatten()[i + 5].set_ylim((0, 1))
            axs.flatten()[i + 5].set_ylabel("Rotational speed")

            min_power = np.min(pump_powers[i])
            max_power = np.max(pump_powers[i])
            min_time = time[np.argmin(pump_powers[i])]
            max_time = time[np.argmax(pump_powers[i])]

            # Add annotations for min and max power draw
            ax2.annotate(f'{min_power:.2f} W', xy=(min_time, min_power), xytext=(min_time + 10, min_power + 10),
                         arrowprops=dict(arrowstyle='-', color='green', linewidth=1.5), color='green',
                         horizontalalignment='left', fontsize=8, alpha=0.8)
            ax2.annotate(f'{max_power:.2f} W', xy=(max_time, max_power), xytext=(max_time - 10, max_power - 10),
                         arrowprops=dict(arrowstyle='-', color='red', linewidth=1.5), color='red',
                         horizontalalignment='right', fontsize=8, alpha=0.8)

            ax = axs.flatten()[4 + 3 * i]
            ax.plot(
                time,
                pump_flows[i],
                label=f"Volume flow at p_{pumps[i]}",
                color=pump_colors[i],
                linewidth=2,
                zorder=2
            )
            ax.fill_between(
                time,
                pump_flows[i],
                where=(pump_flows[i] < 0),
                color='red',
                alpha=0.3,
                label='Invalid (negative)'
            )
            ax.plot(
                time,
                np.zeros_like(time),
                color='red',
                linestyle='--',
                linewidth=1,
                zorder=1
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Volume flow [m³/h]")

        fig.subplots_adjust(
            left=0.05,
            bottom=0.12,
            right=0.95,
            top=0.9,
            hspace=0.4,
            wspace=0.4
        )
        fig.legend(loc="lower center", ncol=12 if pump_actions is not None else 10)
        fig.suptitle(title)

        if save_path is None:
            fig.show()
        else:
            fig.savefig(save_path + ".png")
            plt.close(fig)

