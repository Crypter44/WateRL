import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.abstract_environments import AbstractFluidNetworkEnv
from Sofirpy.networks.control_api import ControlApiCircular
from Sofirpy.simulation import ManualStepSimulator

working_dir = Path(__file__).parent.parent.parent.parent

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
            observation_spaces=None,
            action_spaces=None,
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
            criteria: dict = None,
            labeled_step: bool = False,
    ):
        super().__init__(
            state_space=spaces.Box(low=-500, high=500, shape=(18,)),
            observation_spaces=(
                observation_spaces
                if observation_spaces is not None
                else [spaces.Box(low=-10, high=10, shape=(4,))]
            ),
            action_spaces=(
                action_spaces
                if action_spaces is not None
                else [spaces.Box(low=0, high=1, shape=(1,))]
            ),
            fluid_network_simulator=fluid_network_simulator,
            horizon=horizon,
            gamma=gamma,
            labeled_step=labeled_step,
            n_agents=2,
        )
        self._criteria = criteria if criteria is not None else {"demand": 1}
        self._current_simulation_state = None
        self.actions = []
        self.rewards = []

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
            save_path=save_path,
            rewards=self.rewards,
        )

    def step(self, action):
        # clip action to action space
        action = np.clip(action, self._mdp_info.action_space_for_idx(0).low, self._mdp_info.action_space_for_idx(0).high)
        simulation_states = []
        for i in range(10):  # simulate 10 time steps, to the next control step
            self.sim.do_simulation_step(action)
            simulation_states.append(self._get_current_state())  # save each sim state to calculate reward

        # calculate reward based on how the network behaved during two control steps
        reward = self._reward_fun(self._current_state, action, simulation_states)

        self._current_state, absorbing = self._get_current_state()
        self.actions.append(action)
        self.rewards.append(reward)

        step = {
            "state": self._current_state,
            "obs": self._get_observations(),
            "rewards": reward,
            "absorbing": absorbing,
        }
        return step if self.labeled_step else (self._get_observations(), reward, absorbing, {})

    def reset(self, state=None):
        self.actions = []
        self.rewards = []
        return super().reset(state)

    def _get_current_state(self):
        """
        Return the current state of the simulation, even if it is not observable.
        """
        global_state, done = self.sim.get_current_state()
        try:
            controller_state = global_state["control_api"]
            return np.array(controller_state), done
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")

    def _get_observations(self):
        return [self._current_state[:4] for _ in range(self._mdp_info.n_agents)]

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
        def r_demand(d):
            smoothness = 0.0001
            bound = 0.2
            value_at_bound = 0.01

            b = np.log(value_at_bound) / (np.sqrt(smoothness) - np.sqrt(smoothness + bound ** 2))
            a = np.exp(b * np.sqrt(smoothness))

            return a * np.exp(-b * np.sqrt(d ** 2 + smoothness))

        def r_power(p):
            return 1 / 187 * (193 - p)

        # return -(abs(0.25 - float(action[2]))) * 10
        num_valves = 4

        reward = 0
        sim_states_to_use = sim_states[-1:]
        for s, _ in sim_states_to_use:
            tmp = 0

            if "demand" in self._criteria.keys():
                for i in range(num_valves):
                    tmp += (
                            1 / num_valves
                            * self._criteria["demand"]
                            * r_demand(s[i] - s[i + 4])
                    )
            if "max_power" in self._criteria.keys():
                tmp += self._criteria["max_power"] * min(r_power(s[16]), r_power(s[17]))
            if "mean_power" in self._criteria.keys():
                tmp += self._criteria["mean_power"] * (r_power(s[16]) + r_power(s[17])) / 2
            if "opening" in self._criteria.keys():
                tmp += self._criteria["opening"] * max(s[8], s[9], s[10], s[11])
            if "max_speed" in self._criteria.keys():
                tmp += self._criteria["max_speed"] * min(1 - s[12], 1 - s[13])
            if "mean_speed" in self._criteria.keys():
                tmp += self._criteria["mean_speed"] * (2 - s[12] - s[13]) / 2
            if "negative_flow" in self._criteria.keys():
                if s[14] < 0 or s[15] < 0:
                    tmp -= self._criteria["negative_flow"]

            reward += tmp / len(sim_states_to_use)

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
            rewards=None,
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
            ax2.set_ylim(0, 1)

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

        if rewards is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.plot(rewards)
            ax.set_xlabel("Action step")
            ax.set_ylabel("Reward")
            ax.set_ylim(-1, 1.5)
            ax.set_xticks(range(0, len(rewards), 1))
            ax.grid(True)
            fig.suptitle(title)

            # plot min and max line with value
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            ax.text(0, min_reward, f'{min_reward:.2f}', color='green', fontsize=10, ha='center', va='bottom')
            ax.text(0, max_reward, f'{max_reward:.2f}', color='red', fontsize=10, ha='center', va='top')
            ax.plot(range(0, len(rewards), 1), [min_reward] * len(rewards), color='green', linestyle='--', linewidth=1)
            ax.plot(range(0, len(rewards), 1), [max_reward] * len(rewards), color='red', linestyle='--', linewidth=1)

            if save_path is None:
                fig.show()
            else:
                path = str.replace(save_path, "Epoch", "Epoch_rewards")
                path = str.replace(path, "Final", "Final_rewards")
                fig.savefig(path + ".png")
            plt.close(fig)