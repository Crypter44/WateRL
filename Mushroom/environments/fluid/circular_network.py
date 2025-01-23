import concurrent.futures
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.abstract_environments import AbstractFluidNetworkEnv
from Mushroom.utils.utils import exponential_reward, linear_reward
from Sofirpy.networks.agents import set_demand_for_consumers
from Sofirpy.networks.circular_network.config import get_circular_network_config
from Sofirpy.simulation import ManualStepSimulator


class CircularFluidNetwork(AbstractFluidNetworkEnv):
    def __init__(
            self,
            observation_spaces=None,
            action_spaces=None,
            fluid_network_simulator=None,
            horizon=50,
            gamma: float = 0.99,
            criteria: dict[str, dict[str, float]] = None,
            demand=("uniform_individual", 0.3, 1.5),
            labeled_step: bool = False,
    ):

        if fluid_network_simulator is None:
            fluid_network_simulator = self._setup_simulator(**demand)

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
        self._criteria = criteria if criteria is not None else {"demand": {"w": 1}}
        self._current_simulation_state = None
        self.actions = []
        self.rewards = []

        self._agents = None
        self.qs = None

        # Init rendering using matplotlib
        self._render_executor = concurrent.futures.ThreadPoolExecutor()

    def render(self, title=None, save_path=None):
        results = self.sim.get_results()
        rewards = deepcopy(self.rewards)
        qs = deepcopy(self.qs) if self.qs else None
        self._render_executor.submit(self._render_task, title, save_path, results, rewards=rewards, qs=qs)

    def _render_task(self, title=None, save_path=None, results=None, rewards=None, qs=None):
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
            rewards=rewards,
            qs=qs,
        )

    def enable_q_logging(self, agents):
        assert len(agents) == self._mdp_info.n_agents
        self.qs = [[] for _ in range(self._mdp_info.n_agents)]
        self._agents = agents

    def step(self, action):
        # clip action to action space
        action = np.clip(action, self._mdp_info.action_space_for_idx(0).low,
                         self._mdp_info.action_space_for_idx(0).high)

        if self.qs is not None:
            for i, a in enumerate(self._agents):
                self.qs[i].append(a.critic_approximator.predict(
                    np.array([self._get_observations()[i]]),
                    np.array([action[i]])
                ))

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
        if self.qs:
            self.qs = [[] for _ in range(self._mdp_info.n_agents)]
        return super().reset(state)

    def stop_renderer(self):
        self._render_executor.shutdown()
        plt.close("all")

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

    def _setup_simulator(self, demand_type, low, high):
        config = get_circular_network_config()

        set_demand_for_consumers(config["model_init_args"], self._configure_demand(demand_type, low, high))

        return ManualStepSimulator(
            stop_time=50,
            step_size=1,
            **config,
            logging_step_size=1,
            get_units=False,
            verbose=False,
        )

    @staticmethod
    def _configure_demand(kind, low, high, count) -> list[float]:
        if kind == "uniform_individually":
            return [np.random.uniform(low, high) for _ in range(count)]
        elif kind == "uniform_global":
            demand = np.random.uniform(low * count, high * count)
            random_numbers = np.random.dirichlet(np.ones(count))
            scaled_numbers = random_numbers * (demand - count * low)
            final_numbers = scaled_numbers + low
            while np.any(final_numbers > high):
                excess = np.sum(final_numbers[final_numbers > high] - high)
                final_numbers[final_numbers > high] = high
                final_numbers[final_numbers < high] += excess / np.sum(final_numbers < high)
            return final_numbers
        else:
            raise ValueError(f"Unknown demand type: {kind}")

    def _reward_fun(self, state: np.ndarray, action: np.ndarray, sim_states: list):
        """
        Calculate the reward based on the current state and the action taken.

        Which criteria are used to calculate the reward can be set in the constructor.
        There are several criteria available:
        - demand: The reward is based on the difference between the demand and the actual flow.
        - max_power: The reward is based on the maximum power consumption of the pumps.
        - mean_power: The reward is based on the mean power consumption of the pumps.
        - opening: The reward is based on the opening of the valves, the higher the opening, the higher the reward.
        - target_opening: The reward is based on the difference between the target opening and the actual opening.
        - max_speed: The reward is based on the maximum speed of the pumps.
        - mean_speed: The reward is based on the mean speed of the pumps.
        - target_speed: The reward is based on the difference between the target speed and the actual mean pump speed.
        - negative_flow: The reward is based on the flow of the pumps, if the flow is negative, the reward is negative.

        :param state: The current state of the simulation.
        :param action: The action taken.
        :param sim_states: The states of the simulation during the control step.
        :return: The reward.
        """
        num_valves = 4

        reward = 0
        sim_states_to_use = sim_states[-1:]
        for s, _ in sim_states_to_use:
            tmp = 0

            if "demand" in self._criteria.keys():
                for i in range(num_valves):
                    tmp += (
                            1 / num_valves *
                            self._criteria["demand"]["w"] *
                            exponential_reward(
                                s[i] - s[i + 4],
                                0,
                                self._criteria["demand"].get("smoothness", 0.0001),
                                self._criteria["demand"].get("bound", 0.1),
                                self._criteria["demand"].get("value_at_bound", 0.01),
                            )
                    )
            if "max_power" in self._criteria.keys():
                tmp += self._criteria["max_power"]["w"] * linear_reward(max(s[16], s[17]), 6, 193)
            if "mean_power" in self._criteria.keys():
                tmp += self._criteria["mean_power"]["w"] * linear_reward((s[16] + s[17]) / 2, 6, 193)
            if "opening" in self._criteria.keys():
                tmp += self._criteria["opening"]["w"] * max(s[8], s[9], s[10], s[11])
            if "target_opening" in self._criteria.keys():
                x = max(s[8], s[9], s[10], s[11])
                target = self._criteria["target_opening"].get("target", 0.95)
                smoothness = self._criteria["target_opening"].get("smoothness", 0.01)
                left_bound = self._criteria["target_opening"].get("left_bound", 0.3)
                value_at_left_bound = self._criteria["target_opening"].get("value_at_left_bound", 0.01)
                right_bound = self._criteria["target_opening"].get("right_bound", 0.05)
                value_at_right_bound = self._criteria["target_opening"].get("value_at_right_bound", 0.01)
                if x < target:
                    tmp += self._criteria["target_opening"]["w"] * (
                        exponential_reward(
                            x,  # only consider the valve with the highest opening
                            target,
                            smoothness,
                            left_bound,
                            value_at_left_bound,
                        )
                    )
                elif x > 0.9999:
                    tmp -= self._criteria["target_opening"]["w"]
                else:
                    tmp += self._criteria["target_opening"]["w"] * (
                        exponential_reward(
                            x,
                            target,
                            smoothness,
                            right_bound,
                            value_at_right_bound,
                        )
                    )
            if "max_speed" in self._criteria.keys():
                tmp += self._criteria["max_speed"]["w"] * min(1 - s[12], 1 - s[13])
            if "mean_speed" in self._criteria.keys():
                tmp += self._criteria["mean_speed"]["w"] * (2 - s[12] - s[13]) / 2
            if "target_speed" in self._criteria.keys():
                tmp += self._criteria["target_speed"]["w"] * (
                    exponential_reward(
                        (s[12] + s[13]) / 2,
                        self._criteria["target_speed"].get("target", 0.5),
                        self._criteria["target_speed"].get("smoothness", 0.01),
                        self._criteria["target_speed"].get("bound", 0.8),
                        self._criteria["target_speed"].get("value_at_bound", 0.01),
                    )
                )
            if "negative_flow" in self._criteria.keys():
                if s[14] < 0 or s[15] < 0:
                    tmp -= self._criteria["negative_flow"]["w"]

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
            qs=None,
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

        CircularFluidNetwork._plot_reward_and_q(rewards, qs, title, save_path)

    @staticmethod
    def _plot_reward_and_q(
            rewards=None,
            qs=None,
            title=None,
            save_path=None
    ):
        rows = 0
        if rewards is not None:
            rows += 1
        if qs is not None:
            rows += 1

        if rows == 0:
            return

        fig, subplot_ax = plt.subplots(rows, 1, figsize=(8, rows * 8))
        if rewards is not None:
            if rows == 1:
                ax = subplot_ax
            else:
                ax = subplot_ax[0]
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

        if qs is not None:
            if rows == 1:
                ax = subplot_ax
            else:
                ax = subplot_ax[1]

            for q in qs:
                ax.plot(q)
            ax.set_xlabel("Action step")
            ax.set_ylabel("Q-value")

        if save_path is None:
            fig.show()
        else:
            path = str.replace(save_path, "Epoch", "Epoch_rewards")
            path = str.replace(path, "Final", "Final_rewards")
            fig.savefig(path + ".png")
        plt.close(fig)
