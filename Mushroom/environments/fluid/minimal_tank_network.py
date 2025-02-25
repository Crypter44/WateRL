import concurrent.futures
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.abstract_environments import AbstractFluidNetworkEnv
from Mushroom.utils.utils import exponential_reward, linear_reward
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.networks.tank_minimal.controller import ControllerMinimalTank
from Sofirpy.simulation import ManualStepSimulator


class MinimalTankNetwork(AbstractFluidNetworkEnv):
    def __init__(
            self,
            gamma: float = 0.99,
            criteria: dict = None,
            labeled_step: bool = False,
            demand_curve: str = "tagesgang",
            multi_threaded_rendering: bool = True,
    ):
        self._criteria = criteria or {
            "target_opening": {
                "w": 1.0,
            }
        }

        state_space = spaces.Box(low=-500, high=500, shape=(4,))
        observation_spaces = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # demand of the valve and level of the tank
            spaces.Box(low=-np.inf, high=np.inf, shape=(2,))  # for both agents
        ]
        action_spaces = [
            spaces.Box(low=0, high=1, shape=(1,)),  # pump speed
            spaces.Box(low=-1, high=1, shape=(1,))  # control of the tank
        ]
        stop_time = 86400.0
        self._step_size = 10
        sim = ManualStepSimulator(
            stop_time=stop_time,
            step_size=self._step_size,
            **get_minimal_tank_network_config(demand_curve),
            logging_step_size=self._step_size,
            get_units=True,
            verbose=False,
            ignore_warnings=False,
        )

        self.rewards = []
        self.actions = []

        # Configure the rendering of the environment
        self._multi_threaded_rendering = multi_threaded_rendering
        self._render_executor = concurrent.futures.ThreadPoolExecutor()

        super().__init__(
            state_space=state_space,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            fluid_network_simulator=sim,
            gamma=gamma,
            horizon=int(stop_time // ControllerMinimalTank.CONTROL_STEP_INTERVAL),
            stop_time=int(stop_time),
            n_agents=2,
            labeled_step=labeled_step
        )

    def reset(self, state=None) -> dict:
        self.rewards = []
        self.actions = []
        if state is not None:
            raise NotImplementedError("Resetting to a specific state is not supported.")

        self.sim.reset_simulation(self._stop_time, self._step_size, self._step_size)
        self._current_state, _ = self._get_current_state()

        sample = {
            "state": self._current_state[[0, 1, 6, 9]],
            "obs": self._get_observations(),
            "info": {},
        }
        return sample if self.labeled_step else self._get_observations()

    def step(self, action):
        action = np.clip(action, 0, 1)
        self.actions.append(action)

        error = False
        sim_states = []
        for i in range(ControllerMinimalTank.CONTROL_STEP_INTERVAL // self._step_size):
            try:
                self.sim.do_simulation_step(action)
            except Exception as e:
                print(f"Error in simulation step: {e}")
                error = True
                break
            sim_states.append(self._get_current_state())

        reward = self._reward_fun(sim_states, action)
        self._current_state, done = self._get_current_state()
        self.rewards.append(reward)

        absorbing = done or error

        step = {
            "state": self._current_state[[0, 1, 6, 9]],
            "obs": self._get_observations(),
            "rewards": [reward] * self._mdp_info.n_agents,
            "absorbing": absorbing,
            "info": {},
        }
        return step if self.labeled_step else (self._get_observations(), reward, absorbing, {})

    def _reward_fun(self, sim_states, action):
        reward = 0
        sim_states_to_use = sim_states[-1:]
        for s, _ in sim_states_to_use:
            tmp = 0

            if "demand" in self._criteria.keys():
                tmp += (
                        self._criteria["demand"]["w"] *
                        (
                                self._criteria["demand"].get("max", 1) -
                                self._criteria["demand"].get("min", 0)
                        ) *
                        exponential_reward(
                            s[0] - s[1],
                            0,
                            self._criteria["demand"].get("smoothness", 0.0001),
                            self._criteria["demand"].get("bound", 0.1),
                            self._criteria["demand"].get("value_at_bound", 0.01),
                        ) + self._criteria["demand"].get("min", 0)
                )
            if "demand_switch" in self._criteria.keys():
                diff = s[0] - s[1]
                if abs(diff) > self._criteria["demand_switch"]["tolerance"]:
                    tmp -= self._criteria["demand_switch"].get("neg_w", 5)
                else:
                    tmp += (
                            self._criteria["demand_switch"]["w"] *
                            exponential_reward(
                                diff,
                                0,
                                self._criteria["demand_switch"].get("smoothness", 0.0001),
                                self._criteria["demand_switch"]["tolerance"],
                                self._criteria["demand_switch"].get("value_at_bound", 0.01),
                            )
                    )
            if "target_opening" in self._criteria.keys():
                x = s[2]
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
            if "target_action" in self._criteria.keys():
                tmp += self._criteria["target_action"]["w"] * exponential_reward(
                    np.mean(action),
                    self._criteria["target_action"]["target"],
                    self._criteria["target_action"].get("smoothness", 0.01),
                    self._criteria["target_action"].get("bound", 0.8),
                    self._criteria["target_action"].get("value_at_bound", 0.01),
                )
            if "negative_flow" in self._criteria.keys():
                if s[4] < -1e-6:
                    tmp -= self._criteria["negative_flow"]["w"]

            reward += tmp / len(sim_states_to_use)

        return reward

    def render(self, save_path=None, title=None):
        if self.sim.is_done():
            sim_data, _ = self.sim.get_results()
            rewards = deepcopy(self.rewards)
            actions = deepcopy(self.actions)
            if self._multi_threaded_rendering:
                return self._render_executor.submit(self._render_task, sim_data, rewards, actions, save_path, title)
            else:
                return self._render_task(sim_data, rewards, actions, save_path, title)

    def stop_renderer(self):
        self._render_executor.shutdown()

    @staticmethod
    def _render_task(sim_data, rewards=None, actions=None, save_path=None, title=None):
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(27, 18))
        valve_colors = ['#FFA500']
        pump_colors = ['#1E90FF']
        tank_colors = ['#FF4500']
        stop_time = np.array(sim_data["time"])[-1]

        # Plot the valve
        ax[0, 0].plot(
            sim_data["time"],
            sim_data["control_api.w_v_5"],
            color=valve_colors[0],
            label="Demand",
            linestyle='--',
            linewidth=3,
        )
        ax[0, 0].plot(
            sim_data["time"],
            sim_data["water_network.V_flow_5"],
            color=valve_colors[0],
            label="Volume flow",
            linewidth=2,
        )
        ax[0, 0].fill_between(
            sim_data["time"],
            sim_data["control_api.w_v_5"],
            sim_data["water_network.V_flow_5"],
            where=sim_data["control_api.w_v_5"] > sim_data["water_network.V_flow_5"],
            color='red',
            alpha=0.3,
            label="Deviation from demand",
        )
        ax2 = ax[0, 0].twinx()
        ax2.plot(
            sim_data["time"],
            sim_data["water_network.u_v_5"],
            color='gray',
            alpha=.9,
            label="Opening",
            linewidth=1,
        )
        ax2.set_ylabel("Opening", color='gray')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.1))
        ax[0, 0].set_ylabel("Volume flow [m³/h]", color=valve_colors[0])
        ax[0, 0].set_ylim((0, 4.5))
        ax[0, 0].set_title("Valve 5")

        # Plot pump speed and power
        ax[0, 1].plot(
            sim_data["time"],
            sim_data["control_api.w_p_4"],
            color=pump_colors[0],
            label="Speed",
            zorder=0,
            linewidth=2,
        )
        if actions is not None:
            ax[0, 1].plot(
                np.linspace(0, np.array(sim_data["time"])[-1], len(actions)),
                [a[0] * 1.3 for a in actions],
                color='black',
                label="Pump action",
                alpha=0.35,
                zorder=1,
                linestyle='--',
            )
        ax2 = ax[0, 1].twinx()
        ax2.plot(
            sim_data["time"],
            sim_data["water_network.P_pum_4"],
            color='gray',
            alpha=.9,
            linewidth=1,
            zorder=1,
            label="Power consumption",
        )
        ax2.set_ylabel("Power consumption [W]", color='gray')
        ax2.set_ylim((0, 450))
        ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.1))
        ax[0, 1].set_ylim((0, 1.31))
        ax[0, 1].set_ylabel("Rotational speed", color=pump_colors[0])
        ax[0, 1].set_title("Pump")

        # Plot Volume flow at pump
        ax[0, 2].plot(
            sim_data["time"],
            sim_data["water_network.V_flow_4"],
            color=pump_colors[0],
            label="Volume flow",
            zorder=2,
        )
        ax[0, 2].fill_between(
            sim_data["time"],
            sim_data["water_network.V_flow_4"],
            where=sim_data["water_network.V_flow_4"] < 0,
            color='red',
            alpha=0.3,
            label="Negative flow",
        )
        ax[0, 2].plot(
            sim_data["time"],
            [0] * len(sim_data["time"]),
            color='red',
            linestyle='--',
            linewidth=1,
            zorder=1,
        )
        ax[0, 2].set_ylabel("Volume flow [m³/h]", color=pump_colors[0])
        ax[0, 2].set_title("Volume Flow through Pump")

        # Plot tank level
        ax[1, 0].plot(
            sim_data["time"],
            sim_data["water_network.level_tank_9"],
            color=pump_colors[0],
            label="Level",
        )
        # shade the tank level above 0
        ax[1, 0].fill_between(
            sim_data["time"],
            sim_data["water_network.level_tank_9"],
            where=sim_data["water_network.level_tank_9"] > 0,
            color=pump_colors[0],
            alpha=0.3,
        )
        ax[1, 0].set_ylabel("Level [m]", color=pump_colors[0])
        ax[1, 0].set_title("Tank Level")

        # Plot inflow and outflow
        ax[1, 1].plot(
            sim_data["time"],
            sim_data["water_network.V_flow_7"],
            color='green',
            label="In/Outflow",
        )
        ax[1, 1].set_ylabel("Volume flow [m³/h]", color='green')
        ax[1, 1].set_title("Inflow/Outflow of Tank")

        # Plot tank control
        ax[1, 2].plot(
            sim_data["time"],
            sim_data["control_api.w_v_7"],
            color=tank_colors[0],
            label="Tank action (corrected)",
        )
        if actions is not None:
            ax[1, 2].plot(
                np.linspace(0, np.array(sim_data["time"])[-1], len(actions)),
                [a[1] for a in actions],
                color='black',
                label="Tank action",
                alpha=0.35,
                linestyle='--',
            )
        ax[1, 2].set_ylabel("Tank Valve Control", color=tank_colors[0])
        ax[1, 2].set_title("Tank Valve Control")
        ax[1, 2].set_ylim((0, 1.01))

        for a in ax.flatten():
            a.set_xlabel("Time [h]")
            ticks = range(0, int(stop_time), 60 * 60 * 3)
            custom_labels = [i // 3600 for i in ticks]
            a.set_xticks(ticks)
            a.set_xticklabels(custom_labels)
            a.set_xlim((0, stop_time))
            a.grid(axis='x')
            a.legend(loc='upper left', bbox_to_anchor=(0, -0.1))

        fig.subplots_adjust(
            left=0.05,
            bottom=0.12,
            right=0.95,
            top=0.9,
            hspace=0.4,
            wspace=0.4
        )
        fig.suptitle(title)

        if save_path is None:
            fig.show()
        else:
            fig.savefig(save_path + ".png")
        plt.close(fig)

        if rewards is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.plot(
                np.linspace(0, np.array(sim_data["time"])[-1], len(rewards)),
                rewards,
                color='blue',
                label="Reward",
            )
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Reward")
            ax.set_title("Reward")
            ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1))
            ax.set_xticks(range(0, int(stop_time), 60 * 60 * 3))
            ax.set_xticklabels([i // 3600 for i in range(0, int(stop_time), 60 * 60 * 3)])
            fig.subplots_adjust(
                left=0.05,
                bottom=0.12,
                right=0.95,
                top=0.9,
                hspace=0.4,
                wspace=0.4
            )
            if save_path is None:
                fig.show()
            else:
                path = str.replace(save_path, "Epoch", "Epoch_reward")
                path = str.replace(path, "Final", "Final_reward")
                fig.savefig(path + ".png")
            plt.close(fig)

    def _get_current_state(self) -> (np.ndarray, bool):
        sim_state, done = self.sim.get_current_state()
        try:
            controller_state = sim_state["control_api"]
            sim_state = np.array(controller_state)
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")

        return sim_state, done

    def _get_observations(self) -> List:
        state = self._current_state
        obs = [
            state[[0]],
            state[[6, 9]],
        ]
        return obs
