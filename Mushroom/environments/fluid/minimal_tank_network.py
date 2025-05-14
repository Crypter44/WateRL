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
            state_selector: List[int] = None,
            observation_selector: List[List[int]] = None,
            criteria: dict = None,
            labeled_step: bool = False,
            demand_curve: str = "tagesgang",
            multi_threaded_rendering: bool = True,
            sim_step_size: int = 10,
    ):
        self._criteria = criteria or {
            "target_opening": {
                "w": 1.0,
            }
        }

        self.warmup = False

        self.state_selector = state_selector or [0, 6, 9]
        self.observation_selector = observation_selector or [[0], [9]]

        state_space = spaces.Box(low=-500, high=500, shape=(len(self.state_selector),))
        observation_spaces = [
            spaces.Box(low=-500, high=500, shape=(len(obs),)) for obs in self.observation_selector
        ]
        action_spaces = [
            spaces.Box(low=0, high=1, shape=(1,)),  # pump speed
            spaces.Box(low=-1, high=1, shape=(1,))  # control of the tank
        ]
        stop_time = 86400.0
        self._step_size = sim_step_size
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
        self._current_sim_state, _ = self._get_simulation_state()

        sample = {
            "state": self._get_state(),
            "obs": self._get_observations(),
            "info": {},
        }
        return sample if self.labeled_step else self._get_observations()

    def step(self, action):
        action = np.clip(action, 0, 1)
        self.actions.append(action)

        if self.warmup:
            action[1] = np.random.uniform(0.9, 1)

        error = False
        sim_states = []
        for i in range(ControllerMinimalTank.CONTROL_STEP_INTERVAL // self._step_size):
            try:
                self.sim.do_simulation_step(action)
            except Exception as e:
                print(f"Error in simulation step: {e}")
                error = True
                break
            sim_states.append(self._get_simulation_state())

        reward = self._reward_fun(sim_states, action)
        self._current_sim_state, done = self._get_simulation_state()
        self.rewards.append(reward)

        absorbing = done or error

        step = {
            "state": self._get_state(),
            "obs": self._get_observations(),
            "rewards": [reward] * self._mdp_info.n_agents,
            "absorbing": absorbing,
            "info": {},
        }
        return step if self.labeled_step else (self._get_observations(), reward, absorbing, {})

    def _reward_fun(self, sim_states, action):
        reward = 0
        sim_states_to_use = sim_states[-15:]
        for s, _ in sim_states_to_use:
            tmp = 0

            if "demand" in self._criteria.keys():
                tmp += (
                        self._criteria["demand"]["w"] *
                        (
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
                x = max(s[2], s[8])
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
                action_to_reward = deepcopy(action)
                action_to_reward -= self._criteria["target_action"]["target"]
                tmp += self._criteria["target_action"]["w"] * exponential_reward(
                    np.max(np.abs(action_to_reward)),
                    0,
                    self._criteria["target_action"].get("smoothness", 0.01),
                    self._criteria["target_action"].get("bound", 0.8),
                    self._criteria["target_action"].get("value_at_bound", 0.01),
                )
            if "negative_flow" in self._criteria.keys():
                if s[4] < -1e-6:
                    tmp -= self._criteria["negative_flow"]["w"]
            if "power_per_flow" in self._criteria.keys():
                tmp -= self._criteria["power_per_flow"]["w"] * s[5] / (s[4] + 1)

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

    def _get_simulation_state(self) -> (np.ndarray, bool):
        sim_state, done = self.sim.get_current_state()
        try:
            controller_state = sim_state["control_api"]
            sim_state = np.array(controller_state)
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")

        return sim_state, done

    def _get_state(self) -> np.ndarray:
        state = np.array(self._current_sim_state)
        state = state[self.state_selector]
        return state

    def _get_observations(self) -> List:
        state = np.array(self._current_sim_state)
        obs = [
            state[obs_sel] for obs_sel in self.observation_selector
        ]
        return obs

    @staticmethod
    def _render_task(sim_data, rewards=None, actions=None, save_path=None, title=None):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
        valve_colors = ['#FFA500']
        pump_colors = ['#1E90FF', '#00BFFF']
        tank_colors = ['#00008B', '#4169E1', '#8A2BE2']
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
        ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.15))
        ax[0, 0].set_ylabel("Volume flow [m³/h]", color=valve_colors[0])
        ax[0, 0].set_ylim((0.15, 3.95))
        ax[0, 0].set_title("Valve 5")

        # Plot pump speed and power
        ax[1, 0].plot(
            sim_data["time"],
            sim_data["control_api.w_p_4"] * 10 / 13,
            color=pump_colors[0],
            label="Rotational speed",
            zorder=0,
            linewidth=2,
        )
        if actions is not None:
            ax[1, 0].plot(
                np.linspace(0, np.array(sim_data["time"])[-1], len(actions)),
                [a[0] for a in actions],
                color='black',
                label="Pump action",
                alpha=0.35,
                zorder=1,
                linestyle='--',
            )
        ax2 = ax[1, 0].twinx()
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
        ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.15))
        ax[1, 0].set_ylim((0, 1.01))
        ax[1, 0].set_ylabel("Rotational speed", color=pump_colors[0])
        ax[1, 0].set_title("Pump")

        # Plot Volume flow at pump
        ax[2, 0].plot(
            sim_data["time"],
            sim_data["water_network.V_flow_4"],
            color=pump_colors[1],
            label="Volume flow supplied by pump",
            zorder=2,
        )
        ax[2, 0].fill_between(
            sim_data["time"],
            sim_data["water_network.V_flow_4"],
            where=sim_data["water_network.V_flow_4"] < 0,
            color='red',
            alpha=0.3,
            label="Backwards flow",
        )
        ax[2, 0].plot(
            sim_data["time"],
            [0] * len(sim_data["time"]),
            color='red',
            linestyle='--',
            linewidth=1,
            zorder=1,
        )
        ax[2, 0].set_ylabel("Volume flow [m³/h]", color=pump_colors[1])
        ax[2, 0].set_title("Volume Flow through Pump")

        # Plot tank level
        ax[0, 1].plot(
            sim_data["time"],
            np.array(sim_data["water_network.level_tank_9"])-0.03,
            color=tank_colors[0],
            label="Level of the water in the tank",
        )
        # shade the tank level above 0
        ax[0, 1].fill_between(
            sim_data["time"],
            np.array(sim_data["water_network.level_tank_9"])-0.03,
            where=np.array(sim_data["water_network.level_tank_9"])-0.03 >= 0,
            color=tank_colors[0],
            alpha=0.3,
        )
        ax[0, 1].set_ylabel("Level [m]", color=tank_colors[0])
        ax[0, 1].set_title("Tank Level")

        # Plot inflow and outflow
        ax[1, 1].plot(
            sim_data["time"],
            sim_data["water_network.V_flow_7"],
            color=tank_colors[1],
            label="In/Outflow",
        )
        ax[1, 1].set_ylabel("Volume flow [m³/h]", color=tank_colors[1])
        ax[1, 1].set_ylim((-1, 1))
        ax[1, 1].set_title("Inflow/Outflow of Tank")

        # Plot tank control
        if actions is not None:
            ax[2, 1].plot(
                np.linspace(0, np.array(sim_data["time"])[-1], len(actions)),
                [a[1] for a in actions],
                color=tank_colors[2],
                label="Tank action",
            )
        else:
            ax[2, 1].plot(
                sim_data["time"],
                sim_data["control_api.w_v_7"],
                color=tank_colors[2],
                label="Tank action",
            )
        ax[2, 1].set_ylabel("Tank Valve Control", color=tank_colors[2])
        ax[2, 1].set_title("Tank Valve Control")
        ax[2, 1].set_ylim((0, 1.01))

        # Plot pressure at tank
        ax2 = ax[2, 1].twinx()
        ax2.plot(
            sim_data["time"],
            sim_data["water_network.p_rel_7"],
            color='gray',
            alpha=.9,
            label="Relative pressure of\ntank across its valve",
            linewidth=1,
        )
        ax2.set_ylabel("Pressure [bar]", color='gray')
        ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.15))

        for a in ax.flatten():
            a.set_xlabel("Time [h]")
            ticks = range(0, int(stop_time), 60 * 60 * 3)
            custom_labels = [i // 3600 for i in ticks]
            a.set_xticks(ticks)
            a.set_xticklabels(custom_labels)
            a.set_xlim((0, stop_time))
            a.grid(axis='x')
            a.legend(loc='upper left', bbox_to_anchor=(0, -0.15))

        fig.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.95,
            hspace=0.55,
            wspace=0.4
        )
        # fig.suptitle(title)

        if save_path is None:
            fig.show()
        else:
            fig.savefig(save_path + ".pdf")
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
            ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
            ax.set_xticks(range(0, int(stop_time), 60 * 60 * 3))
            ax.set_xticklabels([i // 3600 for i in range(0, int(stop_time), 60 * 60 * 3)])
            fig.subplots_adjust(
                left=0.05,
                bottom=0.12,
                right=0.95,
                top=1,
                hspace=0.45,
                wspace=0.4
            )
            if save_path is None:
                fig.show()
            else:
                path = str.replace(save_path, "Epoch", "Epoch_reward")
                path = str.replace(path, "Final", "Final_reward")
                fig.savefig(path + ".pdf")
            plt.close(fig)
