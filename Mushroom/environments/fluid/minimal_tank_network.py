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
            criteria: dict = None,
            labeled_step: bool = False
    ):
        self._criteria = criteria or {
            "target_opening": {
                "w": 1.0,
            }
        }

        state_space = spaces.Box(low=-500, high=500, shape=(10,))
        observation_spaces = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),  # demand of the valve and level of the tank
            spaces.Box(low=-np.inf, high=np.inf, shape=(2,))  # for both agents
        ]
        action_spaces = [
            spaces.Box(low=0, high=1, shape=(1,)),  # pump speed
            spaces.Box(low=-1, high=1, shape=(1,))  # control of the tank
        ]
        stop_time = 50000.0
        sim = ManualStepSimulator(
            stop_time=stop_time,
            step_size=1.0,
            **get_minimal_tank_network_config(),
            logging_step_size=1,
            get_units=True,
            verbose=False,
            ignore_warnings=True,
        )

        self.rewards = []

        # Configure the rendering of the environment
        self._render_executor = concurrent.futures.ThreadPoolExecutor()

        super().__init__(
            state_space=state_space,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            fluid_network_simulator=sim,
            gamma=0.99,
            horizon=stop_time // ControllerMinimalTank.CONTROL_STEP_INTERVAL,
            stop_time=int(stop_time),
            n_agents=2,
            labeled_step=labeled_step
        )

    def reset(self, state=None) -> dict:
        self.rewards = []
        return super().reset(state)

    def step(self, action):
        np.clip(action, 0, 1)

        error = False
        sim_states = []
        for i in range(ControllerMinimalTank.CONTROL_STEP_INTERVAL):
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
            "state": self._current_state,
            "obs": self._get_observations(),
            "rewards": [reward] * self._mdp_info.n_agents,
            "absorbing": absorbing,
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
                        exponential_reward(
                            s[0] - s[1],
                            0,
                            self._criteria["demand"].get("smoothness", 0.0001),
                            self._criteria["demand"].get("bound", 0.1),
                            self._criteria["demand"].get("value_at_bound", 0.01),
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
            if "negative_flow" in self._criteria.keys():
                if s[4] < -1e-6:
                    tmp -= self._criteria["negative_flow"]["w"]

            reward += tmp / len(sim_states_to_use)

        return reward

    def render(self, save_path=None, title=None):
        sim_data, _ = self.sim.get_results()
        rewards = deepcopy(self.rewards)
        return self._render_executor.submit(self._render_task, sim_data, rewards, save_path, title)

    def stop_renderer(self):
        self._render_executor.shutdown()

    @staticmethod
    def _render_task(sim_data, rewards=None, save_path=None, title=None):
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        valve_colors = ['#FFA500']
        pump_colors = ['#1E90FF']
        tank_colors = ['#FF4500']

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
        ax[0, 0].set_xlabel("Time [s]")
        ax[0, 0].set_ylabel("Volume flow [m³/h]", color=valve_colors[0])
        ax[0, 0].set_ylim((0, 3.5))
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
        ax2.set_ylim((0, 200))
        ax[0, 1].set_ylim((0, 1))
        ax[0, 1].set_xlabel("Time [s]")
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
        ax[0, 2].set_xlabel("Time [s]")
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
        ax[1, 0].set_xlabel("Time [s]")
        ax[1, 0].set_ylabel("Level [m]", color=pump_colors[0])
        ax[1, 0].set_title("Tank Level")

        # Plot inflow and outflow
        ax[1, 1].plot(
            sim_data["time"],
            sim_data["water_network.V_flow_7"],
            color='green',
            label="In/Outflow",
        )
        ax[1, 1].set_xlabel("Time [s]")
        ax[1, 1].set_ylabel("Volume flow [m³/h]", color='green')
        ax[1, 1].set_title("Inflow/Outflow of Tank")

        # Plot tank control
        ax[1, 2].plot(
            sim_data["time"],
            sim_data["control_api.w_v_7"],
            color=tank_colors[0],
            label="Control",
        )
        ax[1, 2].set_xlabel("Time [s]")
        ax[1, 2].set_ylabel("Tank Valve Control", color=tank_colors[0])
        ax[1, 2].set_title("Tank Valve Control")

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

    def _get_current_state(self) -> (np.ndarray, bool):
        sim_state, done = self.sim.get_current_state()
        try:
            controller_state = sim_state["control_api"]
            return np.array(controller_state), done
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")

    def _get_observations(self) -> List:
        return [self._current_state[:2] for _ in range(self._mdp_info.n_agents)]
