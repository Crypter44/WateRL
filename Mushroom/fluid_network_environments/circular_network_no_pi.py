import json
from pathlib import Path

import numpy as np
from mushroom_rl.utils import spaces

from Mushroom.fluid_network_environments.circular_network import CircularFluidNetwork
from Sofirpy.networks.control_api import ControlApiCircularNoPI
from Sofirpy.simulation import ManualStepSimulator

working_dir = Path(__file__).parent.parent.parent

fmu_dir_path = working_dir / "Fluid_Model" / "circular_water_network_wo_PI"

fmu_path = fmu_dir_path / "mini_circular_water_network_wo_PI.fmu"

agent_config_path = fmu_dir_path / "mini_circular_water_network_wo_PI.json"

connections_config_path = fmu_dir_path / "mini_circular_water_network_wo_PI_connections_config.json"

logging_config_path = fmu_dir_path / "mini_circular_water_network_wo_PI_parameters_to_log.json"

# create interface of multi-agent system to FMU
model_classes = {"control_api": ControlApiCircularNoPI}
fmu_paths = {"water_network": str(fmu_path)}

with open(connections_config_path) as connections_config_json:
    connections_config = json.load(connections_config_json)

with open(logging_config_path) as logging_config_json:
    parameters_to_log = json.load(logging_config_json)


class CircularFluidNetworkWithoutPI(CircularFluidNetwork):
    def __init__(
            self,
            observation_space=spaces.Box(low=-10, high=10, shape=(4,)),
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            fluid_network_simulator=ManualStepSimulator(
                stop_time=50,
                step_size=1,
                fmu_paths=fmu_paths,
                model_classes=model_classes,
                connections_config=connections_config,
                parameters_to_log=parameters_to_log,
                logging_step_size=1,
                get_units=False,
                verbose=False,
            ),
            gamma: float = 0.99,
            power_penalty: float = 0.01,
            penalize_negative_flow: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            fluid_network_simulator=fluid_network_simulator,
            gamma=gamma,
            power_penalty=power_penalty,
            penalize_negative_flow=penalize_negative_flow,
            horizon=50,
        )

    def render(self, title=None, save_path=None):
        results = self.sim.get_results()
        valves = [2, 3, 5, 6]
        pumps = [1, 4]
        self.plot_valve_and_pump_data(
            results["time"],
            valves=valves,
            valve_openings=[results[f"control_api.w_v_{v}"] for v in valves],
            valve_demands=[results[f"control_api.demand_v_{v}"] for v in valves],
            valve_flows=[results[f"water_network.V_flow_{v}"] for v in valves],
            pumps=pumps,
            pump_speeds=[results[f"control_api.w_p_{p}"] for p in pumps],
            pump_powers=[results[f"water_network.V_flow_{p}"] for p in pumps],
            pump_flows=[results[f"water_network.V_flow_{p}"] for p in pumps],
            title=title,
            save_path=save_path,
        )

    def _get_current_state(self):
        """
        Return the observable state of the environment.
        """
        state, absorbing = self._get_current_simulation_state()
        return state[:4].reshape((4, 1)), absorbing

    def local_observation_space(self, agent_index: int):
        return spaces.Box(low=-10, high=10, shape=(1,))

    def local_action_space(self, agent_index: int):
        return spaces.Box(low=0, high=1, shape=(1,))

    def _reward_fun(self, state: np.ndarray, action: np.ndarray, sim_states: list):
        return self._bound_reward(state, action, sim_states)
